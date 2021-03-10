"""Handles the instanciation of processors for the pipeline"""

import datetime
import importlib
import re
import numpy as np

from shennong.audio import Audio
from shennong.logger import get_logger


class PipelineManager:
    """This class handles the instanciation of processors for the pipeline

    Instanciation is the "hard part" because it relies on several
    parameters (CMVN or not, by speaker or not, at which sample rate,
    etc...). All this mechanics is abstracted by this class.

    """
    valid_features = [
        'mfcc', 'plp', 'filterbank', 'bottleneck', 'rastaplp', 'spectrogram']
    """The main features available in shennong, excluding post-processing"""

    valid_processors = {
        'bottleneck': ('processor', 'BottleneckProcessor'),
        'energy': ('processor', 'EnergyProcessor'),
        'filterbank': ('processor', 'FilterbankProcessor'),
        'mfcc': ('processor', 'MfccProcessor'),
        'pitch': ('processor', 'PitchProcessor'),
        'pitch_post': ('processor', 'PitchPostProcessor'),
        'plp': ('processor', 'PlpProcessor'),
        'rastaplp': ('processor', 'RastaPlpProcessor'),
        'spectrogram': ('processor', 'SpectrogramProcessor'),
        'ubm': ('processor', 'DiagUbmProcessor'),
        'vtln': ('processor', 'VtlnProcessor'),
        'cmvn': ('postprocessor', 'CmvnPostProcessor'),
        'delta': ('postprocessor', 'DeltaPostProcessor'),
        'sliding_window_cmvn':
            ('postprocessor', 'SlidingWindowCmvnPostProcessor'),
        'vad': ('postprocessor', 'VadPostProcessor')}
    """The features processors as a dict {name: (module, class)}"""

    def __init__(self, config, utterances,
                 log=get_logger('manager', 'warning')):
        self._config = config
        self._utterances = utterances
        self._warps = {}
        self.log = log

        # the list of speakers
        self._speakers = set(u.speaker for u in self.utterances.values())
        if self._speakers == {None}:
            self._speakers = None
        self._check_speakers()

        # store the metadata because we need to access the sample rate
        # for processors instanciation
        wavs = set(u.file for u in utterances.values())
        self._wavs_metadata = {w: Audio.scan(w) for w in wavs}

        # make sure all the wavs are compatible with the pipeline
        log.info('scanning %s utterances...', len(self._utterances))
        self._check_wavs()

        # the features type to be extracted
        self.features = [
            k for k in self.config.keys() if k in self.valid_features][0]

        # get some framing parameters constant for all processors
        # (retrieve them from a features processor instance)
        proc = self.get_features_processor(next(iter(self.utterances.keys())))
        self.frame_length = proc.frame_length
        self.frame_shift = proc.frame_shift

        # if CMVN by speaker, instanciate a CMVN processor by speaker
        # here, else instanciate a processor per utterance
        if 'cmvn' in self.config:
            if self.config['cmvn']['by_speaker']:
                self._cmvn_processors = {
                    spk: self.get_processor_class('cmvn')(proc.ndims)
                    for spk in self.speakers}
            else:
                self._cmvn_processors = {
                    utt: self.get_processor_class('cmvn')(proc.ndims)
                    for utt in self.utterances}

    @property
    def config(self):
        return self._config

    @property
    def utterances(self):
        return self._utterances

    @property
    def speakers(self):
        return self._speakers

    @property
    def warps(self):
        return self._warps

    @warps.setter
    def warps(self, value):
        self._warps = value

    @property
    def wavs_metadata(self):
        return self._wavs_metadata

    def _check_speakers(self):
        """Ensures the configuration is compatible with speakers information

        On any error raises a ValueError. Logs a warning if speakers
        information is provided but not used by the pipeline. If all is
        good, silently returns None.

        """
        # ensures speakers info provided if cmvn by speaker is requested
        if 'cmvn' not in self.config or not self.config['cmvn']['by_speaker']:
            cmvn_by_speaker = False
        else:  # config['cmvn']['by_speaker'] exists and is True
            assert self.config['cmvn']['by_speaker']
            cmvn_by_speaker = True

        if cmvn_by_speaker and not self.speakers:
            raise ValueError(
                'cmvn normalization by speaker requested '
                'but no speaker information provided')
        if not cmvn_by_speaker and self.speakers:
            self.log.warning(
                'speakers information is provided but will not be used '
                '(CMVN%s disabled)',
                ' by speaker' if 'cmvn' in self.config else '')

    def _check_wavs(self):
        """Ensures all the wav files are compatible with the pipeline"""
        # log the total duration and the number of speakers
        total_duration = sum(w.duration for w in self._wavs_metadata.values())
        # nspeakers = len(self.speakers or self.utterances)
        speakers = ('' if not self.speakers
                    else ' from {} speakers'.format(len(self.speakers)))
        self.log.info(
            'get %s utterances%s in %s wavs, total wavs duration: %s',
            len(self.utterances), speakers, len(self._wavs_metadata),
            datetime.timedelta(seconds=total_duration))

        # make sure all wavs are mono
        if not all(w.nchannels == 1 for w in self._wavs_metadata.values()):
            raise ValueError('all wav files are not mono')

        # check the sample rate (warning if all the wavs are not at the
        # same sample rate)
        samplerates = set(w.sample_rate for w in self._wavs_metadata.values())
        if len(samplerates) > 1:
            self.log.warning(
                'several sample rates found in wav files: %s, features '
                'extraction pipeline will work but this may not be a good '
                'idea to work on heterogeneous data',
                ', '.join(str(s) + 'Hz' for s in samplerates))

        # ensure all (tstart, tstop) pairs are valid (numbers and
        # tstart < tstop)
        tstamps = [
            (w.tstart, w.tstop, w.file) for w in self.utterances.values()]
        for (tstart, tstop, wfile) in tstamps:
            if tstart is not None and tstart > tstop:
                raise ValueError(
                    'timestamps are not in increasing order for {}: '
                    '{} >= {}'.format(wfile, tstart, tstop))

    def _set_logger(self, processor):
        processor.log.setLevel(self.log.getEffectiveLevel())
        return processor

    @classmethod
    def get_processor_class(cls, name):
        """Returns the (post)processor class given its `name`

        This function enables dynamic import of processors classes to
        avoid a big list of useless imports. Raises a ValueError if
        the `name` is not valid or the module/class cannot be
        imported.

        """
        try:
            _module, _class = cls.valid_processors[name]
        except KeyError:
            raise ValueError('invalid processor "{}"'.format(name))
        if name == 'pitch_post':
            name = 'pitch'
        if name == 'sliding_window_cmvn':
            name = 'cmvn'

        module = 'shennong.{}.{}'.format(_module, name)
        try:
            module = importlib.import_module(module)
        except ModuleNotFoundError:  # pragma: nocover
            raise ValueError('cannot import module "{}"'.format(module))

        try:
            return module.__dict__[_class]
        except KeyError:    # pragma: nocover
            raise ValueError(
                'cannot find class "{}" in module {}'.format(_class, module))

    @classmethod
    def get_processor_params(cls, name):
        """Returns the processors parameters given its `name`

        Get the processors class using :func:`get_processor_class`,
        instanciate it and returns its parameters with default values
        as a dict {param: value}.

        """
        return cls.get_processor_class(name)().get_params()

    @classmethod
    def get_docstring(cls, processor, param, default):
        """Returns the docstring of a given processor's parameter"""
        # extract the raw docstring
        docstring = getattr(
            cls.get_processor_class(processor), param).__doc__ or ''

        # postprocess to adapt Python docstring to the YAML output
        # (also adds default value)
        docstring = re.sub(r'\n\n', '. ', docstring)
        docstring = re.sub(r'\n', ' ', docstring)
        docstring = re.sub(r'`', '', docstring)
        docstring = re.sub(':func:', '', docstring)
        docstring += '. Default is {}.'.format(default)
        docstring = re.sub(r'\.+', '.', docstring)
        docstring = re.sub(r' +', ' ', docstring)
        docstring = re.sub(r'\. \.', '.', docstring)

        return docstring.strip()

    def get_audio(self, utterance):
        """Returns the audio data for that `utterance`"""
        utt = self.utterances[utterance]
        audio = Audio.load(utt.file)
        if utt.tstart is not None:
            assert utt.tstop > utt.tstart
            audio = audio.segment([(utt.tstart, utt.tstop)])[0]

        if self.features == 'bottleneck':
            # resample here the signal (this avoid bugs if one part of
            # the pipeline on 8k and the other on 16k), then update
            # the metadata for the wav to be used by the rest of the
            # pipeline
            self.log.debug(
                'resampling audio from %dHz@%db to %dHz@%db',
                audio.sample_rate, audio.dtype.itemsize * 8, 8000, 16)

            audio = audio.resample(8000).astype(np.int16)
            self._wavs_metadata[self.utterances[utterance].file] = (
                Audio._metawav(
                    audio.nchannels, audio.sample_rate,
                    audio.nsamples, audio.duration))
        return audio

    def get_features_processor(self, utterance):
        """Instanciates and returns a features extraction processor"""
        wav = self.utterances[utterance].file
        proc = self.get_processor_class(self.features)(
            **self.config[self.features])

        try:
            proc.sample_rate = self._wavs_metadata[wav].sample_rate
        except AttributeError:
            # bottleneck does not support changing sample rate
            pass
        return self._set_logger(proc)

    def get_energy_processor(self, utterance):
        """Instanciates and returns an energy processor"""
        wav = self.utterances[utterance].file
        proc = self.get_processor_class('energy')()
        proc.frame_length = self.frame_length
        proc.frame_shift = self.frame_shift
        proc.sample_rate = self._wavs_metadata[wav].sample_rate
        return self._set_logger(proc)

    def get_vad_processor(self, utterance):
        """Instanciates and returns a VAD processor"""
        return self._set_logger(
            self.get_processor_class('vad')(**self.config['cmvn']['vad']))

    def get_cmvn_processor(self, utterance):
        """Instanciates and returns a CMVN processor"""
        if self.config['cmvn']['by_speaker']:
            speaker = self.utterances[utterance].speaker
            return self._cmvn_processors[speaker]

        return self._set_logger(self._cmvn_processors[utterance])

    def get_sliding_window_cmvn_processor(self, utterrance):
        """Instanciates and returns a sliding-window CMVN processor"""
        return self._set_logger(
            self.get_processor_class('sliding_window_cmvn')(
                **self.config['sliding_window_cmvn']))

    def get_pitch_processor(self, utterance):
        """Instanciates and returns a pitch processor"""
        wav = self.utterances[utterance].file
        params = {k: v for k, v in self.config['pitch'].items()
                  if k != 'postprocessing'}
        params['sample_rate'] = self._wavs_metadata[wav].sample_rate
        params['frame_shift'] = self.frame_shift
        params['frame_length'] = self.frame_length

        return self._set_logger(self.get_processor_class('pitch')(**params))

    def get_pitch_post_processor(self, utterance):
        """Instanciates and returns a pitch post-processor"""
        return self._set_logger(
            self.get_processor_class('pitch_post')(
                **self.config['pitch']['postprocessing']))

    def get_delta_processor(self, utterance):
        """Instanciates and returns a delta processor"""
        return self._set_logger(
            self.get_processor_class('delta')(**self.config['delta']))

    def get_vtln_processor(self, utterance):
        """Instanciates and returns a VTLN processor"""
        return self._set_logger(self.get_processor_class('vtln')(
            **self.config['vtln']))

    def get_warp(self, utterance):
        """Returns the VTLN warp associated to this utterance"""
        return 1 if utterance not in self.warps else self.warps[utterance]
