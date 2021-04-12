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
        'kaldi_pitch': ('processor', 'KaldiPitchProcessor'),
        'kaldi_pitch_post': ('processor', 'KaldiPitchPostProcessor'),
        'crepe_pitch': ('processor', 'CrepePitchProcessor'),
        'crepe_pitch_post': ('processor', 'CrepePitchPostProcessor'),
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

        self._check_utterances()

        # store the metadata because we need to access the sample rate
        # for processors instanciation
        audio_files = set(utt.audio_file for utt in utterances)
        self._audio_metadata = {}
        for audio in audio_files:
            log.debug('scanning %s', audio)
            self._audio_metadata[audio] = Audio.scan(audio)

        # make sure all the audio files are compatible with the pipeline
        log.info('scanning %s utterances...', len(self._utterances))
        self._check_audio_files()

        # the features type to be extracted
        self.features = [
            k for k in self.config.keys() if k in self.valid_features][0]

        # get some framing parameters constant for all processors
        # (retrieve them from a features processor instance)
        proc = self.get_features_processor(next(iter(self.utterances)))
        self.frame_length = proc.frame_length
        self.frame_shift = proc.frame_shift

        # if CMVN by speaker, instanciate a CMVN processor by speaker
        # here, else instanciate a processor per utterance
        if 'cmvn' in self.config:
            if self.config['cmvn']['by_speaker']:
                self._cmvn_processors = {
                    spk: self.get_processor_class('cmvn')(proc.ndims)
                    for spk in set(utt.speaker for utt in self.utterances)}
            else:
                self._cmvn_processors = {
                    utt.name: self.get_processor_class('cmvn')(proc.ndims)
                    for utt in self.utterances}

    @property
    def config(self):
        """The pipeline configuration"""
        return self._config

    @property
    def utterances(self):
        """Utterance on which to apply the pipeline"""
        return self._utterances

    @property
    def warps(self):
        """VTLN waprs of the utterances (optional)"""
        return self._warps

    @warps.setter
    def warps(self, value):
        self._warps = value

    @property
    def audio_metadata(self):
        """Audio metadata corresponding to utterances"""
        return self._audio_metadata

    def _check_utterances(self):
        """Ensures the configuration is compatible with utterances

        aises a ValueError on error. If all is good, silently returns None.

        """
        # ensures speakers info provided if cmvn by speaker is requested
        if 'cmvn' not in self.config or not self.config['cmvn']['by_speaker']:
            cmvn_by_speaker = False
        else:  # config['cmvn']['by_speaker'] exists and is True
            assert self.config['cmvn']['by_speaker']
            cmvn_by_speaker = True

        if cmvn_by_speaker and not self.utterances.has_speakers():
            raise ValueError(
                'cmvn normalization by speaker requested '
                'but no speaker information provided')
        if not cmvn_by_speaker and self.utterances.has_speakers():
            self.log.warning(
                'speakers information is provided but will not be used '
                '(CMVN%s disabled)',
                ' by speaker' if 'cmvn' in self.config else '')

    def _check_audio_files(self):
        """Ensures all the audio files are compatible with the pipeline"""
        # log the total duration and the number of speakers
        total_duration = self.utterances.duration()
        speakers = (
            '' if not self.utterances.has_speakers()
            else ' from {} speakers'.format(
                len(set(utt.speaker for utt in self.utterances))))

        self.log.info(
            'get %s utterances%s in %s audio files, total duration: %s',
            len(self.utterances), speakers, len(self.audio_metadata),
            datetime.timedelta(seconds=total_duration))

        # make sure all audio files are mono
        if not all(w.nchannels == 1 for w in self.audio_metadata.values()):
            raise ValueError('all audio files are not mono')

        # check the sample rate (warning if all the audiofles are not at the
        # same sample rate)
        samplerates = set(w.sample_rate for w in self.audio_metadata.values())
        if len(samplerates) > 1:
            self.log.warning(
                'several sample rates found in audio files: %s, features '
                'extraction pipeline will work but this may not be a good '
                'idea to work on heterogeneous data',
                ', '.join(str(s) + 'Hz' for s in samplerates))

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

        if 'rastaplp' in name:
            name = 'rasta_plp'
        elif 'crepe_pitch' in name:
            # crepe pitch (post)processor
            name = 'pitch_crepe'
        elif 'kaldi_pitch' in name:
            # kaldi pitch (post)processor
            name = 'pitch_kaldi'
        elif name == 'sliding_window_cmvn':
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
        audio = utterance.load_audio()

        if self.features == 'bottleneck':
            # resample here the signal (this avoid bugs if one part of the
            # pipeline on 8k and the other on 16k), then update the metadata
            # for the audio to be used by the rest of the pipeline
            self.log.debug(
                'resampling audio from %dHz@%db to %dHz@%db',
                audio.sample_rate, audio.dtype.itemsize * 8, 8000, 16)

            audio = audio.resample(8000).astype(np.int16)
            self._audio_metadata[utterance.audio_file] = (
                Audio._metadata(
                    audio.nchannels, audio.sample_rate,
                    audio.nsamples, audio.duration))
        return audio

    def get_features_processor(self, utterance):
        """Instanciates and returns a features extraction processor"""
        proc = self.get_processor_class(self.features)(
            **self.config[self.features])

        try:
            proc.sample_rate = self.audio_metadata[
                utterance.audio_file].sample_rate
        except AttributeError:
            # bottleneck does not support changing sample rate
            pass
        return self._set_logger(proc)

    def get_energy_processor(self, utterance):
        """Instanciates and returns an energy processor"""
        proc = self.get_processor_class('energy')()
        proc.frame_length = self.frame_length
        proc.frame_shift = self.frame_shift
        proc.sample_rate = self._audio_metadata[
            utterance.audio_file].sample_rate
        return self._set_logger(proc)

    def get_vad_processor(self, _):
        """Instanciates and returns a VAD processor"""
        return self._set_logger(
            self.get_processor_class('vad')(**self.config['cmvn']['vad']))

    def get_cmvn_processor(self, utterance):
        """Instanciates and returns a CMVN processor"""
        if self.config['cmvn']['by_speaker']:
            speaker = utterance.speaker
            return self._cmvn_processors[speaker]

        return self._set_logger(self._cmvn_processors[utterance.name])

    def get_pitch_processor(self, utterance):
        """Instanciates and returns a pitch processor"""
        params = {k: v for k, v in self.config['pitch'].items()
                  if k not in ('processor', 'postprocessing')}
        params['sample_rate'] = self._audio_metadata[
            utterance.audio_file].sample_rate
        params['frame_shift'] = self.frame_shift
        params['frame_length'] = self.frame_length

        # fall back to kaldi or crepe processor according to config
        name = 'kaldi_pitch'
        if self.config['pitch']['processor'] == 'crepe':
            name = name.replace('kaldi', 'crepe')
            del params['sample_rate']

        return self._set_logger(self.get_processor_class(name)(**params))

    def get_pitch_post_processor(self, _):
        """Instanciates and returns a pitch post-processor"""
        # fall back to kaldi or crepe post-processor according to config
        name = 'kaldi_pitch_post'
        if self.config['pitch']['processor'] == 'crepe':
            name = name.replace('kaldi', 'crepe')

        return self._set_logger(
            self.get_processor_class(name)(
                **self.config['pitch']['postprocessing']))

    def get_delta_processor(self, _):
        """Instanciates and returns a delta processor"""
        return self._set_logger(
            self.get_processor_class('delta')(**self.config['delta']))

    def get_vtln_processor(self, _):
        """Instanciates and returns a VTLN processor"""
        return self._set_logger(self.get_processor_class('vtln')(
            **self.config['vtln']))

    def get_warp(self, utterance):
        """Returns the VTLN warp associated to this utterance"""
        return (
            1 if utterance.name not in self.warps
            else self.warps[utterance.name])
