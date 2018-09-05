"""Extract pitch from an audio (speech) signal

Uses the Kaldi implementation of pitch extraction.

References
----------

 "A Pitch Extraction Algorithm Tuned for Automatic Speech
 Recognition", Pegah Ghahremani, Bagher BabaAli, Daniel Povey,
 Korbinian Riedhammer, Jan Trmal and Sanjeev Khudanpur, ICASSP 2014.

 http://kaldi-asr.org/doc/pitch-functions_8h.html

"""

from kaldi.feat import pitch
from kaldi.matrix import SubVector, SubMatrix


class PitchProcessor(object):
    def __init__(self, sample_rate=16000, frame_shift=10.0,
                 frame_length=25.0, min_f0=50, max_f0=400,
                 soft_min_f0=10, penalty_factor=0.1,
                 lowpass_cutoff=1000, resample_freq=4000,
                 delta_pitch=0.005, nccf_ballast=7000,
                 lowpass_filter_width=1, upsample_filter_width=5):
        self._options = pitch.PitchExtractionOptions()
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.soft_min_f0 = soft_min_f0
        self.penalty_factor = penalty_factor
        self.lowpass_cutoff = lowpass_cutoff
        self.resample_freq = resample_freq
        self.delta_pitch = delta_pitch
        self.nccf_ballast = nccf_ballast
        self.lowpass_filter_width = lowpass_filter_width
        self.upsample_filter_width = upsample_filter_width

    @property
    def sample_rate(self):
        """Waveform data sample frequency in Hertz"""
        return self._options.samp_freq

    @sample_rate.setter
    def sample_rate(self, value):
        self._options.samp_freq = value

    @property
    def frame_shift(self):
        """Frame shift in milliseconds"""
        return self._options.frame_shift_ms

    @frame_shift.setter
    def frame_shift(self, value):
        self._options.frame_shift_ms = value

    @property
    def frame_length(self):
        """Frame length in milliseconds"""
        return self._options.frame_length_ms

    @frame_length.setter
    def frame_length(self, value):
        self._options.frame_length_ms = value

    @property
    def min_f0(self):
        """Minimum F0 to search for in Hertz"""
        return self._options.min_f0

    @min_f0.setter
    def min_f0(self, value):
        self._options.min_f0 = value

    @property
    def max_f0(self):
        """Maximum F0 to search for in Hertz"""
        return self._options.max_f0

    @max_f0.setter
    def max_f0(self, value):
        self._options.max_f0 = value

    @property
    def soft_min_f0(self):
        """Minimum F0 to search, applied in soft way, in Hertz

        Must not exceed `min_f0`

        """
        return self._options.soft_min_f0

    @soft_min_f0.setter
    def soft_min_f0(self, value):
        self._options.soft_min_f0 = value

    @property
    def penalty_factor(self):
        """Cost factor for F0 change"""
        return self._options.penalty_factor

    @penalty_factor.setter
    def penalty_factor(self, value):
        self._options.penalty_factor = value

    @property
    def lowpass_cutoff(self):
        """Cutoff frequency for low-pass filter, in Hertz"""
        return self._options.lowpass_cutoff

    @lowpass_cutoff.setter
    def lowpass_cutoff(self, value):
        self._options.lowpass_cutoff = value

    @property
    def resample_freq(self):
        """Frequency that we down-sample the signal to, in Hertz

        Must be more than twice `lowpass_cutoff`

        """
        return self._option.resample_freq

    @resample_freq.setter
    def resample_freq(self, value):
        self._options.resample_freq = value

    @property
    def delta_pitch(self):
        """Smallest relative change in pitch that the algorithm measures"""
        return self._options.delta_pitch

    @delta_pitch.setter
    def delta_pitch(self, value):
        self._options.delta_pitch = value

    @property
    def nccf_ballast(self):
        """Increasing this factor reduces NCCF for quiet frames

        This helps ensuring pitch continuity in unvoiced regions

        """
        return self._options.nccf_ballast

    @nccf_ballast.setter
    def nccf_ballast(self, value):
        self._options.nccf_ballast = value

    @property
    def lowpass_filter_width(self):
        """Integer that determines filter width of lowpass filter

        More gives sharper filter

        """
        return self._options.lowpass_filter_width

    @lowpass_filter_width.setter
    def lowpass_filter_width(self, value):
        self._options.lowpass_filter_width = value

    @property
    def upsample_filter_width(self):
        """Integer that determines filter width when upsampling NCCF"""
        return self._options.upsample_filter_width

    @upsample_filter_width.setter
    def upsample_filter_width(self, value):
        self._options.upsample_filter_width = value

    def compute(self, signal):
        return SubMatrix(pitch.compute_kaldi_pitch(
            self._options, SubVector(signal))).numpy()


class PitchPostProcessor(object):
    def __init__(self, pitch_scale=2.0, pov_scale=2.0, pov_offset=0.0,
                 delta_pitch_scale=10.0,
                 delta_pitch_noise_stddev=0.005,
                 normalization_left_context=75,
                 normalization_right_context=75, delta_window=2,
                 delay=0, add_pov_feature=True,
                 add_normalized_log_pitch=True, add_delta_pitch=True,
                 add_raw_log_pitch=False):
        self._options = pitch.ProcessPitchOptions()
        self.pitch_scale = pitch_scale
        self.pov_scale = pov_scale
        self.pov_offset = pov_offset
        self.delta_pitch_scale = delta_pitch_scale
        self.delta_pitch_noise_stddev = delta_pitch_noise_stddev
        self.normalization_left_context = normalization_left_context
        self.normalization_right_context = normalization_right_context
        self.delta_window = delta_window
        self.delay = delay
        self.add_pov_feature = add_pov_feature
        self.add_normalized_log_pitch = add_normalized_log_pitch
        self.add_delta_pitch = add_delta_pitch
        self.add_raw_log_pitch = add_raw_log_pitch

    @property
    def pitch_scale(self):
        """Scaling factor for the final normalized log-pitch value"""
        return self._options.pitch_scale

    @pitch_scale.setter
    def pitch_scale(self, value):
        self._options.pitch_scale = value

    @property
    def pov_scale(self):
        """Scaling factor for final probability of voicing feature"""
        return self._options.pov_scale

    @pov_scale.setter
    def pov_scale(self, value):
        self._options.pov_scale = value

    @property
    def pov_offset(self):
        """This can be used to add an offset to the POV feature

        Intended for use in Kaldi's online decoding as a substitute
        for CMV (cepstral mean normalization)

        """
        return self._options.pov_offset

    @pov_offset.setter
    def pov_offset(self, value):
        self._options.pov_offset = value

    @property
    def delta_pitch_scale(self):
        """Term to scale the final delta log-pitch feature"""
        return self._options.delta_pitch_scale

    @delta_pitch_scale.setter
    def delta_pitch_scale(self, value):
        self._options.delta_pitch_scale = value

    @property
    def delta_pitch_noise_stddev(self):
        """Standard deviation for noise we add to the delta log-pitch

        The stddev is added before scaling. Should be about the same
        as delta-pitch option to pitch creation. The purpose is to get
        rid of peaks in the delta-pitch caused by discretization of
        pitch values.

        """
        return self._options.delta_pitch_noise_stddev

    @delta_pitch_noise_stddev.setter
    def delta_pitch_noise_stddev(self, value):
        self._options.delta_pitch_noise_stddev = value

    @property
    def normalization_left_context(self):
        """Left-context (in frames) for moving window normalization"""
        return self._options.normalization_left_context

    @normalization_left_context.setter
    def normalization_left_context(self, value):
        self._options.normalization_left_context = value

    @property
    def normalization_right_context(self):
        """Right-context (in frames) for moving window normalization"""
        return self._options.normalization_right_context

    @normalization_right_context.setter
    def normalization_right_context(self, value):
        self._options.normalization_right_context = value

    @property
    def delta_window(self):
        """Number of frames on each side of central frame"""
        return self._options.delta_window

    @delta_window.setter
    def delta_window(self, value):
        self._options.delta_window = value

    @property
    def delay(self):
        """Number of frames by which the pitch information is delayed"""
        return self._options.delay

    @delay.setter
    def delay(self, value):
        self._options.delay = value

    @property
    def add_pov_feature(self):
        """If true, the warped NCCF is added to output features"""
        return self._options.add_pov_feature

    @add_pov_feature.setter
    def add_pov_feature(self, value):
        self._options.add_pov_feature = value

    @property
    def add_normalized_log_pitch(self):
        """If true, the normalized log-pitch is added to output features

         Normalization is done with POV-weighted mean subtraction over
         1.5 second window.

        """
        return self._options.add_normalized_log_pitch

    @add_normalized_log_pitch.setter
    def add_normalized_log_pitch(self, value):
        self._options.add_normalized_log_pitch = value

    @property
    def add_delta_pitch(self):
        """If true, time derivative of log-pitch is added to output features"""
        return self._options.add_delta_pitch

    @add_delta_pitch.setter
    def add_delta_pitch(self, value):
        self._options.add_delta_pitch = value

    @property
    def add_raw_log_pitch(self):
        """If true, time derivative of log-pitch is added to output features"""
        return self._options.add_raw_log_pitch

    @add_raw_log_pitch.setter
    def add_raw_log_pitch(self, value):
        self._options.add_raw_log_pitch = value

    def compute(self, raw_pitch):
        return SubMatrix(pitch.process_pitch(
            self._options, SubMatrix(raw_pitch))).numpy()
