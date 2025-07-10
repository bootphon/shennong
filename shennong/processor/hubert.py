"""Extraction of HuBERT features from audio signals

    :class:`~shennong.audio.Audio` ---> HubertProcessor \
    ---> :class:`~shennong.features.Features`

Examples
--------

>>> from shennong.audio import Audio
>>> from shennong.processor.hubert import HubertProcessor
>>> audio = Audio.load('./test/data/test.wav')
>>> processor = HubertProcessor(model_path="facebook/hubert-large-ls960-ft", layer_info=("encoder", 1))

Compute the HuBERT features. the output is an
instance of :class:`~shennong.features.Features`:

>>> hubert = processor.process(audio)
>>> type(hubert)
<class 'shennong.features.Features'>

References
----------

.. [HuBERT] https://arxiv.org/abs/2106.07447

"""

import torch
import fairseq

import numpy as np

from ast import literal_eval
from shennong import Features
from shennong.processor.base import FeaturesProcessor
from transformers import HubertForCTC, AutoFeatureExtractor

class HubertProcessor(FeaturesProcessor):
    """HuBERT features from a pre-trained neural network

    Parameters
    ----------
    model_path : The path to the pre-trained HuBERT model

    layer_info : Tuple with the type of layer to extract features from (encoder or convolutional)
                 as the first element and the layer number as the second element

    Raises
    ------
    RuntimeError
        If the model path does not point to a HuBERT model 
        that can be loaded with either fairseq or huggingface

    ValueError
        If the selected layer does not exist in the given model
        or if the given layer type does not exist.
    """

    _SEED = 3939

    def __init__(self, model_path="", layer_info=("encoder", "1")):
        super().__init__()
        torch.manual_seed(self._SEED)
        np.random.seed(self._SEED)

        self.model_path = model_path
        try:
            self.model, self._cfg, self._task_cfg = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.model_path])
            self._conv_list = self._parse_conv_str(self._cfg['model']['conv_feature_layers'])
            self.model = self.model[0]
            self._model_type = 'fairseq'
        except:
            try:
                self.model = HubertForCTC.from_pretrained(self.model_path)
                self._model_type = 'huggingface'
            except:
                raise RuntimeError(f"The model at {self.model_path} cannot be loaded. Make sure that this is a fairseq model or huggingface model directory.")
        
        self.layer_info = layer_info
        layer_type = layer_info[0]
        layer = layer_info[1]

        self.layer_type = layer_type
        self._check_layer(int(layer))
        self.layer = layer

    @property
    def name(self):
        return 'hubert'

    @property
    def model_path(self):
        """The path to the pretrained HuBERT model"""
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        self._model_path = str(value)

    @property
    def layer(self):
        """The layer to extract features from"""
        return self._layer

    @layer.setter
    def layer(self, value):
        self._layer = int(value)
    
    @property
    def layer_type(self):
        """The type of layer that features are extracted from"""
        return self._layer_type

    @layer_type.setter
    def layer_type(self, value):
        self._layer_type = str(value)

    @property
    def ndims(self):
        """The dimension of extracted frames

        Cannot be tuned because the underlying neural networks are
        trained with this parameter.

        """
        if self.layer_type == 'encoder':
            if self._model_type == 'fairseq':
                return self._cfg['model']['encoder_embed_dim']
            elif self._model_type == 'huggingface':
                return self.model.config.hidden_size
        elif self.layer_type == 'convolutional':
            if self._model_type == 'fairseq':
                return self._conv_list[self.layer - 1][0]
            elif self._model_type == 'huggingface':
                return self.model.config.conv_dim[self.layer - 1]
    
    @property
    def sample_rate(self):
        """Processing sample frequency in Hertz

        Cannot be tuned because the underlying neural networks are
        trained with this parameter.

        """
        return 16000
    
    @property
    def frame_length(self):
        """The length of extracted frames (in seconds)

        Cannot be tuned because the underlying neural networks are
        trained with this parameter.

        """

        def get_receptive_field_length(kernels, strides):
            field_length = 1
            for i in range(len(kernels)):
                field_length += (kernels[i] - 1) * np.prod(strides[:i])
            return field_length

        if self.layer_type == 'encoder':
            # receptive field length of all convolutional layers
            if self._model_type == 'fairseq':
                _, kernels, strides = zip(*self._conv_list)
            elif self._model_type == 'huggingface':
                kernels = self.model.config.conv_kernel
                strides = self.model.config.conv_stride
        elif self.layer_type == 'convolutional':
            # receptive field length of convolution layers up to selected layer
            if self._model_type == 'fairseq':
                _, kernels, strides = zip(*self._conv_list[:self.layer])
            elif self._model_type == 'huggingface':
                kernels = self.model.config.conv_kernel[:self.layer]
                strides = self.model.config.conv_stride[:self.layer]
        
        frame_length = get_receptive_field_length(kernels, strides) / self.sample_rate
        return frame_length

    @property
    def frame_shift(self):
        """The time shift between two consecutive frames (in seconds)

        Cannot be tuned because the underlying neural networks are
        trained with this parameter.

        """
        if self.layer_type == 'encoder':
            # total stride length of all convolutional layers
            if self._model_type == 'fairseq':
                _, _, strides = zip(*self._conv_list)
            elif self._model_type == 'huggingface':
                strides = self.model.config.conv_stride
        elif self.layer_type == 'convolutional':
            # total stride length of convolution layers up to selected layer
            if self._model_type == 'fairseq':
                _, _, strides = zip(*self._conv_list[:self.layer])
            elif self._model_type == 'huggingface':
                strides = self.model.config.conv_stride[:self.layer]
        
        total_stride = np.prod(strides)
        frame_shift = total_stride / self.sample_rate
        return frame_shift
    
    def _check_layer(self, value):
        if self.layer_type == 'encoder':
            if self._model_type == 'fairseq':
                layer_num = self._cfg['model']['encoder_layers']
            elif self._model_type == 'huggingface':
                layer_num = self.model.config.num_hidden_layers
        elif self.layer_type == 'convolutional':
            if self._model_type == 'fairseq':
                layer_num = len(self._conv_list)
            elif self._model_type == 'huggingface':
                layer_num = len(self.model.config.conv_dim)
        else:
             raise ValueError("Invalid layer type")

        if value not in range(layer_num + 1):
            raise ValueError(f"There is no {self.layer_type} layer {value} in this model")
        elif not value:
            raise ValueError("No layers selected")

    def _parse_conv_str(self, conv_str):
        conv_list = []

        for item in conv_str.split("+"):
            item = item.strip()
            if "*" in item:
                feat, mult = item.split("*")
                conv_list.extend(([literal_eval(item)[0] for item in (feat * int(mult)).split()]))
            else:
                conv_list.append(literal_eval(item)[0])

        return conv_list

    def process(self, signal):
        """Computes HuBERT features with the specified options

        Use a pre-trained neural network to extract HuBERT
        features. Features have a frame shift of 20 ms and frame
        length of 20 ms.

        Parameters
        ----------
        signal : Audio, shape = [nsamples, 1]
            The input audio signal to compute the features on, must be
            mono. The signal is up/down-sampled to 16 kHz during
            processing.

        Returns
        -------
        features : Features, shape = [nframes, ndim]
            The computed HuBERT features will either:
            have as many rows as there are frames (depends on the `signal` duration, expect
            50 frames per second) for encoder layers,
            or have as many rows as there are samples divided by the product of the stride lengths 
            (depends on the `signal` duration and the stride lengths) for convolutional layers,
            each frame with the number of dimensions in the layer.

        Raises
        ------
        ValueError
            If the input `signal` has more than one channel (i.e. is
            not mono). If `sample_rate` != `signal.sample_rate`.
        """

        self.model.eval()

        # ensure the signal is correct
        if signal.nchannels != 1:
            raise ValueError(
                'signal must have one dimension, but it has {}'
                .format(signal.nchannels))

        # force resampling to 16 kHz and 32 bit floats
        need_resample = (
            signal.sample_rate != 16000 or
            signal.dtype is not np.dtype(np.float32))

        if need_resample:
            self.log.debug(
                'resampling audio from %dHz@%db to %dHz@%db',
                signal.sample_rate, signal.dtype.itemsize * 8, 16000, 32)
            signal = signal.resample(16000).astype(np.float32)

        if self._model_type == 'fairseq':
            signal = torch.unsqueeze(torch.from_numpy(signal.data), 0)
            self.model.feature_extractor = fairseq.models.hubert.HubertModel.build_model(self._cfg['model'], self._task_cfg).feature_extractor
            input_values = self.model.forward_features(signal).transpose(1, 2).squeeze(0).detach().numpy()
            out_dict = self.model(input_values, features_only=True, mask=False, output_layer=self.layer)
            if self.layer_type == 'encoder':
                data = out_dict["features"][0].squeeze(1).detach().numpy()
            elif self.layer_type == 'convolutional':
                self._cfg['model']['conv_feature_layers'] = str(self._conv_list[:self.layer])
                self.model.feature_extractor = fairseq.models.hubert.HubertModel.build_model(self._cfg['model'], self._task_cfg).feature_extractor
                data = self.model.forward_features(signal).transpose(1, 2).squeeze(0).detach().numpy()
        elif self._model_type == 'huggingface':
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
            input_values = feature_extractor(
                signal.data,
                return_tensors="pt",
                padding=True,
                sampling_rate=16000,
            ).input_values
            out_dict = self.model(input_values, output_hidden_states=True)
            if self.layer_type == 'encoder':
                data = out_dict["hidden_states"][self.layer][0].squeeze(1).detach().numpy()
            elif self.layer_type == 'convolutional':
                self.model.hubert.config.num_feat_extract_layers = self.layer
                self.model.hubert.feature_extractor = HubertForCTC(self.model.hubert.config).hubert.feature_extractor
                data = self.model.hubert.feature_extractor(signal).transpose(1, 2).squeeze(0).detach().numpy()
        del out_dict

        # compute the timestamps for each output frame
        times = np.vstack((
            np.arange(data.shape[0]) * self.frame_shift,
            np.arange(data.shape[0]) * self.frame_shift + self.frame_length)).T

        return Features(
            data, times, properties=self.get_properties())