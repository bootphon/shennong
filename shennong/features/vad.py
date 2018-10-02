###############################################################################
#                                                                             #
#  copyright (C) 2017 by Anna Silnova, Pavel Matejka, Oldrich Plchot,         #
#                        Frantisek Grezl                                      #
#                                                                             #
#                        Brno Universioty of Technology                       #
#                        Faculty of information technology                    #
#                        Department of Computer Graphics and Multimedia       #
#  email: {isilnova,matejkap,iplchot,grezl}@vut.cz                            #
#                                                                             #
###############################################################################
#                                                                             #
#  This software and provided models can be used freely for research          #
#  and educational purposes. For any other use, please contact BUT            #
#  and / or LDC representatives.                                              #
#                                                                             #
###############################################################################
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at                                     #
#                                                                             #
#     http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
###############################################################################
"""Voice Activity Detection"""

import numpy as np

from shennong.core.processor import FeaturesProcessor
from shennong.features.features import Features
from shennong.core import gmm, frames


def framing(a, window, shift=1):
    shape = (int((a.shape[0] - window) / shift + 1), window) + a.shape[1:]
    strides = (a.strides[0] * shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def compute_vad(signal, win_length=200, win_overlap=120,
                n_realignment=5, threshold=0.3, compression=None):
    # power signal for energy computation
    # frame signal with overlap
    # sum frames to get energy
    E = framing(signal ** 2, win_length, shift=win_length-win_overlap).sum(
        axis=1).astype(np.float64)
    if compression == 'sqrt':
        E = np.sqrt(E)
    elif compression == 'log':
        E = np.log(E)

    # normalize the energy
    E -= E.mean()
    try:
        E /= E.std()
        print(E)

        # initialization
        mm = np.array((-1.00, 0.00, 1.00))[:, np.newaxis]
        ee = np.array((1.00, 1.00, 1.00))[:, np.newaxis]
        ww = np.array((0.33, 0.33, 0.33))

        GMM = gmm.gmm_eval_prep(ww, mm, ee)

        E = E[:, np.newaxis]

        for i in range(n_realignment):
            # collect GMM statistics
            llh, N, F, S = gmm.gmm_eval(E, GMM, return_accums=2)

            # update model
            ww, mm, ee = gmm.gmm_update(N, F, S)

            # wrap model
            GMM = gmm.gmm_eval_prep(ww, mm, ee)

        # evaluate the gmm llhs
        llhs = gmm.gmm_llhs(E, GMM)

        llh = gmm.logsumexp(llhs, axis=1)[:, np.newaxis]

        llhs = np.exp(llhs - llh)

        out = np.zeros(llhs.shape[0], dtype=np.bool)
        out[llhs[:, 0] < threshold] = True
    except RuntimeWarning:
        raise ValueError("signal contains only silence")

    return out


class VadProcessor(FeaturesProcessor):
    def __init__(self, sample_rate=16000, frame_shift=0.01,
                 frame_length=0.025, nrealignments=5, threshold=0.3,
                 compression=None):
        self.frame = frames.Frames(
            sample_rate=sample_rate,
            frame_shift=frame_shift,
            frame_length=frame_length)

        self.nrealignments = nrealignments
        self.threshold = threshold

        if compression is not None and compression not in ('log', 'sqtr'):
            raise ValueError(
                'compression must be None, "log" or "sqrt" but is {}'
                .format(compression))
        self.compression = compression

    def parameters(self):
        return {
            'sample_rate': self.frame.sample_rate,
            'frame_shift': self.frame.frame_shift,
            'frame_length': self.frame.frame_length,
            'nrealignments': self.nrealignments,
            'threshold': self.threshold,
            'compression': self.compression}

    def process(self, signal):
        if signal.nchannels() != 1:
            raise ValueError(
                'audio signal must have one channel, but it has {}'
                .format(signal.nchannels()))

        if self.frame.sample_rate != signal.sample_rate:
            raise ValueError(
                'processor and signal mismatche in sample rates: '
                '{} != {}'.format(self.frame.sample_rate, signal.sample_rate))

        # power signal for energy computation
        data = signal.data ** 2

        # compute energy on framed powered  data
        energy = self.frame.framed_array(data).sum(axis=1).asdtype(np.float64)

        # compress the energy if required
        if self.compression == 'sqrt':
            energy = np.sqrt(energy)
        elif self.compression == 'log':
            energy = np.log(energy)

        # normalize the energy and transpose the vector
        energy -= energy.mean()
        energy /= energy.std()
        energy = energy[:, np.newaxis]

        # GMM initialization
        mm = np.array((-1.00, 0.00, 1.00))[:, np.newaxis]
        ee = np.array((1.00, 1.00, 1.00))[:, np.newaxis]
        ww = np.array((0.33, 0.33, 0.33))
        GMM = gmm.gmm_eval_prep(ww, mm, ee)

        for _ in range(self.nrealignments):
            # collect GMM statistics
            llh, N, F, S = gmm.gmm_eval(energy, GMM, return_accums=2)

            # update model
            ww, mm, ee = gmm.gmm_update(N, F, S)

            # wrap model
            GMM = gmm.gmm_eval_prep(ww, mm, ee)

        # evaluate the gmm llhs
        llhs = gmm.gmm_llhs(energy, GMM)

        llh = gmm.logsumexp(llhs, axis=1)[:, np.newaxis]

        llhs = np.exp(llhs - llh)

        out = np.zeros(llhs.shape[0], dtype=np.bool)
        out[llhs[:, 0] < self.threshold] = True

        return Features(
            out[:, np.newaxis],
            frames.mean(axis=1) / self.sample_rate,
            self.parameters())
