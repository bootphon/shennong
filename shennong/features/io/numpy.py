"""Saves and loads features to/from numpy files"""

import numpy as np

from shennong.features import Features, FeaturesCollection
from shennong.features.io.base import FeaturesHandler


# TODO test append
class NumpyHandler(FeaturesHandler):
    """Saves and loads features to/from the numpy 'npz' format"""
    def save_collection(self, features, compress=True):
        # represent the features as dictionaries
        data = {k: v._to_dict() for k, v in features.items()}

        # save (and optionally compress) the features
        save = np.savez_compressed if compress is True else np.savez
        save(self.filename, features=data)

    def load_collection(self):
        data = np.load(self.filename)['features'].tolist()
        features = FeaturesCollection()
        for k, v in data.items():
            features[k] = Features._from_dict(v)
        return features
