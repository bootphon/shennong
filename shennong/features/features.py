"""Builds, saves, loads and manipulate features data"""

import h5features as h5f
import numpy as np


class Features:
    def __init__(self, data, labels, times, parameters):
        self._data = data
        self._labels = labels
        self._times = times
        self._parameters = parameters

        # make sure the features are in a valid state
        self.validate()

    @property
    def data(self):
        """The underlying features data as a numpy array"""
        return self._data

    @property
    def times(self):
        """The frames timestamps on the vertical axis"""
        return self._times

    @property
    def labels(self):
        """The name of the features columns on the horieontal axis"""
        return self._labels

    @property
    def shape(self):
        """The shape of the features data, as (nframes, nlabels)"""
        return self.data.shape

    @property
    def parameters(self):
        """A dictionnary of parameters used to buidld the features"""
        return self._parameters

    def __eq__(self, other):
        if self.shape != other.shape:
            return False
        if not np.array_equal(self.labels, other.labels):
            return False
        if not np.array_equal(self.times, other.times):
            return False
        if not self.parameters == other.parameters:
            return False
        if not np.array_equal(self.data, other.data):
            return False
        return True

    def is_valid(self):
        """Returns True if the features are consitent, False otherwise

        Consistency is checked for features's data, times and labels.

        """
        try:
            self.validate()
        except ValueError:
            return False
        return True

    def validate(self):
        """Raises a ValueError if the features are not in a valid state"""
        errors = []

        # check data dimensions
        ndim = self.data.ndim
        if not ndim == 2:
            errors.append('data dimension must be 2 but is {}'.format(ndim))

        # check times dimensions
        ndim = self.times.ndim
        if not ndim == 1:
            errors.append('times dimension must be 1 but is {}'.format(ndim))

        nframes1 = self.data.shape[0]
        nframes2 = self.times.shape[0]
        if not nframes1 == nframes2:
            errors.append('mismath in number of frames: {} for data but {} '
                          'for times'.format(nframes1, nframes2))

        # check labels dimensions
        ndim = self.labels.ndim
        if not ndim == 1:
            errors.append('labels dimension must be 1 but is {}'.format(ndim))

        nlabels1 = self.data.shape[1]
        nlabels2 = self.labels.shape[1]
        if not nlabels1 == nlabels2:
            errors.append('mismath in number of labels: {} for data but {} '
                          'for labels'.format(nlabels1, nlabels2))

        if len(errors):
            raise ValueError('invalid features: {}'.format(', '.join(errors)))

    def concatenate(self, other):
        """Returns the concatenation of this features with `other`

        Build a new Features instance made of the concatenation of
        this instance with the other instance. The `times` must be the
        same for the two features. The `labels` must be disjoint.

        Parameters
        ----------
        other : Features, shape = [nframes, nlabels2]
            The other features to concatenate at the end of this one

        Returns
        -------
        features : Features, shape = [nframes, nlabels1 + nlabels2]

        """
        pass

    def save(self, filename, groupname=None, append=False):
        h5data = h5f.Data()
        with h5f.Writer(filename) as writer:
            writer.write(h5data, groupname, append=append)
