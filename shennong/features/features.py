"""Builds, saves, loads and manipulate features data"""

import h5features as h5f
import numpy as np


class FeaturesCollection(dict):
    @staticmethod
    def load(filename, format='h5features'):
        pass

    def save(self, filename, format='h5features'):
        pass

    def by_speaker(self, spk2utt):
        """Returns a dict of :class:`FeaturesCollection` indexed by speakers

        Parameters
        ----------
        spk2utt : dict
            A mapping of speakers to their associated utterances
            (items in the FeaturesCollection). We must have
            ``spk2utt.values() == self.keys()``.

        Returns
        -------
        features : dict of FeaturesCollection
            A list of FeaturesCollection instances, one per speaker
            defined in `utt2spk`

        Raises
        ------
        ValueError
            If one utterance in the collection is not mapped in
            `spk2utt`.

        """
        undefined_utts = set(self.keys()).difference(set(spk2utt.values()))
        if len(undefined_utts) != 0:
            raise ValueError(
                'following utterances are not defined in spk2utt: {}'
                .format(sorted(undefined_utts)))

        return {spk: FeaturesCollection({utt: self[utt] for utt in utts})
                for spk, utts in spk2utt}


class Features:
    def __init__(self, data, times, properties=None, validate=True):
        self._data = data
        self._times = times
        self._properties = properties

        # make sure the features are in a valid state
        if validate is True:
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
    def shape(self):
        """The shape of the features data, as (nframes, nlabels)"""
        return self.data.shape

    @property
    def dtype(self):
        """The type of the features data"""
        return self.data.dtype

    @property
    def properties(self):
        """A dictionnary of properties used to build the features

        Properties are references to the features extraction pipeline,
        parameters and source audio file used to generate the
        features.

        """
        return self._properties

    def __eq__(self, other):
        if self.shape != other.shape:
            return False
        if not np.array_equal(self.times, other.times):
            return False
        if not self.properties == other.properties:
            return False
        if not np.array_equal(self.data, other.data):
            return False
        return True

    def is_valid(self):
        """Returns True if the features are in a valid state

        Returns False otherwise. Consistency is checked for features's
        data, times and labels.

        See Also
        --------
        Features.validate

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

        # TODO check properties

        if len(errors):
            raise ValueError('invalid features: {}'.format(', '.join(errors)))

    def concatenate(self, other):
        """Returns the concatenation of this features with `other`

        Build a new Features instance made of the concatenation of
        this instance with the other instance. Their `times` must be
        the equal.

        Parameters
        ----------
        other : Features, shape = [nframes, nlabels2]
            The other features to concatenate at the end of this one

        Returns
        -------
        features : Features, shape = [nframes, nlabels1 + nlabels2]

        Raises
        ------
        ValueError
            If `other` cannot be concatenated because of inconsistencies

        """
        # ensures time axis is shared accross the two features
        if not np.array_equal(self.times, other.times):
            raise ValueError('times are not equal')

        return Features(
            np.hstack((self.data, other.data)),
            np.hstack((self.labels, other.labels)),
            self.times,
            # TODO need a FeaturesParameter class: assign properties
            # per column
            self.properties.update(other.properties))

    def save(self, filename, groupname=None, append=False):
        h5data = h5f.Data()
        with h5f.Writer(filename) as writer:
            writer.write(h5data, groupname, append=append)
