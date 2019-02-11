"""Base classes for features loaders and savers"""

import abc


class FeaturesHandler(metaclass=abc.ABCMeta):
    """Base class of a features file handler

    This class must be specialized to handle a given file type.

    """
    def __init__(self, filename, mode=None):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    # @abc.abstractmethod
    # def load(self):
    #     """Returns a single features matrix from the `filename`

    #     Parameters
    #     ----------
    #     filename : str
    #         The file to read

    #     Returns
    #     -------
    #     features : :class:`~shennong.features.Features`
    #         The features stored in the file.

    #     Raises
    #     ------
    #     IOError
    #         If the input file does not exist or cannot be read.

    #     ValueError
    #         If the features cannot be read/loaded from the
    #         ``filename``.

    #     """
    #     pass

    @abc.abstractmethod
    def load_collection(self):
        """Returns a collection of features matrices from the `filename`"""
        pass

    # @abc.abstractmethod
    # def save(self, features):
    #     """Saves a single features matrix to `filename`"""
    #     pass

    @abc.abstractmethod
    def save_collection(self, features_collection):
        """Saves a collection of features to the `filename`"""
        pass
