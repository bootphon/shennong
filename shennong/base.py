"""Base classes for all shennong components"""

import collections
import inspect
import logging


class BaseProcessor:
    """Base class for all processors in shennong

    Notes
    -----
    All processors should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    The methods :meth:`get_params` and :meth:`set_params` are adapted
    from :class:`sklearn.base.BaseEstimator`

    """
    _log = logging.getLogger()

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the processor"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for param in parameters:
            if param.kind == param.VAR_POSITIONAL:
                raise RuntimeError(
                    'shennong processors should always '
                    'specify their parameters in the signature '
                    'of their __init__ (no varargs). '
                    '%s with constructor %s does not '
                    'follow this convention.'
                    % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this processor.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this processor and
            contained subobjects that are processors. Default to True.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this processor.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If any given parameter in ``params`` is invalid for the
            processor.

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = collections.defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError(
                    'invalid parameter %s for processor %s, '
                    'check the list of available parameters '
                    'with `processor.get_params().keys()`.' %
                    (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                try:
                    setattr(self, key, value)
                except AttributeError:
                    raise ValueError(
                        'cannot set attribute %s for %s'
                        % (key, self))
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
