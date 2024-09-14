import inspect
from typing import Any

import numpy as np

numbers = int | float | np.integer | np.floating


def gather_vars_to_kwargs() -> dict[str, Any]:
    """Gather all local variables in the calling function and return them as a dictionary. If a variable is named `kwargs`, it will be updated with the rest of the variables. This function is useful for initializing classes with a large number of arguments.

    Returns
    -------
    dict[str, Any]
        A dictionary containing all local variables in the calling function, except for those that start and end with double underscores.
    """
    vars = inspect.getouterframes(inspect.currentframe(), 2)[1][0].f_locals
    kwarg = vars.pop('kwargs', {})
    kwarg.update(vars)
    for k in list(kwarg.keys()):
        if k.startswith('__') and k.endswith('__'):
            del kwarg[k]
    return kwarg
