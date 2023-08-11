import fitsio
from datetime import date
from numpy.lib.recfunctions import structured_to_unstructured
from . import __version__


def save_catalog(filename, arr, **kwargs):
    """
    Save a numpy.ndarray to a fits file.

    Parameters:
        arr (numpy.ndarray):
            Numpy array to save.
        filename (str):
            Path of the output fits file.
    """
    for key, value in kwargs.items():
        if not isinstance(value, str):
            raise ValueError(f"Value for key '{key}' is not a string!")
    # change to unstructured data to save disk space
    if arr.dtype.names is not None:
        arr = structured_to_unstructured(arr)
    today = date.today()
    kwargs["compress"] = ("rice",)
    kwargs["image compress"] = ("fpfs",)
    kwargs["version"] = (__version__,)
    kwargs["date"] = (today,)
    # rice compression is used by default
    fitsio.write(filename, arr, header=kwargs, compress="rice")
    return
