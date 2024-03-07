from datetime import date
import fitsio
from numpy.lib.recfunctions import structured_to_unstructured
from . import __version__


def save_catalog(filename: str, arr, **kwargs) -> None:
    """
    Save a numpy.ndarray to a fits file.

    Args:
    filename (str):
        Path of the output fits file.
    arr (ndarray):
        Numpy array to save.
    """
    for key, value in kwargs.items():
        if not isinstance(value, str):
            raise ValueError(f"Value for key '{key}' is not a string!")
    # change to unstructured data to save disk space
    if arr.dtype.names is not None:
        arr = structured_to_unstructured(arr)
    assert "dtype" in kwargs.keys(), "dtype (shape or coords?) is not specific"
    today = date.today()
    kwargs["image compress"] = ("fpfs",)
    kwargs["version"] = (__version__,)
    kwargs["date"] = (today,)
    if kwargs["dtype"] == "shape":
        # gzip compression is used for shape catalogs
        fitsio.write(filename, arr, header=kwargs)
    elif kwargs["dtype"] == "position":
        # position catalogs. array of intergers, no compression
        fitsio.write(filename, arr, header=kwargs)
    else:
        raise ValueError(
            "dtype supports 'shape' or 'position'. %s is not supported."
            % kwargs["dtype"]
        )
    return
