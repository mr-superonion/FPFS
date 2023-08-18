import fitsio
from datetime import date
from numpy.lib.recfunctions import structured_to_unstructured
from . import __version__


def save_catalog(filename, arr, **kwargs):
    """
    Save a numpy.ndarray to a fits file.

    Parameters:
        filename (str):
            Path of the output fits file.
        arr (numpy.ndarray):
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
    kwargs["compress"] = ("fitsio-gzip_2",)
    kwargs["image compress"] = ("fpfs",)
    kwargs["version"] = (__version__,)
    kwargs["date"] = (today,)
    if kwargs["dtype"] == "shape":
        # gzip compression is used for shape catalogs
        fitsio.write(filename, arr, header=kwargs, compress="GZIP_2", qlevel=None)
    elif kwargs["dtype"] == "position":
        # position catalogs. array of intergers, no compression
        fitsio.write(filename, arr, header=kwargs)
    else:
        raise ValueError(
            "dtype supports 'shape' or 'position'. %s is not supported."
            % kwargs["dtype"]
        )
    return


def save_image(filename, arr):
    """
    Save a numpy.ndarray to a fits file.

    Parameters:
        arr (numpy.ndarray):
            Numpy array to save.
        filename (str):
            Path of the output fits file.
    """
    # gzip compression is used by default
    fitsio.write(filename, arr, compress="GZIP_2", qlevel=None)
    return
