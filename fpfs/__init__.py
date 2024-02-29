# flake8: noqa
import os
from .__version__ import __version__
from .__data_dir__ import __data_dir__
from . import io
from . import image
from . import catalog
from . import simulation
from . import plot

# We need accuracy is below 1e-6
from jax import config

config.update("jax_enable_x64", True)

__all__ = [
    "image",
    "catalog",
    "simulation",
    "plot",
    "io",
]
