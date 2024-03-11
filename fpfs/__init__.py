# flake8: noqa
import os

# We need accuracy is below 1e-6
from jax import config

from . import catalog, image, simulation
from .__data_dir__ import __data_dir__
from .__version__ import __version__

config.update("jax_enable_x64", True)

__all__ = [
    "image",
    "catalog",
    "simulation",
]
