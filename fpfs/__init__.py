# flake8: noqa
from .__version__ import __version__
from . import io
from . import image
from . import imgutil
from . import catalog
from . import simutil
from . import pltutil
from . import default
from . import pltutil
from . import tasks
from .default import __data_dir__

# We need accuracy is below 1e-6
from jax import config

config.update("jax_enable_x64", True)

__all__ = [
    "image",
    "imgutil",
    "catalog",
    "simutil",
    "pltutil",
    "default",
    "io",
    "tasks",
]
