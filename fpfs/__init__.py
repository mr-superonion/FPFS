# flake8: noqa
from .__version__ import __version__
from . import io
from . import image
from . import catalog
from . import simulation
from . import pltutil
from . import default
from . import pltutil
from .default import __data_dir__

# We need accuracy is below 1e-6
from jax import config

config.update("jax_enable_x64", True)

__all__ = [
    "image",
    "catalog",
    "simulation",
    "pltutil",
    "default",
    "io",
]
