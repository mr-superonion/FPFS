import os
from setuptools import setup, find_packages


this_dir = os.path.dirname(os.path.realpath(__file__))
__version__ = ""
fname = os.path.join(this_dir, "fpfs", "__version__.py")
with open(fname, "r") as ff:
    exec(ff.read())
long_description = open(os.path.join(this_dir, "README.md")).read()


scripts = [
    "bin/fpfs_config",
    "bin/fpfs_sim.py",
    "bin/fpfs_process_sim.py",
    "bin/fpfs_summary_sim.py",
    "bin/fpfs_process_descsim.py",
    "bin/fpfs_summary_descsim.py",
]


setup(
    name="fpfs",
    version=__version__,
    description="FPFS shear estimator",
    author="Xiangchong Li",
    author_email="mr.superonion@hotmail.com",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "schwimmbad",
        "jax>=0.4.9",
        "jaxlib>=0.4.9",
        "galsim",
        "astropy",
        "matplotlib",
    ],
    packages=find_packages(),
    scripts=scripts,
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/mr-superonion/FPFS/",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
