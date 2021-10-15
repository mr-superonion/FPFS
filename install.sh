# Change the Dir variable to the current directory
# The scipt for installation by developer

Dir="$homeSys/code/FPFS2_Private/"
DirF="$homeSys/code/FPFS_Private/FPFSBASE/"
DirM="$homeSys/code/massMap_Private/"

export PATH="$Dir/bin/":$PATH
export PYTHONPATH="$Dir/bin/":$PYTHONPATH
export PYTHONPATH="$DirF/python/":$PYTHONPATH
export PYTHONPATH="$DirM/python/":$PYTHONPATH
