# Change the Dir variable to the current directory

Dir="$homeSys/code/FPFS2_Private/"
DirF="$homeSys/code/FPFS_Private/FPFSBASE/"
DirM="$homeSys/code/massMap_Private/"

export PYTHONPATH="$Dir/python/":$PYTHONPATH
export PATH="$Dir/bin/":$PATH
export PYTHONPATH="$DirF/python/":$PYTHONPATH
export PATH="$DirF/bin/":$PATH
export PYTHONPATH="$DirM/python/":$PYTHONPATH
export PATH="$DirM/bin/":$PATH
