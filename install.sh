#!/usr/bin/env bash

# put ./bin into PATH
Dir=$(realpath ./bin/) #absolute_path_of_this_dictory
export PATH="$Dir/bin/":$PATH
export PYTHONPATH="$Dir/bin/":$PYTHONPATH
