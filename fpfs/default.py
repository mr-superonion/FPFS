# FPFS shear estimator
# Copyright 20220320 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# python lib
import os

sigM = 0.2
sigR = 0.2
sigP = 0.2
cutM = 25.0
cutR = 0.03
cutP = 0.005
cutRU = 2.0

__data_dir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
