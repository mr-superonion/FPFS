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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors0=[
    "black",
    "#1A85FF",
    "#D41159",
    "#DE8817",
    "#A3D68A",
    "#35C3D7",
    "#8B0F8C",
    ]
colors=[]
for _ic,_cc in enumerate(colors0):
    cc2= mcolors.ColorConverter().to_rgb(_cc)
    colors.append((cc2[0], cc2[1], cc2[2], 1-0.1*_ic))
    del cc2

cblue=[
    "#004c6d",
    "#346888",
    "#5886a5",
    "#7aa6c2",
    "#9dc6e0",
    "#c1e7ff"
    ]

cred=[
    "#DC1C13",
    "#EA4C46",
    "#F07470",
    "#F1959B",
    "#F6BDC0",
    "#F8D8E3"
    ]

def make_figure_axes(ny=1,nx=1,square=True):
    """
    Args:
        ny (int):       number of subplots in y direction
        nx (int):       number of subplots in y direction
        square (bool):  whether using square plot
    """
    if not isinstance(ny, int):
        raise TypeError("ny should be integer")
    if not isinstance(nx, int):
        raise TypeError("nx should be integer")
    axes=[]
    if ny ==1 and nx==1:
        fig=plt.figure(figsize=(6,5))
        ax=fig.add_subplot(ny,nx,1)
        axes.append(ax)
    elif ny==2 and nx==1:
        if square:
            fig=plt.figure(figsize=(6,11))
        else:
            fig=plt.figure(figsize=(6,7))
        ax=fig.add_subplot(ny,nx,1)
        axes.append(ax)
        ax=fig.add_subplot(ny,nx,2)
        axes.append(ax)
    elif ny==1 and nx==2:
        fig=plt.figure(figsize=(11,6))
        for i in range(1,3):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    elif ny==1 and nx==3:
        fig=plt.figure(figsize=(18,6))
        for i in range(1,4):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    elif ny==1 and nx==4:
        fig=plt.figure(figsize=(20,5))
        for i in range(1,5):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    elif ny==2 and nx==3:
        fig=plt.figure(figsize=(15,8))
        for i in range(1,7):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    elif ny==2 and nx==4:
        fig=plt.figure(figsize=(20,8))
        for i in range(1,9):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    else:
        raise ValueError('Do not have option: ny=%s, nx=%s' %(ny,nx))
    return fig,axes
