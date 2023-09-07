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
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

colors = [
    "#000000",
    "#1976D2",
    "#E53935",
    "#43A047",
    "#673AB7",
    "#4DD0E1",
    "#E91E63",
    "#F2D026",
    "#333333",
    "#9E9E9E",
    "#FB8C00",
    "#FFB300",
    "#795548",
]

cblues = ["#004c6d", "#346888", "#5886a5", "#7aa6c2", "#9dc6e0", "#c1e7ff"]
creds = ["#DC1C13", "#EA4C46", "#F07470", "#F1959B", "#F6BDC0", "#F8D8E3"]


def make_figure_axes(ny=1, nx=1, square=True):
    """Makes figure and axes

    Args:
        ny (int):       number of subplots in y direction
        nx (int):       number of subplots in y direction
        square (bool):  whether using square plot
    """
    if not isinstance(ny, int):
        raise TypeError("ny should be integer")
    if not isinstance(nx, int):
        raise TypeError("nx should be integer")
    axes = []
    if ny == 1 and nx == 1:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(ny, nx, 1)
        axes.append(ax)
    elif ny == 2 and nx == 1:
        if square:
            fig = plt.figure(figsize=(6, 11))
        else:
            fig = plt.figure(figsize=(6, 7))
        ax = fig.add_subplot(ny, nx, 1)
        axes.append(ax)
        ax = fig.add_subplot(ny, nx, 2)
        axes.append(ax)
    elif ny == 1 and nx == 2:
        fig = plt.figure(figsize=(11, 6))
        for i in range(1, 3):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    elif ny == 1 and nx == 3:
        fig = plt.figure(figsize=(18, 6))
        for i in range(1, 4):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    elif ny == 1 and nx == 4:
        fig = plt.figure(figsize=(20, 5))
        for i in range(1, 5):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    elif ny == 2 and nx == 3:
        fig = plt.figure(figsize=(15, 8))
        for i in range(1, 7):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    elif ny == 2 and nx == 4:
        fig = plt.figure(figsize=(20, 8))
        for i in range(1, 9):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    else:
        raise ValueError("Do not have option: ny=%s, nx=%s" % (ny, nx))
    return fig, axes


def determine_cuts(data):
    """
    Determine min_cut and max_cut for the data using median and standard deviation.

    Parameters:
        data (ndarray): 2D numpy array containing the image data
        sigma (int): Number of standard deviations to use for max_cut

    Returns:
        min_cut, max_cut: Calculated cuts
    """
    min_cut = np.percentile(np.ravel(data), 5)
    max_cut = np.percentile(np.ravel(data), 98)
    return min_cut, max_cut


def make_plot_image(data):
    min_cut, max_cut = determine_cuts(data)
    sn = simple_norm(data, "asinh", asinh_a=0.1, min_cut=min_cut, max_cut=max_cut)
    fig = plt.imshow(data, aspect="equal", cmap="RdYlBu_r", origin="lower", norm=sn)
    return fig
