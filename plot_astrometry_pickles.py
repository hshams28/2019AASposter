#!/usr/bin/env python
"""Plot the pickles produced by poster_astrometry.py
"""
import os
import pickle

import numpy as np

import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('white')  # noqa: E402
seaborn.set_context("poster")  # noqa: E402


def uv_mean(bbox, xx, yy, uu, vv):
    """Compute the mean of uu and vv on a grid within bbox."""
    nx = 30
    ny = 60
    uMean = np.zeros((nx-1, ny-1))
    vMean = np.zeros((nx-1, ny-1))
    xMean = np.zeros((nx-1, ny-1))
    yMean = np.zeros((nx-1, ny-1))
    ww = np.linspace(0, bbox.getWidth(), nx)
    hh = np.linspace(0, bbox.getHeight(), ny)
    for i, (w0, w1) in enumerate(zip(ww[:-1], ww[1:])):
        inx = (xx >= w0) & (xx <= w1)
        for j, (h0, h1) in enumerate(zip(hh[:-1], hh[1:])):
            iny = (yy >= h0) & (yy <= h1)
            inside = inx & iny
            xMean[i, j] = (w0 + w1)/2
            yMean[i, j] = (h0 + h1)/2
            uMean[i, j] = np.mean(uu[inside])
            vMean[i, j] = np.mean(vv[inside])

    return xMean, yMean, uMean, vMean


def plot_quiver(xx, yy, uu, vv, ccd, label):
    """Make a quiver plot of the astrometry error vectors."""
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    color = cycle[0] if 'jointcal' in label else cycle[1]
    scale = 0.25 if 'mean' in label else 1

    fig = plt.figure(figsize=(6, 10))
    ax = fig.add_subplot(111)

    Q = ax.quiver(xx, yy, uu, vv, units='x', pivot='tail', scale=scale, width=7,
                  headwidth=4, clip_on=False, color=color)
    length = 5/scale if 'mean' in label else 100
    key = ax.quiverkey(Q, 0.85, 0.90, length, '%s mas'%(length), angle=45,
                       coordinates='figure', labelpos='W', fontproperties={'size': 24})

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # fig.subplots_adjust(t)

    filename = "plots/quiver-%s-%s.png"%(ccd, label)
    plt.savefig(filename, bbox_extra_artists=(key,))# bbox_inches="tight")
    plt.close(fig)


name = 'single'
# name = 'jointcal'
filename = '~/lsst/temp/AAS2019/quiver/pickle/quiverData-%s-6.pickle'%name
with open(os.path.expanduser(filename), 'rb') as infile:
    xx, yy, uu, vv, bbox, ccd = pickle.load(infile)
    xMean, yMean, uMean, vMean = uv_mean(bbox, xx, yy, uu, vv)
    plot_quiver(xMean, yMean, uMean, vMean, ccd, 'mean-'+name)
