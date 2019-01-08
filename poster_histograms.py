#!/usr/bin/env python
"""Generate the histograms for my AAS 2019 jointcal poster.

Run `setup lsst_distrib` before this, and have some data processed with validate_drp available.

John K. Parejko
"""

import os.path
import glob

import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('white')  # noqa: E402
seaborn.set_context("poster")  # noqa: E402

from astropy.visualization import quantity_support
quantity_support()  # noqa: E402

from lsst.verify import Name
from lsst.validate.drp import report_performance


name = Name('validate_drp', 'AM1')
# path = "/Users/parejkoj/lsst/temp/AAS2019"
path = "/project/parejkoj/DM-11783/DM-15713/validate-%s/8524"


def load_data(kind):
    """Load data from validate_drp output json files."""
    files = glob.glob(os.path.join(path%kind, '*.json'))
    print("Reading %s files."%len(files))
    metrics = report_performance.ingest_data(files, 'verify_metrics')
    metric = metrics['HSC-I']
    mm = metric.measurements[name]
    data = mm.blobs['MatchedMultiVisitDataset']
    amodel = mm.blobs['AnalyticAstrometryModel']
    pmodel = mm.blobs['AnalyticAstrometryModel']
    return data, amodel, pmodel


# load the data and extract the parts we want
jointcal, amodel, pmodel = load_data('jointcal')
single, amodel, pmodel = load_data('single')
bright_jointcal = jointcal['snr'].quantity.value > amodel['brightSnr'].quantity
bright_single = single['snr'].quantity.value > amodel['brightSnr'].quantity
d1 = jointcal['dist'].quantity[bright_jointcal]
d2 = single['dist'].quantity[bright_single]
m1 = jointcal['magrms'].quantity[bright_jointcal]
m2 = single['magrms'].quantity[bright_single]


# Playing with what's in the validate_drp output
# files = glob.glob(os.path.join(path, 'jointcal', '*.json'))
# metrics = report_performance.ingest_data(files, 'verify_metrics')
# metric = metrics['HSC-R']
# mm=metric.measurements[name]


# jointcal.keys()


# generic figure stuff
bins = 50
labels = ['jointcal', 'processCcd']


# color cycling
i = 0
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


# astrometry
xlim = (0, 50)
plt.figure(figsize=(12, 8))
plt.hist(d1, bins=bins, range=xlim, histtype='step', label=labels[0], lw=2, color=cycle[i])
plt.hist(d2, bins=bins, range=xlim, histtype='step', label=labels[1], lw=2, color=cycle[i+1])
plt.axvline(np.median(d1), lw=1.5, ls='--', color=cycle[i])
plt.text(11, 3000, "%5.2f mas"%np.median(d1.value))
plt.axvline(np.median(d2), lw=1.5, ls='--', color=cycle[i+1])
plt.text(20.5, 3000, "%5.2f mas"%np.median(d2.value))
plt.xlim(xlim)
plt.xlabel('RMS repeat visit source separation (milliarcsec)')
plt.ylabel('N')
plt.legend()
plt.savefig('astrometry-hist.svg', bbox_inches='tight')
plt.savefig('astrometry-hist.pdf', bbox_inches='tight')


xlim = (0, 400)
plt.figure(figsize=(12, 8))
plt.hist(m1.to(u.mmag), bins=bins, range=xlim, histtype='step', label=labels[0], lw=2, color=cycle[i])
plt.hist(m2.to(u.mmag), bins=bins, range=xlim, histtype='step', label=labels[1], lw=2, color=cycle[i+1])
plt.axvline(np.median(m1), lw=1.5, ls='--', color=cycle[i])
plt.text(35, 8000, "%5.2f mmag"%np.median(m1.to_value(u.mmag)))
plt.axvline(np.median(m2), lw=1.5, ls='--', color=cycle[i+1])
plt.text(170, 8000, "%5.2f mmag"%np.median(m2.to_value(u.mmag)))
plt.xlim(xlim)
plt.xlabel('RMS repeat visit source brightness (millimag)')
plt.ylabel('N')
plt.legend()
plt.savefig('photometry-hist.svg', bbox_inches='tight')
plt.savefig('photometry-hist.pdf', bbox_inches='tight')
