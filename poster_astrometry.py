#!/usr/bin/env python
"""Plot astrometric residuals for my 2019 AAS poster."""

import itertools
import collections
import pickle

import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('white')  # noqa: E402
seaborn.set_context("poster")  # noqa: E402

import lsst.daf.persistence
import lsst.meas.astrom
import lsst.afw.table
from lsst.validate.drp.util import positionRmsFromCat, averageRaDecFromCat


# switch this for use on lsst-dev
# datadir = '/Users/parejkoj/lsst/jointcal/jointcal/tests/.test/JointcalTestCFHT/test_jointcalTask_2_visits_constrainedAstrometry_no_photometry'  # noqa
# tract = 0
# filt = 'r'
# visits = [849375, 850587]
# ccds = [12, 13, 14, 21, 22, 23]

# Using: WIDE_8524_HSC-I
datadir = '/project/parejkoj/hscRerun/DM-15713/WIDE'
tract = 8524
visits = [7286, 7288, 7298, 7300, 7302, 7304, 7310, 7338, 7340, 7350, 7352, 7356, 7358, 7364, 7366, 7370, 7372, 7378, 7384, 7386, 7390, 7392, 7394, 7396, 7400, 7402, 7416, 14124, 14126, 14128, 14130, 14142, 14144, 14146, 14164, 14166, 14176, 14178, 14196, 14198, 14206, 14208, 14210, 1623]  # noqa
# visits = [7286, 7350]
ccds = list(range(1, 103))
filt = 'HSC-I'
ccds.remove(9)  # ccd 9 is bad
ccds = np.array(ccds)
# ccds = np.arange(40, 60)  # subset of ccds to check things quickly


def filter_matches(allMatches, fluxField):
    """Filter down to matches with at least 2 sources and good flags."""
    flagKeys = [allMatches.schema.find("base_PixelFlags_flag_%s" % flag).key
                for flag in ("saturated", "cr", "bad", "edge")]
    nMatchesRequired = 2

    fluxSnrKey = allMatches.schema.find(fluxField + "_snr").key
    # NOTE: alias oddities mean I have to change "_flux" <-> "_instFlux" depending on catalog version.
    fluxKey = allMatches.schema.find(fluxField + "_flux").key

    def goodFilter(cat, goodSnr=10):
        if len(cat) < nMatchesRequired:
            return False
        for flagKey in flagKeys:
            if cat.get(flagKey).any():
                return False
        if not (cat.get(fluxKey) > 0).all():
            return False
        snr = np.median(cat.get(fluxSnrKey))
        # Note that this also implicitly checks for snr being non-nan.
        return snr >= goodSnr

    return allMatches.where(goodFilter)


def prep_matching(butler, visits, ccds):
    dataId = dict(visit=int(visits[0]), ccd=int(ccds[0]), tract=tract, filter=filt)
    catalog = butler.get('src', dataId=dataId)
    bbox = butler.get('calexp_bbox', dataId=dataId)
    oldSchema = catalog.schema
    fluxField = oldSchema.getAliasMap().get("slot_CalibFlux")

    # make the new schema, with a field for S/N
    mapper = lsst.afw.table.SchemaMapper(oldSchema)
    mapper.addMinimalSchema(oldSchema)
    mapper.addOutputField(lsst.afw.table.Field[float](fluxField + '_snr', 'flux SNR'))
    newSchema = mapper.getOutputSchema()
    newSchema.setAliasMap(oldSchema.getAliasMap())

    multiMatch = lsst.afw.table.MultiMatch(newSchema, {"visit": np.int32, "ccd": np.int32, 'tract': np.int32})
    return bbox, fluxField, newSchema, mapper, multiMatch


def do_match(multiMatch, butler, visits, ccds, fluxField, newSchema, mapper, useJointcal=False):
    """Make the multiMatch, identify good matches, and compute aggregate statistics."""
    for visit, ccd in itertools.product(visits, ccds):
        dataId = dict(visit=int(visit), ccd=int(ccd), tract=tract, filter=filt)
        try:
            oldCat = butler.get('src', dataId=dataId)
        except lsst.daf.persistence.butlerExceptions.NoResults:
            # ignore missing data
            print("No data:", visit, ccd)
            continue
        if useJointcal:
            wcs = butler.get('jointcal_wcs', dataId=dataId)
            lsst.afw.table.updateSourceCoords(wcs, oldCat)

        catalog = lsst.afw.table.SourceCatalog(newSchema)
        tmpCat = lsst.afw.table.SourceCatalog(lsst.afw.table.SourceCatalog(newSchema).table)
        tmpCat.extend(oldCat, mapper=mapper)
        tmpCat[fluxField + '_snr'][:] = tmpCat[fluxField + '_instFlux'] / tmpCat[fluxField + '_instFluxErr']
        catalog.extend(tmpCat, False)

        multiMatch.add(catalog, dataId=dataId)
        print("Loaded:", visit, ccd, len(tmpCat))

    matchCat = multiMatch.finish()
    allMatches = lsst.afw.table.GroupView.build(matchCat)
    print("Found matches, groups:", len(matchCat), len(allMatches))

    goodMatches = filter_matches(allMatches, fluxField)
    print("Good groups:", len(goodMatches))

    averageCoord = goodMatches.aggregate(averageRaDecFromCat,
                                         dtype=[('ra', np.float64), ('dec', np.float64)])
    distance = goodMatches.aggregate(positionRmsFromCat) * u.milliarcsecond
    return goodMatches, averageCoord, distance


def count_ccds(goodMatches):
    """Count how many objects are on each ccd."""
    counts = collections.defaultdict(int)
    for group in goodMatches.groups:
        for x in group:
            counts[x['ccd']] += 1
    print("ccd counts:", ', '.join("%s: %s"%(k, v) for k, v in counts.items()))
    return counts


def compute_errors(counts, goodMatches, averageCoord, ccd):
    """Return the ra/dec error for each centroided object."""
    xx = np.zeros(counts[ccd])
    yy = np.zeros(counts[ccd])
    uu = np.zeros(counts[ccd])
    vv = np.zeros(counts[ccd])
    centroid = 'base_SdssCentroid_'
    i = 0
    for group, coord in zip(goodMatches.groups, averageCoord):
        good = group['ccd'] == ccd
        n = good.sum()
#         print(n, group['ccd'], group[good]['ccd'])
        xx[i:i + n] = group[good][centroid+'x']
        yy[i:i + n] = group[good][centroid+'y']
        uu[i:i + n] = group[good]['coord_ra'] - coord[0]
        vv[i:i + n] = group[good]['coord_dec'] - coord[1]
        i += n
    uu = (uu*u.radian).to_value(u.milliarcsecond)
    vv = (vv*u.radian).to_value(u.milliarcsecond)
    return xx, yy, uu, vv


def uv_mean(bbox, xx, yy, uu, vv):
    """Compute the mean of uu and vv on a grid within bbox."""
    nx = 20
    ny = 40
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
    scale = 0.2 if 'mean' in label else 1

    fig = plt.figure(figsize=(6, 10))
    ax = fig.add_subplot(111)

    Q = ax.quiver(xx, yy, uu, vv, units='x', pivot='tail', scale=scale, width=7,
                  headwidth=4, clip_on=False, color=color)
    length = 5/scale if 'mean' in label else 100
    ax.quiverkey(Q, 0.85, 0.90, length, '%s mas'%(length), angle=45,
                 coordinates='figure', labelpos='W', fontproperties={'size': 24})

    plt.title('{}'.format(ccd))
    filename = "plots/quiver-%s-%s.png"%(ccd, label)
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-j", "--jointcal", action='store_true',
                        help="Use jointcal output to update the coordinates.")
    args = parser.parse_args()
    name = 'jointcal' if args.jointcal else 'single'

    butler = lsst.daf.persistence.Butler(datadir)

    bbox, fluxField, newSchema, mapper, multiMatch = prep_matching(butler, visits, ccds)
    goodMatches, averageCoord, distance = do_match(multiMatch, butler, visits, ccds,
                                                   fluxField, newSchema, mapper, useJointcal=args.jointcal)
    counts = count_ccds(goodMatches)

    for ccd in ccds: #[41,42,40,51]:
        xx, yy, uu, vv = compute_errors(counts, goodMatches, averageCoord, ccd)

        filename = "pickles/quiverData-%s-%s.pickle"%(name, ccd)
        with open(filename, 'wb') as outfile:
            pickle.dump((xx, yy, uu, vv, bbox, ccd),
                        outfile,
                        protocol=pickle.HIGHEST_PROTOCOL)

        plot_quiver(xx, yy, uu, vv, ccd, 'all-'+name)
        xMean, yMean, uMean, vMean = uv_mean(bbox, xx, yy, uu, vv)
        plot_quiver(xMean, yMean, uMean, vMean, ccd, 'mean-'+name)


if __name__ == "__main__":
    main()
