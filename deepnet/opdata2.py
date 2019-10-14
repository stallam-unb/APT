import sys
import os
import numpy as np
import tensorflow as tf
import logging

import matplotlib.pyplot as plt
from itertools import islice
import time

import PoseTools
import heatmap

ISPY3 = sys.version_info >= (3, 0)

def create_affinity_labels(locs, imsz, graph, tubewidth=1.0):
    """
    Create/return part affinity fields

    locs: (nbatch x npts x 2) (x,y) locs, 0-based. (0,0) is the center of the
        upper-left pixel.
    imsz: [2] (nr, nc) size of affinity maps to create/return

    graph: (nlimb) array of 2-element tuples; connectivity/skeleton
    tubewidth: width of "limb". *Warning* maybe don't choose tubewidth exactly equal to 1.0

    returns (nbatch x imsz[0] x imsz[1] x nlimb*2) paf hmaps.
        4th dim ordering: limb1x, limb1y, limb2x, limb2y, ...
    """

    nlimb = len(graph)
    nbatch = locs.shape[0]
    out = np.zeros([nbatch, imsz[0], imsz[1], nlimb * 2])
    n_steps = 2 * max(imsz)

    for cur in range(nbatch):
        for ndx, e in enumerate(graph):
            start_x, start_y = locs[cur, e[0], :]
            end_x, end_y = locs[cur, e[1], :]
            assert not (np.isnan(start_x) or np.isnan(start_y) or np.isnan(end_x) or np.isnan(end_y))
            assert not (np.isinf(start_x) or np.isinf(start_y) or np.isinf(end_x) or np.isinf(end_y))

            ll = np.sqrt((start_x - end_x) ** 2 + (start_y - end_y) ** 2)

            if ll == 0:
                # Can occur if start/end labels identical
                # Don't update out/PAF
                continue

            dx = (end_x - start_x) / ll / 2
            dy = (end_y - start_y) / ll / 2
            zz = None
            TUBESTEP = 0.25
            ntubestep = int(2.0 * float(tubewidth) / TUBESTEP + 1)
            # for delta in np.arange(-tubewidth, tubewidth, 0.25):
            for delta in np.linspace(-tubewidth, tubewidth, ntubestep):
                # delta indicates perpendicular displacement from line/limb segment (in px)

                # xx = np.round(np.linspace(start_x,end_x,6000))
                # yy = np.round(np.linspace(start_y,end_y,6000))
                # zz = np.stack([xx,yy])
                xx = np.round(np.linspace(start_x + delta * dy, end_x + delta * dy, n_steps))
                yy = np.round(np.linspace(start_y - delta * dx, end_y - delta * dx, n_steps))
                if zz is None:
                    zz = np.stack([xx, yy])
                else:
                    zz = np.concatenate([zz, np.stack([xx, yy])], axis=1)
                # xx = np.round(np.linspace(start_x-dy,end_x-dy,6000))
                # yy = np.round(np.linspace(start_y+dx,end_y+dx,6000))
                # zz = np.concatenate([zz,np.stack([xx,yy])],axis=1)
            # zz now has all the pixels that are along the line.
            # or "tube" of width tubewidth around limb
            zz = np.unique(zz, axis=1)
            # zz now has all the unique pixels that are along the line with thickness==tubewidth.
            dx = (end_x - start_x) / ll
            dy = (end_y - start_y) / ll
            for x, y in zz.T:
                xint = int(round(x))
                yint = int(round(y))
                if xint < 0 or xint >= out.shape[2] or yint < 0 or yint >= out.shape[1]:
                    continue
                out[cur, yint, xint, ndx * 2] = dx
                out[cur, yint, xint, ndx * 2 + 1] = dy

    return out

def rescale_points(locs_hires, scale):
    '''
    Rescale (x/y) points to a lower res

    :param locs_hires: (nbatch x npts x 2) (x,y) locs, 0-based. (0,0) is the center of the upper-left pixel.
    :param scale: downsample factor. eg if 2, the image size is cut in half
    :return: (nbatch x npts x 2) (x,y) locs, 0-based, rescaled (lo-res)
    '''

    bsize, npts, d = locs_hires.shape
    assert d == 2
    assert issubclass(locs_hires.dtype.type, np.floating)
    locs_lores = (locs_hires - float(scale - 1) / 2) / scale
    return locs_lores

def unscale_points(locs_lores, scale):
    '''
    Undo rescale_points

    :param locs_lores:
    :param scale:
    :return:
    '''

    bsize, npts, d = locs_lores.shape
    assert d == 2
    assert issubclass(locs_lores.dtype.type, np.floating)
    locs_hires = float(scale) * (locs_lores + 0.5) - 0.5
    return locs_hires

def data_generator(conf, db_type, distort, shuffle, debug=False):
    if db_type == 'val':
        filename = os.path.join(conf.cachedir, conf.valfilename) + '.tfrecords'
    elif db_type == 'train':
        filename = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
    else:
        raise IOError('Unspecified DB Type')  # KB 20190424 - py3

    batch_size = conf.batch_size
    vec_num = len(conf.op_affinity_graph)
    heat_num = conf.n_classes
    N = PoseTools.count_records(filename)

    # Py 2.x workaround nested functions outer variable rebind
    # https://www.python.org/dev/peps/pep-3104/#new-syntax-in-the-binding-outer-scope
    class Namespace:
        pass

    ns = Namespace()
    ns.iterator = None

    def iterator_reset():
        if ns.iterator:
            ns.iterator.close()
        ns.iterator = tf.python_io.tf_record_iterator(filename)
        # print('========= Resetting ==========')

    def iterator_read_next():
        if not ns.iterator:
            ns.iterator = tf.python_io.tf_record_iterator(filename)
        try:
            if ISPY3:
                record = next(ns.iterator)
            else:
                record = ns.iterator.next()
        except StopIteration:
            iterator_reset()
            if ISPY3:
                record = next(ns.iterator)
            else:
                record = ns.iterator.next()
        return record

    while True:
        all_ims = []
        all_locs = []
        all_info = []
        for b_ndx in range(batch_size):
            # AL: this 'shuffle' seems weird
            n_skip = np.random.randint(30) if shuffle else 0
            for _ in range(n_skip + 1):
                record = iterator_read_next()

            example = tf.train.Example()
            example.ParseFromString(record)
            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            depth = int(example.features.feature['depth'].int64_list.value[0])
            expid = int(example.features.feature['expndx'].float_list.value[0])
            t = int(example.features.feature['ts'].float_list.value[0])
            img_string = example.features.feature['image_raw'].bytes_list.value[0]
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((height, width, depth))
            locs = np.array(example.features.feature['locs'].float_list.value)
            locs = locs.reshape([conf.n_classes, 2])
            if 'trx_ndx' in example.features.feature.keys():
                trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0])
            else:
                trx_ndx = 0
            info = np.array([expid, t, trx_ndx])

            all_ims.append(reconstructed_img)
            all_locs.append(locs)
            all_info.append(info)

        ims = np.stack(all_ims)  # [bsize x height x width x depth]
        locs = np.stack(all_locs)  # [bsize x ncls x 2]
        info = np.stack(all_info)  # [bsize x 3]

        assert conf.op_rescale == 1, \
            "Need further mods/corrections below for op_rescale~=1"
        assert conf.op_label_scale == 8, \
            "Expected openpose scale of 8"  # Any value should be ok tho

        ims, locs = PoseTools.preprocess_ims(ims, locs, conf,
                                             distort, conf.op_rescale)
        # locs has been rescaled per op_rescale (but not op_label_scale)

        imszuse = conf.imszuse
        (imnr_use, imnc_use) = imszuse
        ims = ims[:, 0:imnr_use, 0:imnc_use, :]

        # Needed for VGG pretrained weights which expect imgdepth of 3
        assert conf.img_dim == ims.shape[-1]
        if conf.img_dim == 1:
            ims = np.tile(ims, 3)

        # locs -> PAFs, MAP
        # Generates hires maps here but only used below if conf.op_hires
        dc_scale = conf.op_hires_ndeconv**2
        locs_lores = rescale_points(locs, conf.op_label_scale)
        locs_hires = rescale_points(locs, conf.op_label_scale // dc_scale)
        imsz_lores = [int(x / conf.op_label_scale / conf.op_rescale) for x in imszuse]
        imsz_hires = [int(x / conf.op_label_scale * dc_scale / conf.op_rescale) for x in imszuse]
        label_map_lores = heatmap.create_label_hmap(locs_lores, imsz_lores, conf.op_map_lores_blur_rad)
        label_map_hires = heatmap.create_label_hmap(locs_hires, imsz_hires, conf.op_map_hires_blur_rad)

        label_paf_lores = create_affinity_labels(locs_lores, imsz_lores,
                                                 conf.op_affinity_graph,
                                                 tubewidth=conf.op_paf_lores_tubewidth)

        npafstg = conf.op_paf_nstage
        nmapstg = conf.op_map_nstage
        targets = [label_paf_lores,] * npafstg + [label_map_lores,] * nmapstg
        if conf.op_hires:
            targets.append(label_map_hires)

        if debug:
            yield [ims], targets, locs, info
        else:
            yield [ims], targets
            # (inputs, targets)

if __name__ == "__main__":

    # class Timer(object):
    #     def __init__(self, name=None):
    #         self.name = name
    #
    #     def __enter__(self):
    #         self.tstart = time.time()
    #
    #     def __exit__(self, type, value, traceback):
    #         if self.name:
    #             print('[%s]' % self.name,)
    #         print('Elapsed: %s' % (time.time() - self.tstart))
    #
    #
    # tf.enable_eager_execution()

    import nbHG
    print "OPD MAIN!"

    conf = nbHG.createconf(nbHG.lblbub, nbHG.cdir, 'cvi_outer3_easy__split0', 'bub', 'openpose', 0)
    #conf.op_affinity_graph = conf.op_affinity_graph[::2]
    conf.imszuse = (176, 176)
    # dst, dstmd, dsv, dsvmd = create_tf_datasets(conf)
    ditrn = data_generator(conf, 'train', True, True, debug=True)
    dival = data_generator(conf, 'val', False, False, debug=True)

    xtrn = [x for x in islice(ditrn,5)]
    xval = [x for x in islice(dival,5)]

    #imstrn, pafsmapstrn, locstrn, mdtrn = zip(*xtrn)
    #mdlintrn = zip(imstrn, pafsmapstrn)

    #imsval, pafsmapsval, locsval, mdval = zip(*xval)
    #mdlinval = zip(imsval, pafsmapsval)

    #ds1, ds2, ds3 = create_tf_datasets(conf)
    #
    #
    # if True:
    #     x1 = [x for x in ds1.take(10)]
    #     x2 = [x for x in ds2.take(10)]
    #     x3 = [x for x in ds3.take(10)]
    #     #locs10 = [x for x in dslocsinfo.take(10)]
    # else:
    #     dst10 = [x for x in dst.take(1)]
    #     dst10md = [x for x in dstmd.take(1)]
    #     dsv10 = [x for x in dsv.take(1)]
    #     dsv10md = [x for x in dsvmd.take(1)]

    # N = 100
    # with Timer('tf.data'):
    #     xds = [x for x in dst.take(N)]
    # with Timer('it'):
    #     xit = []
    #     for i in range(N):
    #         xit.append(ditrn.next())



    # ds2,ds3,ds4 = test_dataset_with_rand()




# locs = np.array([[0,0],[0,1.5],[0,4],[0,6.9],[0,7.],[0,7.49],[0,7.51],[1,2],[10,12],[16,16]])
# locs = locs[np.newaxis,:,:]
# imsz = (48, 40)
# locsrs = rescale_points(locs, 8)
# imszrs = (6, 5)
#
# import matplotlib.pyplot as plt
# hm1 = create_label_images_with_rescale(locs,imsz,8,3)
# hm2 = heatmap.create_label_hmap(locsrs, imszrs, 3)
