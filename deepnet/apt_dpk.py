from __future__ import division
from __future__ import print_function

import os
import math
import matplotlib.pyplot as plt
import numpy as np
import datetime
import tensorflow as tf
import json
import logging
import shutil
import argparse
import sys
import pickle
import importlib
import ast
import copy
import contextlib

import tensorflow.keras as tfk
import keras.backend as K
import imgaug.augmenters as iaa
import imgaug as ia
import deepposekit.io.DataGenerator
import deepposekit.io.TrainingGenerator
from deepposekit.augment import FlipAxis
import deepposekit.callbacks
from deepposekit.models import StackedDenseNet

import TrainingGeneratorTFRecord as TGTFR
import tfdatagen as opd
import open_pose4 as op4
import heatmap as hm
import PoseTools
import multiResData
import kerascallbacks
import poseConfig
import APT_interface as apt
import run_apt_expts as rae
import deepposekit.io.utils as dpkut
import deepposekit.utils.keypoints
import util



bubtouchroot = '/groups/branson/home/leea30/apt/ar_flybub_touching_op_20191111'
lblbubtouch = os.path.join(bubtouchroot,'20191125T170226_20191125T170453.lbl')
cvitouch = os.path.join(bubtouchroot,'cvi_trn4702_tst180.mat')
kwtouch = '20191125_base_trn4702tst180'
cdirtouch = os.path.join(bubtouchroot, 'cdir' + kwtouch)
outtouch = os.path.join(bubtouchroot, 'out' + kwtouch)
exptouch = 'cvi_trn4702_tst180__split1' # trn4702, tst180

cacheroot = '/nrs/branson/al/cache'

isotri='/groups/branson/home/leea30/apt/dpk20191114/isotri.png'
isotrilocs = np.array([[226., 107.], [180., 446.], [283., 445.]])
isotriswapidx = np.array([-1, 2, 1])

skeleton_csvs = {
    'alice': ['/dat0/jrcmirror/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_dpk_skeleton.csv'],
    'stephen': [
            '/dat0/jrcmirror/groups/branson/bransonlab/apt/experiments/data/sh_dpk_skeleton_vw0_side.csv',
            '/dat0/jrcmirror/groups/branson/bransonlab/apt/experiments/data/sh_dpk_skeleton_vw1_front.csv',
        ],
    'romain': [
            '/dat0/jrcmirror/groups/branson/bransonlab/apt/experiments/data/romain_dpk_skeleton_vw0.csv',
            '/dat0/jrcmirror/groups/branson/bransonlab/apt/experiments/data/romain_dpk_skeleton_vw1.csv',
        ],
    'roian': ['/dat0/jrcmirror/groups/branson/bransonlab/apt/experiments/data/roian_dpk_skeleton.csv'],
    'larva': ['/dat0/jrcmirror/groups/branson/bransonlab/apt/experiments/data/larva_dpk_skeleton.csv'],
    'dpkfly':    ['/dat0/jrcmirror/groups/branson/home/leea30/git/dpkd/datasets/fly/skeleton.csv'],
    'dpklocust': ['/dat0/jrcmirror/groups/branson/home/leea30/git/dpkd/datasets/locust/skeleton.csv'],
    'dpkzebra':  ['/dat0/jrcmirror/groups/branson/home/leea30/git/dpkd/datasets/zebra/skeleton.csv'],
}


def viz_targets(ims, tgts, npts, ngrps, ibatch=0):
    '''

    :param ims: [nb x h x w x nchan] images generated by generator
    :param tgts: [nb x hds x wds x nmap] Hmap targets generated by generator
    :return:
    '''

    #n_keypoints = data_generator.keypoints_shape[0]
    n_keypoints = npts

    #batch = train_generator(batch_size=1, validation=False)[1]
    #inputs = batch[0]
    #outputs = batch[1]
    inputs = ims
    outputs = tgts

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    ax1.set_title('image')
    ax1.imshow(inputs[ibatch, ..., 0], cmap='gray', vmin=0, vmax=255)

    ax2.set_title('posture graph')
    ax2.imshow(outputs[ibatch, ..., n_keypoints:-1].max(-1))

    ax3.set_title('keypoints confidence')
    ax3.imshow(outputs[ibatch, ..., :n_keypoints].max(-1))

    ax4.set_title('posture graph and keypoints confidence')
    ax4.imshow(outputs[ibatch, ..., -1], vmin=0)
    plt.show()

    #kp
    minkpthmap = np.min(outputs[ibatch, ..., :npts])
    maxkpthmap = np.max(outputs[ibatch, ..., :npts])
    print("kpthmap min/max is {}/{}".format(minkpthmap, maxkpthmap))
    axnr = 3
    axnc = math.ceil(npts/axnr)
    figkp, axs = plt.subplots(axnr, axnc)
    axsflat = axs.flatten()
    for ipt in range(n_keypoints):
        im = axsflat[ipt].imshow(outputs[ibatch, ..., ipt],
                            vmin=minkpthmap,
                            vmax=maxkpthmap)
        axsflat[ipt].set_title("pt{}".format(ipt))
        if ipt == 0:
            figkp.colorbar(im)
    plt.show()

    #grps
    mingrps = np.min(outputs[ibatch, ..., npts:npts+ngrps])
    maxgrps = np.max(outputs[ibatch, ..., npts:npts+ngrps])
    print("grps min/max is {}/{}".format(mingrps, maxgrps))
    axnr = 1
    axnc = math.ceil(ngrps/axnr)
    figgrps, axs = plt.subplots(axnr, axnc)
    axsflat = axs.flatten()
    for i in range(ngrps):
        im = axsflat[i].imshow(outputs[ibatch, ..., npts+i],
                               vmin=mingrps,
                               vmax=maxgrps)
        axsflat[i].set_title("grp{}".format(i))
        if i == 0:
            figgrps.colorbar(im)
    plt.show()

    # limbs
    nlimbs = outputs.shape[-1] - npts - ngrps - 2
    minlimbs = np.min(outputs[ibatch, ..., npts+ngrps:npts+ngrps+nlimbs])
    maxlimbs = np.max(outputs[ibatch, ..., npts+ngrps:npts+ngrps+nlimbs])
    print("limbs min/max is {}/{}".format(minlimbs, maxlimbs))
    axnr = 3
    axnc = math.ceil(nlimbs / axnr)
    figlimbs, axs = plt.subplots(axnr, axnc)
    axsflat = axs.flatten()
    for i in range(nlimbs):
        im = axsflat[i].imshow(outputs[ibatch, ..., npts+ngrps+i],
                               vmin=minlimbs,
                               vmax=maxlimbs)
        axsflat[i].set_title("limb{}".format(i))
        if i == 0:
            figlimbs.colorbar(im)
    plt.show()

    # globals
    nglobs = 2
    minglobs = np.min(outputs[ibatch, ..., -2:])
    maxglobs = np.max(outputs[ibatch, ..., -2:])
    print("globs min/max is {}/{}".format(minglobs, maxglobs))
    axnr = 1
    axnc = math.ceil(nglobs / axnr)
    figglobs, axs = plt.subplots(axnr, axnc)
    axsflat = axs.flatten()
    for i in range(nglobs):
        im = axsflat[i].imshow(outputs[ibatch, ..., -2+i])
        axsflat[i].set_title("glob{}".format(i))
        figglobs.colorbar(im, ax=axsflat[i])
    plt.show()

    return figkp, figgrps, figlimbs, figglobs

def viz_skel(ims, locs, graph):
    image = ims
    keypoints = locs

    plt.figure(figsize=(5, 5))
    image = image7[0] if image.shape[-1] is 3 else image[0, ..., 0]
    cmap = None if image.shape[-1] is 3 else 'gray'
    plt.imshow(image, cmap=cmap, interpolation='none')
    for idx, jdx in enumerate(graph):
        if jdx > -1:
            plt.plot(
                [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
                [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
                'r-'
            )
    plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1],
                c=np.arange(keypoints.shape[1]),
                s=50,
                cmap=plt.cm.hsv,
                zorder=3)

    plt.show()

def toymodel(nout):
    X = tfk.Input(shape=(10, 10, 1), name='img')
    inputs = [X,]
    outputs = []
    for i in range(nout):
        X = tfk.layers.Conv2D(8, 3,
                              padding='same',
                              activation='relu',
                              name='out{}'.format(i),
                              kernel_initializer='random_uniform')(X)
        outputs.append(X)

    model = tfk.Model(inputs=inputs, outputs=outputs, name='toymodel')

    model.compile("adam", "mse")
    return model

def toy(m):

    nout = len(m.outputs)
    x = tf.constant(np.random.normal(size=(6, 10, 10, 1)))
    y = [tf.constant(np.random.normal(size=(6, 10, 10, 8))) for i in range(nout)]

    yp0 = m.predict_on_batch(x)
    yp1 = m.predict_on_batch(x)
    if nout==1:
        yp0 = [yp0,]
        yp1 = [yp1,]
    losses = m.evaluate(x, y, steps=1)
    with tf.Session().as_default():
        ye0 = [x.eval() for x in y]
    with tf.Session().as_default():
        ye1 = [x.eval() for x in y]

    print('mse of yp0 and yp1 els')
    for x, y in zip(yp0, yp1):
        print(np.mean((x-y)**2))

    print('mse of ye0 and ye1 els')
    for x, y in zip(ye0, ye1):
        print(np.mean((x-y)**2))

    print('losses per evaluate: {}'.format(losses))
    print('losses evaled manually:')
    for x, y in zip(ye0, yp0):
        print(np.mean((x-y)**2))

    return yp0, ye0, losses, x, y

def check_flips(im, locs, dpk_swap_index):
    im_lr = im.copy()
    im_ud = im.copy()
    im_lria = im.copy()
    im_udia = im.copy()
    locs_lr = locs.copy()
    locs_ud = locs.copy()
    locs_lria = locs.copy()
    locs_udia = locs.copy()

    augmenter_ud = [FlipAxis(dpk_swap_index, axis=0, p=1.0)]
    augmenter_lr = [FlipAxis(dpk_swap_index, axis=1, p=1.0)]
    augmenter_ud = iaa.Sequential(augmenter_ud)
    augmenter_lr = iaa.Sequential(augmenter_lr)

    im_lr, locs_lr = PoseTools.randomly_flip_lr(im_lr, locs_lr)
    im_ud, locs_ud = PoseTools.randomly_flip_ud(im_ud, locs_ud)
    im_lria, locs_lria = opd.imgaug_augment(augmenter_lr, im_lria, locs_lria)
    im_udia, locs_udia = opd.imgaug_augment(augmenter_ud, im_udia, locs_udia)

    return (im_lr, locs_lr), (im_ud, locs_ud), (im_lria, locs_lria), (im_udia, locs_udia)

#region Callbacks

def create_callbacks(conf, sdn, runname='deepnet'):

    logging.warning("configing callbacks")

    # `Logger` evaluates the validation set( or training set if `validation_split = 0` in the `TrainingGenerator`) at the end of each epoch and saves the evaluation data to a HDF5 log file( if `filepath` is set).
    nowstr = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    logfile = 'log{}.h5'.format(nowstr)
    '''
    logger = deepposekit.callbacks.Logger(
                    filepath=os.path.join(outtouch,logfile),
                    validation_batch_size=10)
    '''

    if conf.dpk_reduce_lr_on_plat:
        logging.info("LR callback: using reduceLROnPlateau")
        lr_cbk = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", # monitor="val_loss"
            factor=0.2,
            verbose=1,
            patience=20)
    else:
        logging.info("LR callback: using APT fixed sched")
        lr_cbk = kerascallbacks.create_lr_sched_callback(
            conf.display_step,
            conf.dpk_base_lr_used,
            conf.gamma,
            conf.decay_steps)

    # training, infinite, shuffling
    train_generator = sdn.train_generator(
        n_outputs=sdn.n_outputs,
        batch_size=conf.batch_size,
        validation=False,
        confidence=True
    )
    # val, infinite, nonshuffled, tgts-as-kpts
    keypoint_generator = sdn.train_generator(
        n_outputs=1,
        batch_size=conf.batch_size,
        validation=True,
        confidence=False,
        infinite=True,  # use "infinite" val generators here, for logging val_dist only
    )
    aptcbk = kerascallbacks.APTKerasCbk(conf, (train_generator, keypoint_generator),
                                        runname=runname)


    # `ModelCheckpoint` automatically saves the model when the validation loss improves at the end of each epoch. This allows you to automatically save the best performing model during training, without having to evaluate the performance manually.
    '''
    ckptfile = 'ckpt{}.h5'.format(nowstr)
    ckpt = os.path.join(outtouch, ckptfile)
    model_checkpoint = deepposekit.callbacks.ModelCheckpoint(
        ckpt,
        monitor="val_loss", # monitor="val_loss"
        verbose=1,
        save_best_only=True,
    )
    '''

    # `EarlyStopping` automatically stops the training session when the validation loss stops improving for a set number of epochs, which is set with the `patience` argument. This allows you to save time when training your model if there's not more improvment.
    '''
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", # monitor="val_loss"
        min_delta=0.001,
        patience=100,
        verbose=1
    )
    '''

    #callbacks = [early_stop, reduce_lr, model_checkpoint, logger]
    callbacks = [lr_cbk, aptcbk]

    return callbacks


#endregion

'''
for our dsets:
1. rapt load/reload, then setup.
2. run_normal_training. this updates common_conf.
3. run_trainining. this calls run_training_conf_helper which copies common_conf and 
tweaks for dset/trainingtype, producing conf_opts (a copied/standalone dict).
4. conf_opts is string-ized and given to APT_interf
5. APT_interf creates a conf via
 conf = create_conf(lblfile, cur_view, name, net_type=net_type, cache_dir=args.cache,conf_params=args.conf_params). This requires a lblfile and also does net-specific 
  tweaks!!!! It also applies/overlays the conf_opts.
6. This conf is now passed onto train routines

for dpk dsets:
- Everything can proceed as above, except we need a create_conf that doesn't require
a lblfile. 

*** RAE+DPK strat ###
For our existing dsets, we will run through rae since that is where everything is
and it needs to be reproduceable/preserved for posterity. rae.setup will, for each
dset, need to configure new/addnl things for dpk: eg the dpk skeleton and swap indices.
Aug-, Train-related params we will assume we can reuse as-is. (So far this is seems
to be justified with our replication exps.)

rae calls out to APT_interf (having str-ized all conf_params). APT_interf will need
to be updated to be able to call out to apt_dpk.py for actual train/predict.

For running on DPK dsets, this needs to be done and proven to ourselves but is less 
core to our ppr/results. We can have a front-end/submit call in rae that calls apt_dpk
in a shell. The idea here is to show that our DPK impl repros the ppr result 
on their dsets; also in particular that using our i) datagen (tfrecords instead of 
h5), ii) augpipe (PoseTools instead of imgaug), and iii) training (simpler 
lr/stopping/etc vs earlystop/bestsave etc) do not make significant differences.

'''

#region Configurations

def compute_padding_imsz_net(imsz, rescale, n_transition_min):
    '''
    From the raw image size, desired rescale, and desired n_transition_min,
    compute the necessary padding and resulting imsz_net (input-to-network-size)

    :param imsz: [2] raw im size
    :param rescale: float, desired rescale.
    :param n_transition_min: desired minimum n_transition
    :return: padx, pady, imsz_pad, imsz_net
    '''

    # in tfdatagen, the input pipeline is read->pad->rescale/distort->ready_for_network

    # we set the padding so the rescale is 'perfect' ie the desired rescale is the one precisely
    # used ie the imsz-after-pad is precisely divisible by rescale

    assert isinstance(rescale, int) or rescale.is_integer(), "Expect rescale to be integral value"

    imsz_pad_should_be_divisible_by = int(rescale * 2 ** n_transition_min)
    dsfac = imsz_pad_should_be_divisible_by
    roundupeven = lambda x: int(np.ceil(x/dsfac)) * dsfac

    imsz_pad = (roundupeven(imsz[0]), roundupeven(imsz[1]))
    padx = imsz_pad[1] - imsz[1]
    pady = imsz_pad[0] - imsz[0]
    imsz_net = (int(imsz_pad[0]/rescale), int(imsz_pad[1]/rescale))

    return padx, pady, imsz_pad, imsz_net

def update_conf_dpk(conf_base,
                    graph,
                    swap_index,
                    n_keypoints=None,  # optional. if not provided conf.n_classes can be already set
                    imshape=None,  # " .imsz, .img_dim "
                    useimgaug=False,
                    imgaugtype='dpkfly',  # used only if useimgaug==True
                    ):
    '''
    Massage a given APT conf for dpk. This mostly sets dpk_* props etc.

    :param conf_base: conf starting pt.
    :return: This returns the same handle as conf_base
    '''

    conf = conf_base

    '''
    # this is prob unnec and could be dumb
    KEEPATTS = [
        'trainfilename', 'valfilename', 'cachedir', 'batch_size',
        'dl_steps', 'display_step', 'save_step',
        'rescale', 'n_classes', 'imsz',
        'adjust_contrast', 'clahe_grid_size', 'horz_flip', 'vert_flip',
        'rrange', 'trange', 'scale_range', 'check_bounds_distort', 'brange', 'crange',
        'imax', 'normalize_img_mean', 'img_dim', 'perturb_color', 'normalize_batch_mean']
    attrs = vars(conf).keys()
    for att in attrs:
        if not att.startswith('dpk_') and att not in KEEPATTS:
            setattr(conf, att, ['__FOO_UNUSED__', ])
    '''

    # stuff that is set from lblfile by apt.create_conf; OR that
    # we are now adding (if no lbl avail)

    if n_keypoints is not None:
        if hasattr(conf, 'n_classes'):
            assert conf.n_classes == n_keypoints
        else:
            conf.n_classes = n_keypoints
    if imshape is not None:
        if hasattr(conf, 'imsz'):
            assert conf.imsz == imshape[:2]
        else:
            conf.imsz = imshape[:2]
        if hasattr(conf, 'img_dim'):
            assert conf.img_dim == imshape[2]
        else:
            conf.img_dim = imshape[2]
    assert hasattr(conf, 'n_classes') and hasattr(conf, 'imsz') and hasattr(conf, 'img_dim')

    conf.dpk_im_padx, conf.dpk_im_pady, conf.dpk_imsz_pad, conf.dpk_imsz_net = \
        compute_padding_imsz_net(conf.imsz, conf.rescale, conf.dpk_n_transition_min)

    logging.info("DPK size stuff: imsz={}, imsz_pad={}, imsz_net={}, rescale={}, n_trans_min={}".format(
        conf.imsz, conf.dpk_imsz_pad, conf.dpk_imsz_net, conf.rescale, conf.dpk_n_transition_min))

    conf.dpk_graph = graph
    conf.dpk_swap_index = swap_index

    conf.dpk_use_augmenter = useimgaug
    conf.dpk_augmenter_type = {'type': imgaugtype} if useimgaug else None

    return conf

def read_skel_csv(skel_csv):
    s = dpkut.initialize_skeleton(skel_csv)
    skeleton = s[["tree", "swap_index"]].values
    graph = skeleton[:, 0]
    swap_index = skeleton[:, 1]
    return graph, swap_index

def swap_index_to_flip_landmark_matches(swap_idx):
    flm = {str(idx): val for (idx, val) in enumerate(list(swap_idx)) if val != -1}
    return flm

def update_conf_dpk_skel_csv(conf_base, skel_csv):
    graph, swap_index = read_skel_csv(skel_csv)
    conf = update_conf_dpk(conf_base, graph, swap_index)
    return conf


def skel_graph_test(ty):
    skel_csv = skeleton_csvs[ty]
    for idxskel, csv in enumerate(skel_csv):
        print("### View {}".format(idxskel))
        graph, swap_index = read_skel_csv(csv)
        # this stuff from dpk.utils.keypoints
        edge_labels = deepposekit.utils.keypoints.graph_to_edges(graph)
        labels = np.unique(edge_labels)
        for idx, label in enumerate(labels):  # loop over groups
            print("  grp {}: rootidx={}".format(idx, label))
            lines = graph[edge_labels == label]  # parent conns for this grp
            lines_idx = np.where(edge_labels == label)[0]  # downstream conns "
            print(np.stack([lines, lines_idx, ]))
            for jdx, (line_idx, line) in enumerate(zip(lines_idx, lines)):
                if line >= 0:  # parent conn exists
                    pass
                else:
                    assert jdx == 0  # the first member of each group must be the parent/root of that group.


def print_dpk_conf(conf):
    PFIXESSKIP = ['unet_', 'mdn_', 'op_', 'sb_','dlc_', 'rnn_',
                  'save_', 'leap_', 'att_', 'clahe_grid',
                  'holdoutratio', 'cos_steps', 'LEAP', 'do_time',
                  'Unet', 'DeepLabCut', 'max_n_animals',
                  'time_window_size', 'selpts', 'fulltrainfilename',
                  'valdatafilename', 'valratio', 'valfilename']
    print("### CONF ###")
    keys = sorted(vars(conf).keys())
    for k in keys:
        if any([k.startswith(x) for x in PFIXESSKIP]):
            continue
        v = getattr(conf, k)
        if isinstance(v, list) and len(v)>0 and v[0] == '__FOO_UNUSED__':
            pass
        else:
            print("{} -> {}".format(k, v))

    print("### CONF END ###")


#endregion

#region Augment

def make_imgaug_augmenter(imgaugtype, data_generator_or_swap_index):
    if imgaugtype == 'dpkfly':
        # Take step3 ex nb, slightly mod to match ppr desc;
        # do not add additional steps described in ppr
        augmenter = []

        augmenter.append(FlipAxis(data_generator_or_swap_index, axis=0))  # flip image up-down
        augmenter.append(FlipAxis(data_generator_or_swap_index, axis=1))  # flip image left-right

        sometimes = []
        sometimes.append(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                                    shear=(-8, 8),
                                    order=ia.ALL,
                                    cval=ia.ALL,
                                    mode=ia.ALL)
                         )
        sometimes.append(iaa.Affine(scale=(0.9, 1.1),
                                    mode=ia.ALL,
                                    order=ia.ALL,
                                    cval=ia.ALL)
                         )
        augmenter.append(iaa.Sometimes(0.75, sometimes))
        augmenter.append(iaa.Affine(rotate=(-180, 180),
                                    mode=ia.ALL,
                                    order=ia.ALL,
                                    cval=ia.ALL)
                         )
        augmenter = iaa.Sequential(augmenter)

    else:
        assert False, "Unimplemented"

    # Noise

    # Dropout

    # Blur/sharpen

    # Contrast

    return augmenter

#endregion

def apt_db_from_datagen(dg, train_tf, val_idx=None, val_tf=None):
    '''
    Create APT-style train/val tfrecords from a DPK DataGenerator
    :param dg: DataGenerator instance
    :param train_tf: full path, training tfrecord to be written; include .tfrecords extension
    :param split_file: (opt) json containing 'val_idx' field that lists 0b row indices for val split
    :param val_tf: (opt) like train_tf, val db if split_file spec'ed
    :return:
    '''

    n = len(dg)

    '''
    dosplit = split_file is not None
    if dosplit:
        with open(split_file) as fp:
            js = json.load(fp)
        val_idx = js['val_idx']
        nval = len(val_idx)
        assert all((x < n for x in val_idx))
        print("Read json file {}. Found {} val_idx elements. Datagenerator has {} els.".format(
            split_file, nval, n))
    '''
    doval = val_idx is not None
    assert doval is (val_tf is not None)

    print("Datagenerator image/keypt shapes are {}, {}.".format(
        dg.compute_image_shape(), dg.compute_keypoints_shape() ) )

    env = tf.python_io.TFRecordWriter(train_tf)
    if doval:
        val_env = tf.python_io.TFRecordWriter(val_tf)

    count = 0
    val_count = 0
    for idx in range(n):
        im = dg.get_images([idx, ])
        loc = dg.get_keypoints([idx, ])
        info = [int(idx), int(idx), int(idx)]

        towrite = apt.tf_serialize([im[0, ...], loc[0, ...], info])
        if doval and idx in val_idx:
            val_env.write(towrite)
            val_count += 1
        else:
            env.write(towrite)
            count += 1

        if idx % 100 == 99:
            print('%d,%d number of examples added to the training db and val db' % (count, val_count))

    print('%d,%d number of examples added to the training db and val db' % (count, val_count))

#region Train

def compile(conf):
    tgtfr = TGTFR.TrainingGeneratorTFRecord(conf)

    sdn = StackedDenseNet(tgtfr,
                          n_stacks=conf.dpk_n_stacks,
                          growth_rate=conf.dpk_growth_rate,
                          pretrained=conf.dpk_use_pretrained)

    base_lr = conf.dpk_base_lr_factory if conf.dpk_reduce_lr_on_plat \
        else conf.learning_rate
    base_lr_used = base_lr * conf.learning_rate_multiplier
    conf.dpk_base_lr_used = base_lr_used
    logging.info("apt_dpk compile: base_lr_used={}".format(base_lr_used))

    optimizer = tf.keras.optimizers.Adam(
        lr=base_lr_used, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    sdn.compile(optimizer=optimizer, loss='mse')

    return tgtfr, sdn

def train(conf,
          create_cbks_fcn=create_callbacks,
          runname='deepnet'):
    print_dpk_conf(conf)

    conf_file = os.path.join(conf.cachedir, '{}.conf.pickle'.format(runname))
    with open(conf_file, 'wb') as fh:
        pickle.dump({'conf': conf}, fh)
    logging.info("Saved conf to {}".format(conf_file))

    tgtfr, sdn = compile(conf)
    cbks = create_cbks_fcn(conf, sdn, runname=runname)

    tgconf = tgtfr.get_config()
    sdnconf = sdn.get_config()
    with open(conf_file, 'ab') as fh:
        pickle.dump({'tg': tgconf, 'sdn': sdnconf}, fh)
    logging.info("Saved other confs to {}".format(conf_file))

    # XXX valbatch stuff, valsteps stuff. See apt_dpk_exps
    nval = tgtfr.n_validation
    nvalbatch = nval // conf.batch_size
    nvalbatch = min(nvalbatch, conf.dpk_max_val_batches)
    logging.info("n_validation={}, n_validationbatch={}".format(nval, nvalbatch))
    assert nvalbatch > 0, 'Number of validation batches must be greater than 0.'
    sdn.fit(
        batch_size=conf.batch_size,
        validation_batch_size=conf.batch_size,
        callbacks=cbks,
        epochs=conf.dl_steps // conf.display_step,
        steps_per_epoch=conf.display_step,
        validation_steps=nvalbatch, )  # default validation_freq of 1 seems fine # validation_freq=10)

#endregion

#region Predict

def get_pred_fn(conf0, model_file, tmr_pred=None):

    assert model_file is not None, "model_file is currently required"

    if tmr_pred is None:
        tmr_pred = contextlib.suppress()

    conf = copy.deepcopy(conf0)
    exp_dir = conf.cachedir
    sdn, conf_saved, _ = load_apt_cpkt(exp_dir, model_file)

    print("Comparing conf to conf_saved:")
    util.dictdiff(vars(conf), vars(conf_saved))

    pred_model = sdn.predict_model

    def pred_fn(imsraw):
        '''

        :param imsraw: BHWC
        :return:
        '''

        bsize = imsraw.shape[0]
        assert bsize == conf.batch_size
        locs_dummy = np.zeros((bsize, conf.n_classes, 2))
        # can do non-distort img preproc
        ims, _, _ = opd.ims_locs_preprocess_dpk_noconf_nodistort(imsraw, locs_dummy, conf, False)

        assert ims.shape[1:3] == conf.dpk_imsz_net
        assert ims.shape[3] == conf.img_dim

        with tmr_pred:
            predres = pred_model.predict(ims)

        locs = predres[..., :2]  # 3rd/last col is confidence
        confidence = predres[..., 2]

        # locs are in imsz_net-space which is post-pad, post rescale
        # Right now we are setting padding in update_conf_dpk so the rescale is precisely used
        locs = PoseTools.unscale_points(locs, conf.rescale, conf.rescale)  # makes a copy, returns new array
        locs[..., 0] -= conf.dpk_im_padx//2  # see tfdatagen.pad_ims_black
        locs[..., 1] -= conf.dpk_im_pady//2

        ret_dict = {}
        ret_dict['locs'] = locs
        ret_dict['confidence'] = confidence
        return ret_dict

    def close_fn():
        K.clear_session()

    return pred_fn, close_fn, model_file

def predict_stuff(sdn, ims, locsgt, hmfloor=0.1, hmncluster=1):
    '''

    :param sdn:
    :param ims: PREPROCESSED ims NOT raw ims
    :param locsgt:
    :param hmfloor:
    :param hmncluster:
    :return:
    '''
    mt = sdn.train_model
    mp = sdn.predict_model

    sdnconf = sdn.get_config()
    npts = sdnconf['keypoints_shape'][0]
    unscalefac = 2**sdnconf['downsample_factor']

    assert False, "need to preproc ims"
    yt = mt.predict(ims)
    yhm = op4.clip_heatmap_with_warn(yt[-1][..., :npts])
    locsTlo = hm.get_weighted_centroids(yhm, hmfloor, hmncluster)
    locsThi = opd.unscale_points(locsTlo, unscalefac)

    locsPhi = mp.predict(ims)
    locsPhi = locsPhi[..., :2]  # 3rd/last col is confidence

    errT = np.sqrt(np.sum((locsgt-locsThi)**2, axis=-1))
    errP = np.sqrt(np.sum((locsgt-locsPhi)**2, axis=-1))

    return errT, errP, locsThi, locsPhi

def conf_load(conf_file):
    '''
    Handles multiple-pickled-dicts in a pickle
    :param conf_file:
    :return:
    '''
    conf = {}
    with open(conf_file, 'rb') as f:
        while True:
            try:
                d = pickle.load(f)
            except EOFError:
                break
            if isinstance(d, dict):
                assert not any([x in conf for x in d])
                conf.update(d)
    return conf


def load_apt_cpkt(exp_dir, mdlfile):
    '''
    Load an APT-style saved model checkpoint
    :param exp_dir:
    :param mdlfile: eg deepnet-25000
    :return:
    '''

    conf_file = os.path.join(exp_dir, 'conf.pickle')
    mdl_wgts_file = os.path.join(exp_dir, mdlfile)

    conf_dict = conf_load(conf_file)
    conf = conf_dict['conf']
    model_config = conf_dict['sdn']

    tgtfr, sdn = compile(conf)

    sdn.__init_train_model__()
    sdn.train_model.load_weights(mdl_wgts_file)
    kwargs = {}
    kwargs["output_shape"] = model_config["output_shape"]
    kwargs["keypoints_shape"] = model_config["keypoints_shape"]
    kwargs["downsample_factor"] = model_config["downsample_factor"]
    kwargs["output_sigma"] = model_config["output_sigma"]
    sdn.__init_predict_model__(**kwargs)

    return sdn, conf, model_config

#endregion

