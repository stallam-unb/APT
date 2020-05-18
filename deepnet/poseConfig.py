from __future__ import division
from builtins import object
from past.utils import old_div
import os
import re
import localSetup
import numpy as np
import copy
import logging

class config(object):
    # ----- Names

    # ----- Network parameters
    def __init__(self):
        self.rescale = 1  # how much to downsize the base image.
        self.label_blur_rad = 3.  # 1.5

        self.batch_size = 8
        self.view = 0
        self.gamma = 0.1
        self.display_step = 50
        self.num_test = 8
        self.dl_steps = 60000 # number of training iters
        self.decay_steps = 25000
        self.learning_rate = 0.0001
        # rate will be reduced by gamma every decay_step iterations.

        # range for contrast, brightness and rotation adjustment
        self.trx_align_theta = True
        self.horz_flip = False
        self.vert_flip = False
        self.brange = [-0.2, 0.2]
        self.crange = [0.7, 1.3]
        self.rrange = 30
        self.trange = 10
        self.scale_range = 0.1
        self.scale_factor_range = 1.1
        # KB 20191218 - if scale_factor_range is read in, use that
        # otherwise, if scale_range is read in, use that
        self.use_scale_factor_range = True
        self.imax = 255.
        self.check_bounds_distort = True
        self.adjust_contrast = False
        self.clahe_grid_size = 20
        self.normalize_img_mean = False
        self.normalize_batch_mean = False
        self.perturb_color = False
        self.flipLandmarkMatches = {}
        self.learning_rate_multiplier = 1.

        # ----- Data parameters
        # l1_cropsz = 0
        self.splitType = 'frame'
        self.trainfilename = 'train_TF'
        self.fulltrainfilename = 'fullTrain_TF'
        self.valfilename = 'val_TF'
        self.valdatafilename = 'valdata'
        self.valratio = 0.3
        self.holdoutratio = 0.8
        self.max_n_animals = 1
        self.flipud = False

        # ----- UNet params
        self.unet_rescale = 1
        #self.unet_steps = 20000
        self.unet_keep_prob = 1.0 # essentially don't use it.
        self.unet_use_leaky = False #will change it to True after testing.
        self.use_pretrained_weights = True

        # ----- MDN params
        self.mdn_min_sigma = 3. # this should just be maybe twice the cell size??
        self.mdn_max_sigma = 4.
        self.mdn_logit_eps_training = 0.001
        self.mdn_extra_layers = 1
        self.mdn_use_unet_loss = True
        self.mdn_pred_dist = True

        # ----- OPEN POSE PARAMS
        self.op_label_scale = 8
        self.op_im_pady = None  # computed at runtime
        self.op_im_padx = None  # "
        self.op_imsz_hires = None  # "
        self.op_imsz_lores = None  # "
        self.op_imsz_net = None  # "
        self.op_imsz_pad = None  # "
        self.op_backbone = 'resnet50_8px'
        self.op_backbone_weights = 'imagenet'
        self.op_map_lores_blur_rad = 1.0
        self.op_map_hires_blur_rad = 2.0
        self.op_paf_lores_tubewidth = 0.95 # not used if tubeblur=True
        self.op_paf_lores_tubeblur = False
        self.op_paf_lores_tubeblursig = 0.95
        self.op_paf_lores_tubeblurclip = 0.05
        self.op_paf_nstage = 5
        self.op_map_nstage = 1
        self.op_hires = True
        self.op_hires_ndeconv = 2
        self.op_base_lr = 4e-5  # Gines 5e-5
        self.op_weight_decay_kernel = 5e-4
        self.op_hmpp_floor = 0.1
        self.op_hmpp_nclustermax = 1
        self.op_pred_raw = False
        self.n_steps = 4.41

        # ---
        #self.sb_rescale = 1
        self.sb_n_transition_supported = 5  # sb network in:out size can be up to 2**<this> (as factor of 2). this
            # is for preproc/input pipeline only; see sb_output_scale for actual ratio
        self.sb_im_pady = None  # computed at runtime
        self.sb_im_padx = None  # "
        self.sb_imsz_net = None  # "
        self.sb_imsz_pad = None  # "
        self.sb_base_lr = 4e-5
        self.sb_weight_decay_kernel = 5e-4
        self.sb_backbone = 'ResNet50_8px'
        self.sb_backbone_weights = 'imagenet'
        self.sb_num_deconv = 3
        self.sb_deconv_num_filt = 512
        self.sb_output_scale = None  # output heatmap dims relative to imszuse (network input size), computed at runtime
        self.sb_upsamp_chan_handling = 'direct_deconv'  # or 'reduce_first'
        self.sb_blur_rad_input_res = 3.0  # target hmap blur rad @ input resolution
        self.sb_blur_rad_output_res = None  # runtime-computed
        self.sb_hmpp_floor = 0.1
        self.sb_hmpp_nclustermax = 1

        # ------ Leap params
        self.leap_net_name = "leap_cnn"

        # ----- Deep Lab Cut
        self.dlc_train_img_dir = 'deepcut_train'
        self.dlc_train_data_file = 'deepcut_data.mat'
        self.dlc_augment = True

        # ---- dpk
        # "early" here is eg after initial setup in APT_interface
        self.dpk_max_val_batches = 1       # maximum number of validation batches
        self.dpk_downsample_factor = 2      # (immutable after early) integer downsample                                            *power* for output shape
        self.dpk_n_stacks = 2
        self.dpk_growth_rate = 48
        self.dpk_use_pretrained = True
        self.dpk_n_outputs = 1              # (settable at TGTFR._call_-time)
        self.dpk_use_augmenter = False      # if true, use dpk_augmenter if distort=True
        self.dpk_augmenter = None           # iaa obj
        self.dpk_n_transition_min = 5       # target n_transition=this; in practice could be more if imsz is perfect power of 2 etc
        self.dpk_im_pady = None             # auto-computed
        self.dpk_im_padx = None             # auto-computed
        self.dpk_imsz_net = None            # auto-computed
        self.dpk_imsz_pad = None            # auto-computed
        self.dpk_use_graph = True           # (immutable after early) bool
        self.dpk_graph = None               # (immutable after early)
        self.dpk_swap_index = None          # (immutable after early)
        self.dpk_graph_scale = 1.0          # (immutable after early) float, scale factor                                           applied to grp/limb/global confmaps
        self.dpk_output_shape = None        # (computed at TGTFR/init) conf map output shape
        self.dpk_output_sigma = None        # (computed at TGTFR/init) target hmap gaussian                                         sd in output coords
        self.dpk_input_sigma = 5.0          # (immutable after early) target hmap gaussian                                          sd in input coords
        self.dpk_base_lr_factory = .001
        self.dpk_base_lr_used = None        # (auto-computed at compile-time; actual base lr used)
        self.dpk_reduce_lr_on_plat = True   # True is as published for dpk, using K cbk (starting from dpk_base_lr_used);
                                            # False is APT-style scheduled (using learning_rate, decay_steps, gamma)


        # ============== EXTRA ================

        # ----- Time parameters
        self.time_window_size = 1
        self.do_time = False

        # ------ RNN Parameters
        self.rnn_before = 9
        self.rnn_after = 0

        # ------------ ATTention parameters
        self.att_hist = 128
        self.att_layers = [1] # use layer this far from the middle (top?) layers.

        # ----- Save parameters

        self.save_time = None
        self.save_step = 2000
        self.save_td_step = 100
        self.maxckpt = 30
        self.cachedir = None

        # ----- Legacy
        # self.scale = 2
        # self.numscale = 3
        # self.pool_scale = 4
        # self.pool_size = 3
        # self.pool_stride = 2
        # self.cos_steps = 2 #number of times the learning rate is decayed
        # self.step_size = 100000 # not used anymore


    def set_exp_name(self, exp_name):
        self.expname = exp_name
        # self.baseoutname = self.expname + self.baseName
        # self.baseckptname = self.baseoutname + 'ckpt'
        # self.basedataname = self.baseoutname + 'traindata'
        # self.fineoutname = self.expname + self.fineName
        # self.fineckptname = self.fineoutname + 'ckpt'
        # self.finedataname = self.fineoutname + 'traindata'
        # self.mrfoutname = self.expname + self.mrfName
        # self.mrfckptname = self.mrfoutname + 'ckpt'
        # self.mrfdataname = self.mrfoutname + 'traindata'


    def getexpname(self, dirname):
        return os.path.basename(os.path.dirname(dirname)) + '_' + os.path.splitext(os.path.basename(dirname))[0]

    def getexplist(self, L):
        return L['movieFilesAll'][self.view,:]

    def get(self,name,default):
        if hasattr(self,name):
            logging.info('OVERRIDE: Using {} with value {} from config '.format(name,getattr(self,name)))
        else:
            logging.info('DEFAULT: For {} using with default value {}'.format(name, default))
            setattr(self,name,default)
        return getattr(self,name,default)


# -- alice fly --

aliceConfig = config()
aliceConfig.cachedir = os.path.join(localSetup.bdir, 'cache', 'alice')
#aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_20170925_cv.lbl')
# aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_20180107.lbl') # round1
# aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_expandedbehavior_20180425_local.lbl')
aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_expandedbehavior_20180425.lbl')
def alice_exp_name(dirname):
    return os.path.basename(os.path.dirname(dirname))

aliceConfig.getexpname = alice_exp_name
aliceConfig.has_trx_file = True
aliceConfig.imsz = (180, 180)
aliceConfig.selpts = np.arange(0, 17)
aliceConfig.img_dim = 1
aliceConfig.n_classes = len(aliceConfig.selpts)
aliceConfig.splitType = 'frame'
aliceConfig.set_exp_name('aliceFly')
aliceConfig.trange = 5
aliceConfig.nfcfilt = 128
aliceConfig.sel_sz = 144
aliceConfig.num_pools = 1
aliceConfig.dilation_rate = 2
# aliceConfig.pool_scale = aliceConfig.pool_stride**aliceConfig.num_pools
# aliceConfig.psz = aliceConfig.sel_sz / 4 / aliceConfig.pool_scale / aliceConfig.dilation_rate
aliceConfig.valratio = 0.25
# aliceConfig.mdn_min_sigma = 70.
# aliceConfig.mdn_max_sigma = 70.
aliceConfig.adjust_contrast = False
aliceConfig.clahe_grid_size = 10
aliceConfig.brange = [0,0]
aliceConfig.crange = [1.,1.]
aliceConfig.mdn_extra_layers = 1
aliceConfig.normalize_img_mean = False
aliceConfig.mdn_groups = [range(17)]


aliceConfig_time = copy.deepcopy(aliceConfig)
aliceConfig_time.do_time = True
aliceConfig_time.cachedir = os.path.join(localSetup.bdir, 'cache','alice_time')


aliceConfig_rnn = copy.deepcopy(aliceConfig)
aliceConfig_rnn.cachedir = os.path.join(localSetup.bdir, 'cache','alice_rnn')
aliceConfig_rnn.batch_size = 2
# aliceConfig_rnn.trainfilename_rnn = 'train_rnn_TF'
# aliceConfig_rnn.fulltrainfilename_rnn = 'fullTrain_rnn_TF'
# aliceConfig_rnn.valfilename_rnn = 'val_rnn_TF'

# -- felipe bees --

felipeConfig = config()
felipeConfig.cachedir = os.path.join(localSetup.bdir, 'cache','felipe')
felipeConfig.labelfile = os.path.join(localSetup.bdir,'data','felipe','doesnt_exist.lbl')
def felipe_exp_name(dirname):
    return dirname

def felipe_get_exp_list(L):
    return 0

felipeConfig.getexpname = felipe_exp_name
felipeConfig.getexplist = felipe_get_exp_list
felipeConfig.view = 0
felipeConfig.imsz = (300, 300)
felipeConfig.selpts = np.arange(0, 5)
felipeConfig.img_dim = 3
felipeConfig.n_classes = len(felipeConfig.selpts)
felipeConfig.splitType = 'frame'
felipeConfig.set_exp_name('felipeBees')
felipeConfig.trange = 20
felipeConfig.nfcfilt = 128
felipeConfig.sel_sz = 144
felipeConfig.num_pools = 2
felipeConfig.dilation_rate = 1
# felipeConfig.pool_scale = felipeConfig.pool_stride**felipeConfig.num_pools
# felipeConfig.psz = felipeConfig.sel_sz / 4 / felipeConfig.pool_scale / felipeConfig.dilation_rate


##  -- felipe multi bees

# -- felipe bees --

felipe_config_multi = config()
felipe_config_multi.cachedir = os.path.join(localSetup.bdir, 'cache', 'felipe_m')
felipe_config_multi.labelfile = os.path.join(localSetup.bdir, 'data', 'felipe_m', 'doesnt_exist.lbl')
def felipe_exp_name(dirname):
    return dirname

def felipe_get_exp_list(L):
    return 0

felipe_config_multi.getexpname = felipe_exp_name
felipe_config_multi.getexplist = felipe_get_exp_list
felipe_config_multi.view = 0
felipe_config_multi.imsz = (360, 380)
felipe_config_multi.selpts = np.array([1, 3, 4])
felipe_config_multi.img_dim = 3
felipe_config_multi.n_classes = len(felipe_config_multi.selpts)
felipe_config_multi.splitType = 'frame'
felipe_config_multi.set_exp_name('felipeBeesMulti')
felipe_config_multi.trange = 20
felipe_config_multi.nfcfilt = 128
felipe_config_multi.sel_sz = 256
felipe_config_multi.num_pools = 2
felipe_config_multi.dilation_rate = 1
# felipe_config_multi.pool_scale = felipe_config_multi.pool_stride ** felipe_config_multi.num_pools
# felipe_config_multi.psz = felipe_config_multi.sel_sz / 4 / felipe_config_multi.pool_scale / felipe_config_multi.dilation_rate
felipe_config_multi.max_n_animals = 17
