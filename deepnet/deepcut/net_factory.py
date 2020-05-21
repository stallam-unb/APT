"""
Modified by Mayank Kabra
Adapted from DeepLabCut2.0 Toolbox (deeplabcut.org)
copyright A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

"""
from nnet.pose_net import PoseNet


def pose_net(cfg):
    cls = PoseNet
    return cls(cfg)
