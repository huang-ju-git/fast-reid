# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from torch import nn

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class VerifHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = 1024 #cfg.MODEL.HEADS.EMBEDDING_DIM

        num_classes   = 2
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck   = True
        norm_type     = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.pool_layer = nn.Identity()
        elif pool_type == "flatten":     self.pool_layer = Flatten()
        else:                            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        #if with_bnneck:
        #    bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*bottleneck)

        # identity classification layer
        # fmt: off
        #if cls_type == 'linear':          self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        #elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        ##elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        #elif cls_type == 'cosSoftmax':    self.classifier = CosSoftmax(cfg, feat_dim, num_classes)
        #else:                             raise KeyError(f"{cls_type} is not supported!")
        # fmt: on
        self.classifier = nn.Sequential(nn.Linear(feat_dim,512),
                                         nn.BatchNorm1d(512),
                                         nn.Dropout(0.2),
                                         
                                         nn.PReLU(),
                                         nn.Linear(512,256),
                                         nn.BatchNorm1d(256),
                                         nn.Dropout(0.2),
                                         nn.PReLU(),
                                         nn.Linear(256,128),
                                         nn.BatchNorm1d(128),
                                         nn.Dropout(0.2),
                                         nn.PReLU(),
                                         nn.Linear(128,64),
                                         nn.BatchNorm1d(64),
                                         nn.Dropout(0.2),
                                         nn.PReLU(),
                                         nn.Linear(64,32),
                                         nn.BatchNorm1d(32),
                                         nn.Dropout(0.2),
                                         nn.PReLU(),
                                         nn.Linear(32,16),
                                         nn.BatchNorm1d(16),
                                         nn.Dropout(0.2),
                                         nn.PReLU(),
                                         nn.Linear(16,2,bias=False))
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.drop = nn.Dropout(0.5)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = features.view(features.size(0),features.size(1),1,1)
        # global_feat = self.drop(global_feat)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        #bn_feat = self.drop(bn_feat)


        # Training
        if True or self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
            pred_class_logits = cls_outputs #F.linear(bn_feat, self.classifier.weight)

        # fmt: off
        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "features": feat,
        }
