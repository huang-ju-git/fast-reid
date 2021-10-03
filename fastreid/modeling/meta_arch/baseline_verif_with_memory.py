# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads, build_verif_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
import random
import itertools

@META_ARCH_REGISTRY.register()
class BaselineVerifMemBank(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        self.bank = torch.zeros(cfg.DATALOADER.PERSON_NUMBER, 2560).cuda() # TODO: add config for this place
        self.bank_test = torch.zeros(cfg.DATALOADER.PERSON_NUMBER_TEST, 2560).cuda() # TODO: add config for this place
        # id数*2048
        self.bank.requires_grad = False
        self.bank_test.requires_grad = False
        # bank的作用是降低运算量，bank存储每个id的平均特征

        # # backbone
        # self.backbone = build_backbone(cfg) #用于提取特征

        # head
        self.heads = build_heads(cfg)

        self.verif_model = build_verif_heads(cfg) # 20210419: (TianyuZhang) Currently, it is a classifier with the same setting as self.heads, except numclass is 2

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # images = self.preprocess_image(batched_inputs)
        # _, features = self.backbone(images)
        features, targets = batched_inputs
        features.cuda()
        # targets.cuda()

        if self.training:
            # assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            # targets = batched_inputs["targets"].to(self.device)

            # # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # # may be larger than that in the original dataset, so the circle/arcface will
            # # throw an error. We just set all the targets to 0 to avoid this problem.
            # if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets) #backbone的特征要经过一个多分类器的模块来训练

            length=len(targets)
            for feat, label in zip(outputs["features"],targets):
                self.bank[label] = self.bank[label] + feat.detach()/length

            bank_features = []
            bank_targets = []
            all_features={} #key为label，value为list
            for feat, label in zip(outputs["features"],targets):
                if label not in all_features.keys():
                    all_features[label]=[]
                all_features[label].append(feat)
                for i in range(self.bank.size(0)): #遍历每个id
                    if i!=label and random.randint(1,10)<4:
                       continue # speed up training. too slow, can't bear it anymore
                    # if i!=label:
                    #     continue
                    bank_features.append(torch.pow(feat- self.bank[i],2))
                    bank_targets.append(0 if i!=label else 1)

            for key in all_features.keys(): #对于每一个id
                temp_features=all_features[key]
                if len(temp_features)>1:
                    #新增随机正样本
                    rand_len=random.randint(2,len(temp_features))
                    iter1=list(itertools.combinations(temp_features, rand_len))
                    for idx,element in enumerate(iter1):
                        avg_feature=[]
                        for term in element:
                            avg_feature=avg_feature+term/rand_len
                        avg_feature=torch.Tensor(avg_feature)
                        temp_features.append(avg_feature)
                    iter = list(itertools.combinations(temp_features, 2))
                    for idx,(f1,f2) in enumerate(iter):
                        bank_features.append(torch.pow(f1- f2,2))
                        bank_targets.append(1)



            bank_features = torch.stack(bank_features).to(self.device)

            bank_targets = torch.tensor(bank_targets).to(self.device)
            
            bank_outputs = self.verif_model(bank_features)
            #verif_outputs,pn_targets = None, None
            #for feat, label in zip(outputs["features"],targets):
            #    self.bank[label] = self.bank[label]*0.5 + feat.detach()*0.5
            self.outputs = outputs
            self.targets = targets
            return {
                "outputs": outputs,
                "targets": targets,
                #"verif_outputs": verif_outputs,
                #"verif_targets": pn_targets,
                "bank_outputs": bank_outputs,
                "bank_targets":bank_targets
            }
        else:
            outputs = self.heads(features)

            length=len(targets)
            for feat, label in zip(outputs,targets):
                self.bank_test[label] = self.bank_test[label] + feat.detach()/length

            bank_features = []
            bank_targets = []
            all_features={} #key为label，value为list
            for feat, label in zip(outputs,targets):
                if label not in all_features.keys():
                    all_features[label]=[]
                all_features[label].append(feat)
                for i in range(self.bank_test.size(0)): #遍历每个id
                    if i!=label and random.randint(1,10)<4:
                       continue # speed up training. too slow, can't bear it anymore
                    # if i!=label:
                    #     continue
                    bank_features.append(torch.pow(feat- self.bank_test[i],2))
                    bank_targets.append(0 if i!=label else 1)

            for key in all_features.keys(): #对于每一个id，获取所有正样本
                if len(all_features[key])>1:
                    iter = list(itertools.combinations(all_features[key], 2))
                    for idx,(f1,f2) in enumerate(iter):
                        bank_features.append(torch.pow(f1- f2,2))
                        bank_targets.append(1)        
            
            bank_features = torch.stack(bank_features).to(self.device)
            bank_targets = torch.tensor(bank_targets).to(self.device)
            bank_outputs = self.verif_model(bank_features)
            

            return bank_outputs, bank_targets

    def update(self):
        for feat, label in zip(self.outputs["features"],self.targets):
            self.bank[label] = self.bank[label]*0.8 + feat.detach()*0.2

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]
        # model predictions
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on
        #verif_outputs = outs["verif_outputs"]
        #if verif_outputs is not None:
        #    verif_cls_outputs = verif_outputs["cls_outputs"]
        #    verif_targets = outs["verif_targets"]
        # Log prediction accuracy
        #    log_accuracy(verif_outputs['pred_class_logits'].detach(), verif_targets)

        bank_cls_outputs = outs["bank_outputs"]['cls_outputs']
        bank_targets  = outs["bank_targets"]


        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        # 用于解决正样本多负样本少的问题，在计算每一个样本loss的时候增加权重，偏移越大的权重越大，越要加强优化
        if "VerificationLoss" in loss_names:
            #loss_dict["loss_verif"] = cross_entropy_loss(
            #    verif_cls_outputs,
            #    verif_targets,
            #    0.0,
            #    self._cfg.MODEL.LOSSES.CE.ALPHA,
            #) * self._cfg.MODEL.LOSSES.CE.SCALE 
            loss_dict["loss_bank_verif"] = focal_loss(
                bank_cls_outputs,
                bank_targets,
                alpha=1.0,gamma=5.0,reduction="sum"
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet"] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict["loss_circle"] = pairwise_circleloss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE

        if "Cosface" in loss_names:
            loss_dict["loss_cosface"] = pairwise_cosface(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.COSFACE.MARGIN,
                self._cfg.MODEL.LOSSES.COSFACE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.COSFACE.SCALE

        return loss_dict

