# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys

import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from torch.backends import cudnn
import pickle
import os.path as osp

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo

os.environ['CUDA_VISIBLE_DEVICES']='1'

# import some modules added in project like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        # default="./configs/MSMT17/bagtricks_R101-ibn.yml"
        default="./configs/MSMT17/bagtricks_R50.yml"
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
        default="/home/huangju/dataset/msmt17/test/*.jpg"
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        #default="MODEL.WEIGHTS ./checkpoints/msmt_bot_R101-ibn.pth",
        #default="MODEL.WEIGHTS ./checkpoints/msmt_bot_R50.pth",
        nargs=argparse.REMAINDER,
    )
    return parser


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    # pathss=[]
    # dir_path="/home/huangju/dataset/dml_ped25-v2/dml_ped25-v2/"
    # ids=["F2", "S2", 'R2', 'T2'] 
    # for term in ids:
    #     new_root=dir_path+term
    #     img_paths = glob.glob(osp.join(new_root, '*/*.jpg'))
    #     for img_path in img_paths:
    #         if img_path.split('/')[-1].split('.')[0].split("_")[-1]!="face":
    #             pathss.append(img_path)

    pathss=[]
    dir_path="/home/huangju/dataset/msmt17/test"
    img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))
    for img_path in img_paths:
        pathss.append(img_path)

    msmt_dict={}
    for path in tqdm(pathss):
        img = cv2.imread(path)
        # print("img!!")
        # print(img)
        feat = demo.run_on_image(img)
        feat = postprocess(feat)
        key="test/"+path.split("/")[-1]
        # print(key)
        # print(feat)
        msmt_dict[key]=feat[0]
    
    #32621张训练集人体图像
    with open("msmt17-test.pkl","wb") as f: 
        pickle.dump(msmt_dict,f)

    # with open("msmt17-train.pkl","rb") as f: 
    #     msmt_dict=pickle.load(f)
    #     print(len(msmt_dict))

    
# 运行命令：python demo/demo.py --opts MODEL.WEIGHTS checkpoints/msmt_bot_R50.pth