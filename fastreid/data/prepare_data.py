import pickle
import numpy as np
from torch import nn
import torch

# with open("msmt17-train-face.pkl","rb") as f: 
#     msmt_dict=pickle.load(f)

# keys=list(msmt_dict.keys())
# print(len(keys))
# keys_np=np.array(keys)
# np.save('msmt17-test-face-keys',keys_np) 

# features=[]
# for term in keys:
#     features.append(msmt_dict[term])
# features_np=np.array(features)
# np.save('msmt17-test-face-features',features_np) 

pool_layer = nn.AdaptiveAvgPool2d(1)
features=torch.randn(32,2560).reshape(32,2560,1)
global_feat = pool_layer(features)
print(global_feat.shape)

torch.nn.AdaptiveAvgPool2d