from torch.utils.data import DataLoader,Dataset

import torch
import numpy as np
import pickle


# train_face_feature=np.load("msmt17-train-face-features.npy")
# train_label=np.load("msmt17-train-face-keys.npy")

# test_face_feature=np.load("msmt17-test-face-features.npy")
# test_label=np.load("msmt17-test-face-keys.npy")

# with open("msmt17-train.pkl","rb") as f: 
#     train_body_dict=pickle.load(f)

# with open("msmt17-test.pkl","rb") as f: 
#     test_body_dict=pickle.load(f)

class mydataset(Dataset):
    def __init__(self,face_feature,label,body_dict):
        self.face_feature = face_feature #一个list，按顺序存储人脸特征，和label相对应
        self.label = label #一个list，按顺序存储人脸特征对应的key
        self.body_dict = body_dict #存储所有人体特征的字典，key和label对应，格式为：train/0715_016_01_0302afternoon_1346_5.jpg
    
    def __getitem__(self,item):
        face=self.face_feature[item]
        key=self.label[item]
        body=self.body_dict[key]
        feature=np.append(body,face)
        label=key.split('/')[-1].split('_')[0]
        # label=np.array(label)
        # label=torch.from_numpy(label)
        label = torch.from_numpy(np.fromstring(label, dtype=int, sep=','))
        return torch.tensor(feature),label  #根据需要进行设置

    def __len__(self):
        return len(self.face_feature)



def get_train_dataloader():
    train_face_feature=np.load("msmt17-train-face-features.npy")
    train_label=np.load("msmt17-train-face-keys.npy")
    with open("msmt17-train.pkl","rb") as f: 
        train_body_dict=pickle.load(f)
    dataset = mydataset(train_face_feature,train_label,train_body_dict)
    train_loader= DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    return train_loader

def get_test_dataloader():
    test_face_feature=np.load("msmt17-test-face-features.npy")
    test_label=np.load("msmt17-test-face-keys.npy")
    with open("msmt17-test.pkl","rb") as f: 
        test_body_dict=pickle.load(f)
    dataset = mydataset(test_face_feature,test_label,test_body_dict)
    test_loader= DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    return test_loader

# data_loader=get_test_dataloader()
# 每个idx都是一个batch的序号，batch的数量向下取整
# inputs[0]是32张图片的特征，inputs[1]是32张图片对应的标签，即id
# for idx, inputs in enumerate(data_loader):
#     features, targets = inputs
#     print(inputs.type)
#     print(inputs.shape)
#     break

# data_loader_iter = iter(data_loader)
# data = next(data_loader_iter)
# print(data.type)
# print(data.shape)

label='1'
label = torch.from_numpy(np.fromstring(label, dtype=int, sep=','))
print(label[0])
print(label[0]==1)
l=[1,2]
l[label[0]]=5
print(l)


