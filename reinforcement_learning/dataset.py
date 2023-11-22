
import os
import os.path as osp
import json
import torch
import PIL.Image as Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RLDataset(Dataset):

    def __init__(self, mode):

        self.feature_dataset_dir="/home/wsy/sketch/code_6/1/preprocess_procedure1/checkpoint/st"

        #/feature_test1.pth

        self.feature_dataset_path=os.path.join(self.feature_dataset_dir,"feature_"+mode+".pth")
        
        self.feature_dataset = torch.load(self.feature_dataset_path)

        self.feature_dataset_keys = list(self.feature_dataset.keys())
            
    def __len__(self):
        return len(self.feature_dataset_keys)

    def __getitem__(self, idx):
        
        cur_feature=self.feature_dataset[self.feature_dataset_keys[idx]]

        #/home/wsy/sketch/code_6/dataset/dataset2/dataset/png/025/C02501P10S022-0-8.png

        stroke_num=self.feature_dataset_keys[idx].split("-")[-1].split(".")[0]
        cur_stroke_index=self.feature_dataset_keys[idx].split("-")[-2]

        if int(cur_stroke_index)+1==int(stroke_num):
            next_state_feature=torch.zeros((1024))
            next_state_key=None

        else:
            next_state_key=self.feature_dataset_keys[idx].split("-")[-3]+"-"+str(int(cur_stroke_index)+1)+"-"+stroke_num+".png"
            next_state_feature=self.feature_dataset[next_state_key]
        
        category_label=int(self.feature_dataset_keys[idx].split("/")[-1][1:4])

        if int(cur_stroke_index)+1==int(stroke_num):
            done=torch.tensor(1).bool()
        else:
            done=torch.tensor(0).bool()

        return cur_feature,next_state_feature, category_label, done


if __name__ == '__main__':
    a=RLDataset("train")
    print(a[0][0].size())