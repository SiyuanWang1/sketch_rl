
import os
import os.path as osp
import json
import torch
import PIL.Image as Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ADataset(Dataset):

    def __init__(self, mode):

        self.max_stroke_num=35
        
        self.dir="/home/wsy/sketch/code_6/dataset/dataset2/dataset"

        self.png_path=os.path.join(self.dir,"png")

        self.transforms=transforms.Compose([transforms.ToTensor()])
        
        #拿到json文件
        self.json_path=os.path.join(self.dir,"txt",mode+".json")
        self.json_content=json.load(open(self.json_path,"r"))
        self.json_keys=list(self.json_content.keys())
            
    def __len__(self):
        return len(self.json_keys)

    def __getitem__(self, idx):
        
        json_key=self.json_keys[idx]
        json_value=self.json_content[json_key]
        
        pngs=torch.zeros((1,3,224,224))

        sketch_ids=[]

        #取每一个笔画的图像
        for stroke_index in range(len(json_value)):
            cur_stroke_path=os.path.join(self.png_path,json_key[1:4],json_value[stroke_index])

            cur_img=Image.open(cur_stroke_path,"r")

            cur_img=self.transforms(cur_img)
            
            cur_img=cur_img.unsqueeze(0)
            pngs=torch.cat([pngs,cur_img],dim=0)

            sketch_ids.append(cur_stroke_path)
        
        pngs=pngs[1:]

        #每一个草图的都是76*3*224*224
        padding_pngs=torch.zeros((self.max_stroke_num-len(json_value),3,224,224))
        pngs=torch.cat([pngs,padding_pngs],dim=0)

        category=torch.tensor(int(json_key[1:4])-1).long()

        
        return {
            'pngs':pngs,
            'stroke_num':len(json_value),
            'recog_label':category,
            'sketch_ids':sketch_ids
        }

def dataloader_fn(batch):

    pngs=torch.zeros((1,3,224,224))
    stroke_num=[]
    recog_label=[]
    
    sketch_ids=[]

    graph_batch=torch.zeros((1))
    for item_index,item in enumerate(batch):
        cur_png=item['pngs']
        cur_stroke_num=item['stroke_num']
        cur_recog_label=item['recog_label']

        cur_sketch_id=batch[item_index]['sketch_ids']
        sketch_ids.append(cur_sketch_id)

        pngs=torch.cat([pngs,cur_png[:cur_stroke_num,:,:,:]],dim=0)
        stroke_num.append(cur_stroke_num)
        recog_label.append(cur_recog_label)

        graph_batch=torch.cat([graph_batch,torch.repeat_interleave(torch.tensor(item_index),repeats=cur_stroke_num)],dim=0)

    pngs=pngs[1:,:,:,:]
    stroke_num=torch.tensor(stroke_num).long()
    recog_label=torch.tensor(recog_label).long()
    graph_batch=graph_batch[1:].long()
    return pngs,stroke_num,recog_label,graph_batch,sketch_ids


if __name__ == '__main__':
    a=ADataset("train")
    print(a[0][0].size())