
import os
import os.path as osp
import json
import torch
import PIL.Image as Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader


class BDataset(Dataset):

    def __init__(self, mode):
        
        self.max_stroke_num=35

        self.json_content=json.load(open(os.path.join("/home/wsy/sketch/code_6/dataset/dataset2/dataset/txt/train_test2",mode+".json"),"r"))
        self.json_content_keys=list(self.json_content.keys())
        self.transforms=transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.json_content_keys)
    
    def __getitem__(self, idx):

        cur_state_name=self.json_content[self.json_content_keys[idx]][0]
        next_state_name=self.json_content[self.json_content_keys[idx]][1]

        category_label=int(cur_state_name[0][1:4])-1

        #当前的状态 35*3*224*224，其中第一个维度需要填充到35
        cur_state=torch.zeros((35,3,224,224))
        
        
        for index,i in enumerate(cur_state_name):
            cur_path=os.path.join("/home/wsy/sketch/code_6/dataset/dataset2/dataset/png",i[1:4],i)
            cur_img=Image.open(cur_path,"r")
            cur_img=self.transforms(cur_img)

            cur_state[index,:,:,:]=cur_img[:,:,:]
        

        #当前的状态 35*3*224*224，其中第一个维度需要填充到35
        next_state=torch.zeros((35,3,224,224))

        if len(next_state_name)!=0:
            for index,i in enumerate(next_state_name):
                cur_path=os.path.join("/home/wsy/sketch/code_6/dataset/dataset2/dataset/png",i[1:4],i)
                cur_img=Image.open(cur_path,"r")
                cur_img=self.transforms(cur_img)

                next_state[index,:,:,:]=cur_img[:,:,:]

            done=torch.tensor(0).bool()

        else:
            done=torch.tensor(1).bool()


        cur_state_len=torch.tensor(len(cur_state_name))
        next_state_len=torch.tensor(len(next_state_name))

        cur_state_name_=[]
        next_state_name_=[]

        for i in range(self.max_stroke_num):
            if i <len(cur_state_name):
                cur_state_name_.append(cur_state_name[i])
            else:
                cur_state_name_.append("")

            if i < len(next_state_name):
                next_state_name_.append(next_state_name[i])
            else:
                next_state_name_.append("")
        
        return cur_state,next_state,done,cur_state_len,next_state_len,torch.tensor(category_label),cur_state_name_,next_state_name_
        # return {
        #     "cur_state":cur_state,"next_state":next_state,"done":done,
        #     "cur_state_len":torch.tensor(len(cur_state_name)),
        #     "next_state_len":torch.tensor(len(next_state_name)),
        #     "category_label":torch.tensor(category_label),
        #     "cur_state_name":cur_state_name,
        #     "next_state_name":next_state_name
        # }

def dataloader_fn(batch):

    cur_state=torch.zeros((1,35,3,224,224))
    next_state=torch.zeros((1,35,3,224,224))
    done=torch.zeros((1)).bool()
    cur_state_len=torch.zeros((1))
    next_state_len=torch.zeros((1))
    category_label=torch.zeros((1))
    cur_state_name=[]
    next_state_name=[]

    for i in batch:
        
        cur_state=torch.cat([cur_state,i['cur_state'].unsqueeze(0)],dim=0)
        next_state=torch.cat([next_state,i['next_state'].unsqueeze(0)],dim=0)
        done=torch.cat([done,i['done'].unsqueeze(0)],dim=0)
        cur_state_len=torch.cat([cur_state_len,i['cur_state_len'].unsqueeze(0)],dim=0)
        next_state_len=torch.cat([next_state_len,i['next_state_len'].unsqueeze(0)],dim=0)
        category_label=torch.cat([category_label,i['category_label'].unsqueeze(0)],dim=0)
        cur_state_name.append(i['cur_state_name'])
        next_state_name.append(i['next_state_name'])

    cur_state=cur_state[1:,:,:,:,:]
    next_state=next_state[1:,:,:,:,:]
    done=done[1:]
    cur_state_len=cur_state_len[1:]
    next_state_len=next_state_len[1:]
    category_label=category_label[1:]

    

    return cur_state,next_state,done,cur_state_len,next_state_len,category_label,cur_state_name,next_state_name

if __name__ == '__main__':
    a=BDataset("test1")
    b=DataLoader(a,batch_size=24)
    for cur_state,next_state,done,cur_state_len,next_state_len,category_label,cur_state_name,next_state_name in b:
        state_name=[]
        # for i in range(len(cur_state_name)):
        for i in range(24):
            cur_state_name_=[]
            
            for j in range(cur_state_len[i]):
                cur_state_name_.append(cur_state_name[j][i])
            state_name.append(cur_state_name_)
        
        breakpoint()
    # print(a[0])
    # print(a[0][-1])