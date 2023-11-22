import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3" 
import torch
from test_dataset import ADataset,dataloader_fn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import model
import argparse
import json

if __name__ == '__main__':

    mode="train1"

    test_dataset=ADataset(mode)
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=32,num_workers=24,collate_fn=dataloader_fn)

    model=model()
    model.load_state_dict(torch.load("/home/wsy/sketch/code_6/1/preprocess_procedure1/checkpoint/st/best_model.pth"))
    model.cuda()

    right=0
    all=0

    features={}
    
    model.eval()
    with torch.no_grad():
        for png,num,recog_label,graph_batch,sketch_ids in tqdm(test_dataloader):

            png=png.cuda()
            num=num.cuda()
            recog_label=recog_label.cuda()
            graph_batch=graph_batch.cuda()

            
            predict,predict_feature=model(png,num,graph_batch)

            for cur_batch_index in range(predict.size()[0]):
                
                for cur_stroke_index in range(len(sketch_ids[cur_batch_index])):
                    
                    features[sketch_ids[cur_batch_index][cur_stroke_index]]=predict_feature[cur_batch_index,cur_stroke_index,:].cpu()

            output=torch.argmax(predict,dim=1)

            right+=torch.sum((output==recog_label).long()).item()
            all+=recog_label.size()[0]
    print(right/all)

    # w_feature_path=os.path.join("/home/wsy/sketch/code_6/1/preprocess_procedure1/checkpoint/st","feature_"+mode+".pth")
    # torch.save(features,w_feature_path)

            