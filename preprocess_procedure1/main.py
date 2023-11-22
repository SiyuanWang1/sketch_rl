import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3" 
import torch
from dataset import ADataset,dataloader_fn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import model
import argparse

epochs=20
lr=0.02

w_dir="/home/wsy/sketch/code_6/1/preprocess_procedure1/checkpoint"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--checkpoint", type=str, default="0")
    args = parser.parse_args()

    if len(args.checkpoint)==0:
        print("input checkpoint!")
        exit()

    if not(os.path.isdir(os.path.join(w_dir,args.checkpoint))):
        os.makedirs(os.path.join(w_dir,args.checkpoint))

    w_path=os.path.join(w_dir,args.checkpoint,"acc.txt")
    
    w_file=open(w_path,"w")

    train_dataset=ADataset("train1")
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=24,collate_fn=dataloader_fn)

    test_dataset=ADataset("test1")
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=32,num_workers=24,collate_fn=dataloader_fn)

    model=model().cuda()
    optimizer=optim.Adam(model.parameters())
    loss_fn=torch.nn.CrossEntropyLoss()

    max_acc=0
    for epoch in range(10):

        right=0
        all=0

        model.train()
        for png,num,recog_label,graph_batch in tqdm(train_dataloader):

            png=png.cuda()
            num=num.cuda()
            recog_label=recog_label.cuda()
            graph_batch=graph_batch.cuda()

            
            optimizer.zero_grad()
            predict,_=model(png,num,graph_batch)

            loss=loss_fn(predict,recog_label)
            loss.backward()
            optimizer.step()

            output=torch.argmax(predict,dim=1)
            right+=torch.sum((output==recog_label).long()).item()
            all+=recog_label.size()[0]

        print(right/all)
        right=0
        all=0
        model.eval()
        with torch.no_grad():
            for png,num,recog_label,graph_batch in tqdm(test_dataloader):

                png=png.cuda()
                num=num.cuda()
                recog_label=recog_label.cuda()
                graph_batch=graph_batch.cuda()

                
                predict,_=model(png,num,graph_batch)

                output=torch.argmax(predict,dim=1)

                right+=torch.sum((output==recog_label).long()).item()
                all+=recog_label.size()[0]
        print(right/all)
        w_file.write(str(right/all)+"\n")

        if (right/all)>max_acc:
            max_acc=right/all
            print(max_acc)
            torch.save(model.state_dict(),os.path.join(w_dir,args.checkpoint,"best_model.pth"))
            