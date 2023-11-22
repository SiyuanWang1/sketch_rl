import torch
import torch.nn as nn
import torchvision.models as models
from torch_geometric.utils import to_dense_batch

from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class model(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_size=1024
        
        self.cnn_=torch.nn.Sequential(*(list(models.resnet18().children())[:-1]))

        self.rnn_= nn.GRU(input_size=512,hidden_size=1024,num_layers=3, batch_first=True)

        self.fc=nn.Sequential(
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,374)
        )

    def forward(self,pngs,sketch_pngs_num,graph_batch):
        
    

        # pngs,graph_batch=self.split_pngs(pngs,sketch_pngs_num)

        x=self.cnn_(pngs)
        x=x.squeeze()

        x,mask=to_dense_batch(x,graph_batch)

        _, idx_sort = torch.sort(sketch_pngs_num, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        x=x.index_select(0,idx_sort)
        length_list=sketch_pngs_num[idx_sort]
        
        x=pack_padded_sequence(x,lengths= length_list.cpu(),batch_first=True)
        
        x0,x1=self.rnn_(x)
        x,_=pad_packed_sequence(x0,batch_first=True)
        
        x=x.index_select(0,idx_unsort)

        x_features_=x
        x_features=torch.sum(x, dim=1)

        x=self.fc(x_features)
        
        return x,x_features_
    
    #先把b*76*png改成，n*png，n为当前batch所有的草图真实笔画数量
    def split_pngs(self,pngs,sketch_pngs_num):
        
        
        re_pngs=torch.zeros((1,pngs.size()[-3],pngs.size()[-2],pngs.size()[-1])).cuda()

        #graph_batch=[0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,...],每位上表示当前的图像属于哪个草图
        graph_batch=torch.zeros((1)).cuda()
        

        for cur_batch in range(pngs.size()[0]):

            cur_stroke_pngs=pngs[cur_batch,:sketch_pngs_num[cur_batch],:,:,:]

            re_pngs=torch.cat([re_pngs,cur_stroke_pngs],dim=0)

            graph_batch=torch.cat([graph_batch,torch.repeat_interleave(torch.tensor(cur_batch).cuda(),repeats=sketch_pngs_num[cur_batch])],dim=0)

        re_pngs=re_pngs[1:]
        graph_batch=graph_batch[1:].long()
        
        return re_pngs,graph_batch