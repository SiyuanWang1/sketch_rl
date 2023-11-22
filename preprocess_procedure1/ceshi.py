import torch
import numpy 

a=torch.load("/home/wsy/sketch/code_6/1/preprocess_procedure1/checkpoint/st/feature_test1.pth")
print(a[list(a.keys())[0]].size())