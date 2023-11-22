import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import gym
from Elements import DQN, ReplayBuffer, SketchEnv
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import dataloader_fn, BDataset
from torch.utils.data.dataloader import DataLoader
from model import model


dataset=BDataset("test1")
dataloader=DataLoader(dataset=dataset,batch_size=32,shuffle=True)

batch_size = 32
capacity = batch_size*5  # 经验池容量
lr = 2e-2  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 贪心系数
target_update = 200  # 目标网络的参数的更新频率
n_hidden = 128  # 隐含层神经元个数
min_size = 400  # 经验池超过200后再训练
return_list = []  # 记录每个回合的回报
top_k = 5
# 加载环境
env = SketchEnv()

n_states = env.observation_space  # 状态的通道数768
n_actions = env.action_space  # 动作的类别数 374

device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

agent = DQN(n_states=n_states,
            n_hidden=n_hidden,
            n_actions=n_actions,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device,
        )

agent.q_net.cuda()
agent.target_q_net.cuda()

agent.q_net.load_state_dict(torch.load("/home/wsy/sketch/code_6/2/code/checkpoint/0/19_qnet.pth"))

right_state=0
all_state=0
for index,(cur_state,next_state,done,cur_state_len,next_state_len,category_label,cur_state_name,next_state_name) in enumerate(tqdm(dataloader)):
    cur_states_name=[]
    next_states_name=[]

    for i in range(done.size()[0]):
            cur_state_name_=[]
            next_state_name_=[]
            for j in range(cur_state_len[i]):
                cur_state_name_.append(cur_state_name[j][i])
            for j in range(next_state_len[i]):
                next_state_name_.append(next_state_name[j][i])
            cur_states_name.append(cur_state_name_)
            next_states_name.append(next_state_name_)


    cur_state=cur_state.cuda()
    next_state=next_state.cuda()
    done=done.cuda()
    cur_state_len=cur_state_len.cuda().long()
    next_state_len=next_state_len.cuda().long()
    category_label=category_label.cuda().long()
    
    
    action = agent.take_action(cur_state,cur_state_len)
    reward, done, predict_list = env.step(action,category_label,done)

    

    right_state+=torch.sum(predict_list).cpu().item()
    all_state+=done.size()[0]

print(right_state/all_state)

    # print(action)
    # print(category_label)
    # print(reward)
    # print(done)
    # breakpoint()