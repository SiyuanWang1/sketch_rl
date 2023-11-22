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
import argparse
# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

# ------------------------------- #
# 全局变量
# ------------------------------- #

epochs=20

batch_size = 32
capacity = batch_size*1000  # 经验池容量
lr = 2e-4  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 贪心系数
target_update = 5  # 目标网络的参数的更新频率
n_hidden = 128  # 隐含层神经元个数
min_size = 400  # 经验池超过200后再训练
return_list = []  # 记录每个回合的回报
top_k = 5

# 加载环境
env = SketchEnv()

n_states = env.observation_space  # 状态的通道数768
n_actions = env.action_space  # 动作的类别数 374



# 实例化经验池
replay_buffer = ReplayBuffer(capacity,n_hidden,top_k)
# 实例化DQN
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

model=model().cuda()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--checkpoint", type=str, default="0")
    args = parser.parse_args()

    if len(args.checkpoint)==0:
        print("input checkpoint!")
        exit()

    w_dir="./checkpoint"
    if not(os.path.isdir(os.path.join(w_dir,args.checkpoint))):
        os.makedirs(os.path.join(w_dir,args.checkpoint))

    test_dataset=BDataset("test1")
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=8,shuffle=True)


    w_path=open(os.path.join(w_dir,args.checkpoint,"acc.txt"),"w")

    max_acc=0
    for epoch in range(epochs):
        print(epoch)
        episode_return = 0
        right_state=0
        all_state=0
        for index,(cur_state,next_state,done,cur_state_len,next_state_len,category_label,cur_state_name,next_state_name) in enumerate(tqdm(test_dataloader)):
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
            
            replay_buffer.add(cur_states_name, action, reward, next_states_name, done, cur_state_len, next_state_len)

            if replay_buffer.size()==capacity:
                state,action,reward,next_state,done,state_graph_batch,next_state_graph_batch,cur_state_len,next_state_len=replay_buffer.sample(batch_size)

                agent.update(state,action,reward,next_state,done,state_graph_batch,next_state_graph_batch,cur_state_len,next_state_len)

            right_state+=torch.sum(predict_list).cpu().item()
            all_state+=done.size()[0]

            if index%300==0:
                print(right_state/all_state)

            episode_return += torch.sum(reward).cpu().item()
        torch.save(agent.q_net.state_dict(),os.path.join(w_dir,args.checkpoint,str(epoch)+"_qnet.pth"))
        if right_state/all_state>max_acc:
            print("best_acc:",right_state/all_state)
            max_acc=right_state/all_state
            torch.save(agent.q_net.state_dict(),os.path.join(w_dir,args.checkpoint,"best_qnet.pth"))

        print("acc: ",right_state/all_state)
        print(episode_return)
        
        w_path.write(str(right_state/all_state)+"   "+str(episode_return)+"\n")


