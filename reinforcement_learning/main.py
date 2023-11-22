import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import gym
from Elements import DQN, ReplayBuffer, SketchEnv
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import RLDataset
from torch.utils.data.dataloader import DataLoader

# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

# ------------------------------- #
# 全局变量
# ------------------------------- #

epochs=100

batch_size = 256
capacity = batch_size*100  # 经验池容量
lr = 2e-3  # 学习率
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


test_dataset=RLDataset("train1")
test_dataloader=DataLoader(test_dataset,batch_size=batch_size,num_workers=24,shuffle=True)

max_acc=0
for epoch in range(epochs):

    # 训练模型
    # 遍历所有序列
    episode_return = 0
    right_state=0
    all_state=0
    for cur_state, next_state, category_label, done in tqdm(test_dataloader):
        cur_state=cur_state.cuda()
        next_state=next_state.cuda()
        category_label=category_label.cuda()
        done = done.cuda()

        action = agent.take_action(cur_state)
        action=action.squeeze()

        reward, done, predict_list = env.step(action,category_label,done)

        replay_buffer.add(cur_state, action, reward, next_state, done)

        if replay_buffer.size() == capacity:
                # 从经验池中随机抽样作为训练集
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                # 构造训练集
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'dones': d,
                }

                # 网络更新
                agent.update(transition_dict)
        right_state+=torch.sum(predict_list).cpu().item()
        all_state+=batch_size
        
        # 更新回合回报
        episode_return += torch.sum(reward).cpu().item()
    
    if right_state/all_state>max_acc:
        max_acc=right_state/all_state 
        torch.save(agent.q_net.state_dict(),"/home/wsy/sketch/code_6/1/reinforcement_learning/best.pth")
    print("acc: ",right_state/all_state)
    print(episode_return)


# 绘图
episodes_list = list(range(len(return_list)))
