import gym
from Elements import DQN, ReplayBuffer, SketchEnv
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import RLDataset

# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

# ------------------------------- #
# 全局变量
# ------------------------------- #

epochs=100

batch_size = 10
capacity = batch_size*10  # 经验池容量
lr = 2e-3  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.9  # 贪心系数
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

states=torch.zeros((10,6,768))
labels=torch.zeros(10).long()

for epoch in range(epochs):

    # 训练模型
    # 遍历所有序列
    for cur_sequence_index in tqdm(range(states.size()[0])):  # 100回合
        # 每个回合开始前重置环境
        # state = env.reset()  # len=4
        
        # 记录每个回合的回报
        episode_return = 0
        done = False
        
        #对一个序列中的所有状态进行遍历
        
        for cur_state_index in range(states[cur_sequence_index].size()[0]):
            if cur_state_index==0:
                state=env.reset(states[cur_sequence_index][0])

            # 获取当前状态下需要采取的动作
            action = agent.take_action(state)
            # 更新环境
            reward, done = env.step(action,labels[cur_sequence_index],states[cur_sequence_index].size()[0],cur_state_index)

            if cur_state_index==states[cur_sequence_index].size()[0]-1:
                next_state=torch.zeros(768)
            else:
                next_state=states[cur_sequence_index][cur_state_index+1]
            # 添加经验池

            replay_buffer.add(state, action, reward, next_state, done)
            # 更新当前状态
            
            state = next_state
            # 更新回合回报
            episode_return += sum(reward)

            # 当经验池超过一定数量后，训练网络
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
            # 找到目标就结束
            if done: break
        
        # 记录每个回合的回报
        return_list.append(episode_return)

        # 更新进度条信息
        if (cur_sequence_index+1)%10==0:
            print(sum(return_list[cur_sequence_index+1-10:-1])/10)

# 绘图
episodes_list = list(range(len(return_list)))
