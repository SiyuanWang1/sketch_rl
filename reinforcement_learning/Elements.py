import gym
from gym import spaces
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import collections
import random

class SketchEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self):
        self.xth = 0
        self.target_x = 0
        self.target_y = 0
        
        self.action_space = 374
        
        #768表示每个状态的初始通道数
        self.observation_space = 1024

        self.state = None
    
    #认为给定的action是一个按概率大小排列的top的预测类别索引
    #如果当前状态是最后一个状态，或真值在当前预测的top5里面，就终止
    def step(self, action,label,done):
        
        #看每一个状态是否是最后一个状态
        done = done.long()

        #把每一个状态是否预测对了，现在这里得到。
        #predict_list为1表示当前状态预测对了，为0表示当前状态预测错了
        #predict_index表示预测的最大动作在5个动作中的索引
        predict_list,predict_index=torch.max((torch.repeat_interleave(label.unsqueeze(1),repeats=5,dim=1)==action).long(),dim=1)

        reward= torch.empty((action.size())).cuda()

        for cur_state_index in range(action.size()[0]):

            cur_state_done = done[cur_state_index]


            #当前预测对了
            if predict_list[cur_state_index]==1:
                
                #当前是最后一个状态
                if cur_state_done==1:
                    cur_reward=torch.zeros((5)).cuda()
                    cur_reward[predict_index[cur_state_index]]=10
                else:
                    cur_reward=torch.zeros((5)).cuda()
                    cur_reward[predict_index[cur_state_index]]=1
            
            #当前预测错了
            elif predict_list[cur_state_index]==0:
                
                #当前是最后一个状态
                if cur_state_done==1:
                    cur_reward=torch.ones((5)).cuda()*(-10)
                else:
                    cur_reward=torch.ones((5)).cuda()*(-1)
            reward[cur_state_index]=cur_reward

        return reward, done, predict_list
    
    def reset(self,state):
        self.state=state
        return self.state
        
    def render(self, mode='human'):
        return None
        
    def close(self):
        return None
    

class ReplayBuffer():
    def __init__(self, capacity,n_hidden,top_k):
        self.capacity=capacity
        self.count=0
        self.current_index=0

        self.index_list=[i for i in range(capacity)]
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.states = torch.empty((capacity,n_hidden*8)).cuda()
        self.actions = torch.empty((capacity,top_k)).cuda()
        self.next_states = torch.empty((capacity,n_hidden*8)).cuda()
        self.rewards = torch.empty((capacity,top_k)).cuda()
        self.isdone = torch.empty((capacity)).cuda()
    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done):
        
        self.count+=state.size()[0]
        if self.count>self.capacity:
            self.count=self.capacity

        #分为两种添加方式，第一种为从当前位置往后放，第二种为第一种放满了，得从头重新来。
        #第一种可以放的数量
        first_add_num=self.capacity-(self.current_index)
        if first_add_num>state.size()[0]:
            first_add_num=state.size()[0]
        second_add_num=state.size()[0]-first_add_num

        if first_add_num!=0:
            self.states[self.current_index:self.current_index+first_add_num,:]=state[:first_add_num,:]
            self.actions[self.current_index:self.current_index+first_add_num,:]=action[:first_add_num,:]
            self.next_states[self.current_index:self.current_index+first_add_num,:]=next_state[:first_add_num,:]
            self.rewards[self.current_index:self.current_index+first_add_num,:]=reward[:first_add_num,:]
            self.isdone[self.current_index:self.current_index+first_add_num]=done[:first_add_num]
            self.current_index+=first_add_num
            if self.current_index==self.capacity:
                self.current_index=0
        
        if second_add_num!=0:
            self.states[:second_add_num,:]=state[first_add_num:,:]
            self.actions[:second_add_num,:]=action[first_add_num:,:]
            self.next_states[:second_add_num,:]=next_state[first_add_num:,:]
            self.rewards[:second_add_num,:]=reward[first_add_num:,:]
            self.isdone[:second_add_num]=done[first_add_num:]
            self.current_index=second_add_num

  
    # 随机采样batch_size行数据
    def sample(self, batch_size):
        
        sample_index=torch.tensor(random.sample(self.index_list,batch_size)).long().cuda()
        
        state=torch.index_select(self.states,dim=0,index=sample_index)
        action=torch.index_select(self.actions,dim=0,index=sample_index)
        next_state=torch.index_select(self.next_states,dim=0,index=sample_index)
        reward=torch.index_select(self.rewards,dim=0,index=sample_index)
        done=torch.index_select(self.isdone,dim=0,index=sample_index)
        
        return state,action,reward,next_state,done
    # 目前队列长度
    def size(self):
        return self.count
    
class Net(nn.Module):
    # 构造只有一个隐含层的网络
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        # [b,n_states]-->[b,n_hidden]
        self.fc1 = nn.Linear(n_states, n_hidden)
        # [b,n_hidden]-->[b,n_actions]
        self.fc2 = nn.Linear(n_hidden, n_actions)

        self.fc = nn.Sequential(
            nn.Linear(n_states,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,n_actions)
        )
    # 前传
    def forward(self, x):  # [b,n_states]
        
        # x = self.fc1(x)
        # x = self.fc2(x)
        x=self.fc(x)
        return x

# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #

class DQN:
    #（1）初始化
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        # 计数器，记录迭代次数
        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        # 实例化目标网络
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)

        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    #（2）动作选择
    def take_action(self, state):
        # 维度扩充，给行增加一个维度，并转换为张量shape=[1,4]
        state = torch.Tensor(state[np.newaxis, :]).cuda()
        
        # 前向传播获取该状态对应的动作的reward
        actions_value = self.q_net(state)
        # 获取reward最大值对应的动作索引
        # action = actions_value.argmax().item()  # int
        action = torch.topk(actions_value,k=5)[1]

        return action

    #（3）网络训练
    def update(self, transition_dict):  # 传入经验池中的batch个样本
        # 获取当前时刻的状态 array_shape=[b,4]
        states = transition_dict['states']
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = transition_dict['actions'].long()
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
        rewards = transition_dict['rewards']
        # 下一时刻的状态 array_shape=[b,4]
        next_states = transition_dict['next_states']
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = transition_dict['dones']

        # 输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
        # 根据actions索引在训练网络的输出的第1维度上获取对应索引的q值（state_value）
        
        q_values = self.q_net(states).gather(1, actions)  # [b,1]
        
        # 下一时刻的状态[b,4]-->目标网络输出下一时刻对应的动作q值[b,2]-->
        # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
        
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)


        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
       
        q_targets = rewards + torch.repeat_interleave((self.gamma * max_next_q_values.squeeze() * (1-dones)).unsqueeze(1),repeats=5,dim=1)

        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.optimizer.step()

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        
        self.count += 1
        
if __name__ == '__main__':
    env = SketchEnv()
    env.reset()
    env.step(env.action_space.sample())
    print(env.state)
    env.step(env.action_space.sample())
    print(env.state)