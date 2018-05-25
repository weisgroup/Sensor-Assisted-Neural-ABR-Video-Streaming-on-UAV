import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from helperwithspeed import *
gama=0.9
total_times =42


class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        self.rnn = nn.LSTM(
            input_size=2,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )

        self.fc1 = nn.Linear(69, 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 4)
        self.relu = nn.ReLU()

    def forward(self, x):  # x shape(1x1x13)
        x1 = np.zeros([8], dtype=np.float32)  # 代表throughput
        for i in range(8):
            x1[i] = x[i]

        x2 = np.array([x[8]], dtype=np.float32)  # 代表buffersize
        x3 = np.array([x[9]], dtype=np.float32)  # 代表lastAction
        x4 = np.array([x[10]], dtype=np.float32)  # 代表speed
        x5 = np.array([x[11]], dtype=np.float32)  # 代表distance
        x6 = np.array([x[12]], dtype=np.float32)  # 代表acce

        x1 = Variable(torch.from_numpy(x1))
        x2 = Variable(torch.from_numpy(x2))
        x3 = Variable(torch.from_numpy(x3))
        x4 = Variable(torch.from_numpy(x4))
        x5 = Variable(torch.from_numpy(x5))
        x6 = Variable(torch.from_numpy(x6))

        x1 = x1.view(-1, 4, 2)
        x2 = x2.view(1, -1)
        x3 = x3.view(1, -1)
        x4 = x4.view(1, -1)
        x5 = x5.view(1, -1)
        x6 = x6.view(1, -1)

        r_out, (h_n, h_c) = self.rnn(x1, None)

        #r_out = r_out.view(-1, num_flat_features(r_out))
        datain = torch.cat((r_out[:,-1,:], x2), 1)
        datain = torch.cat((datain, x3), 1)
        datain = torch.cat((datain, x4), 1)
        datain = torch.cat((datain, x5), 1)
        datain = torch.cat((datain, x6), 1)

        out = self.relu(self.fc1(datain))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return F.log_softmax(out)

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.rnn = nn.LSTM(
            input_size=2,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )

        self.fc1 = nn.Linear(69, 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):  # x shape(1x1x13)
        x1 = np.zeros([8], dtype=np.float32)  # 代表throughput
        for i in range(8):
            x1[i] = x[i]

        x2 = np.array([x[8]], dtype=np.float32)  # 代表buffersize
        x3 = np.array([x[9]], dtype=np.float32)  # 代表lastAction
        x4 = np.array([x[10]], dtype=np.float32)  # 代表speed
        x5 = np.array([x[11]], dtype=np.float32)  # 代表distance
        x6 = np.array([x[12]], dtype=np.float32)  # 代表acce

        x1 = Variable(torch.from_numpy(x1))
        x2 = Variable(torch.from_numpy(x2))
        x3 = Variable(torch.from_numpy(x3))
        x4 = Variable(torch.from_numpy(x4))
        x5 = Variable(torch.from_numpy(x5))
        x6 = Variable(torch.from_numpy(x6))

        x1 = x1.view(-1, 4, 2)
        x2 = x2.view(1, -1)
        x3 = x3.view(1, -1)
        x4 = x4.view(1, -1)
        x5 = x5.view(1, -1)
        x6 = x6.view(1, -1)

        r_out, (h_n, h_c) = self.rnn(x1, None)

        #r_out = r_out.view(-1, num_flat_features(r_out))
        datain = torch.cat((r_out[:,-1,:], x2), 1)
        datain = torch.cat((datain, x3), 1)
        datain = torch.cat((datain, x4), 1)
        datain = torch.cat((datain, x5), 1)
        datain = torch.cat((datain, x6), 1)

        out = self.relu(self.fc1(datain))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def roll_out(actor_network,value_network,TestThroughput,TestSpeed,TestDistance,TestAcce):
    #initial
    CurrentBufferSize =0
    LastBitRate = 0
    train_time =1
    initial_state=Input(TestThroughput,TestSpeed,TestDistance,TestAcce, CurrentBufferSize, LastBitRate, train_time)
    state=initial_state

    #return data
    states=[]
    actions =[]
    rewards =[]
    buffers=[]
    rebuffer_all=[]
    action_all=[]
    buffers.append(CurrentBufferSize)
    #is_done =False
    final_r =0

    for j in range(total_times-1):
        states.append(state)
        log_softmax_action =actor_network(state)
        softmax_action =torch.exp(log_softmax_action)
        action=np.random.choice(4,p=softmax_action.cpu().data.numpy()[0])
        #action=makeChoice(softmax_action.cpu().data.numpy()[0])
        print(action)
        action_all.append(action)
        one_hot_action=[int (k==action) for k in range(4)]
        throughput=TestThroughput[train_time+8]
        CurrentBufferSize,rebuffer =updateBuffer(CurrentBufferSize,action,throughput)
        rebuffer_all.append(rebuffer)
        buffers.append(CurrentBufferSize)

        reward=Reward(action,LastBitRate,rebuffer)
        LastBitRate = action
        train_time =train_time+1
        next_state =Input(TestThroughput,TestSpeed,TestDistance,TestAcce, CurrentBufferSize, LastBitRate, train_time)
        final_state=next_state
        state=next_state

        actions.append(one_hot_action)
        rewards.append(reward)

        if (j == total_times - 1):
            last_softmax_action = actor_network(final_state)
            last_action = torch.exp(last_softmax_action)
            last_choose_action = np.random.choice(4, p=last_action.cpu().data.numpy()[0])
            last_throughput = TestThroughput[train_time + 8]
            last_buffer, last_rebuffer = updateBuffer(CurrentBufferSize, last_choose_action, last_throughput)
            final_r = Reward(last_action, LastBitRate, last_rebuffer)


    return states,actions,rewards,buffers,final_r,action_all,rebuffer_all


def discount_reward(r,gama,final_r):
    discounted_r =np.zeros_like(r)
    running_add =final_r
    for t in reversed(range(0,len(r))):
        running_add=running_add*gama+r[t]
        discounted_r[t]=running_add
    return discounted_r

def main():
    value_network =ValueNetwork()
    actor_network = ActorNetwork()

    total_valueloss=[]
    total_actorloss=[]
    total_reward=[]

    maxReward = -100
    maxepoch = 0
    '''
    # best learning rate
    decayed_learning_rate_value =0.01
    decayed_learning_rate_actor = 0.00002
    '''
    decayed_learning_rate_value = 0.01
    decayed_learning_rate_actor = 0.00002

    test_data, test_speed,test_distance,test_acce=getThroughputData(0)
    test_reward=[]
    test_buffer1=[]
    test_buffer2=[]
    choose_action =[]
    train_before=[]
    train_after=[]

    print("start training")
    for step in range(N):#需要修改
        if(step==0):
            for i in range(M):
                test1, test2, test3, test4 = getThroughputData(i)
                _, _, rewards, this_buffers, final_r, action_all,_ = roll_out(actor_network, value_network, test1,
                                                                            test2, test3, test4)
                train_before.append(sum(rewards) + final_r)

        if (step == N-2):
            for i in range(M):
                test1, test2, test3, test4 = getThroughputData(i)
                _, _, rewards, this_buffers, final_r, action_all,_ = roll_out(actor_network, value_network, test1,
                                                                            test2, test3, test4)
                train_after.append(sum(rewards) + final_r)

        print("epoch",step)
        #if (step)%400==0:
            #decayed_learning_rate_value =decayed_learning_rate_value* 0.92

        #if (step)%100==0:
            #decayed_learning_rate_actor =decayed_learning_rate_actor * 0.99

        train_throughput,train_speed,train_distance,train_acce= getThroughputData(step)

        value_network_optim = torch.optim.Adam(value_network.parameters(), lr=decayed_learning_rate_value)
        actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=decayed_learning_rate_actor)

        states, actions, rewards, buffers, final_r, _,_ = roll_out(actor_network, value_network, train_throughput,train_speed,train_distance,train_acce)
        data=sum(rewards)
        total_reward.append(data)

        new_states=np.zeros([len(states),len(states[0])],dtype=np.float32)

        for i in range(len(states)):
            for j in range(len(states[i])):
                new_states[i][j]=states[i][j]

        actions_var=Variable(torch.Tensor(actions).view(-1,4))
        states_var= Variable(torch.from_numpy(new_states))
        #train actor_network
        actor_network_optim.zero_grad()
        log_softmax_actions=actor_network(states[0])
        for i in range(1,len(states)):
            log_softmax_actions=torch.cat((log_softmax_actions,actor_network(states[i])),0)

        vs=value_network(states[0])
        for i in range(1,len(states)):
            vs=torch.cat((vs,value_network(states[i])),0)
        vs=vs.detach()
        qs=Variable(torch.Tensor(discount_reward(rewards,0.99,final_r)))
        advantages= qs-vs
        actor_network_loss=-torch.mean(torch.sum(log_softmax_actions*actions_var,1)*advantages)
        total_actorloss.append(actor_network_loss.cpu().data.numpy())
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
        actor_network_optim.step()

        #train value_network
        value_network_optim.zero_grad()
        target_value =qs
        values=value_network(states[0])
        for i in range(1,len(states)):
            values=torch.cat((values,value_network(states[i])),0)

        target_value = target_value.view(-1,1)
        criterion =nn.MSELoss()
        value_network_loss=criterion(values,target_value)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(),0.5)
        value_network_optim.step()

        total_valueloss.append(value_network_loss.cpu().data.numpy())

        _, _, rewards, this_buffers, final_r, action_all,test_rebuffer = roll_out(actor_network, value_network, test_data,test_speed,test_distance,test_acce)
        test_reward.append(sum(rewards) + final_r)

        print(sum(rewards) + final_r)
        #print(value_network_loss.cpu().data.numpy()[0])
        #print(actor_network_loss.cpu().data.numpy()[0])

        if (sum(rewards) > maxReward):
            maxReward = sum(rewards)
            maxepoch = step
            test_buffer2 = this_buffers
            choose_action = action_all
            print(maxReward)
            print(maxepoch)
            print("action")
            print(choose_action)
            print("every rebuffer")
            print(test_rebuffer)
            print("every reward")
            print(rewards)
            print("the buffer")
            print(this_buffers)
        if (step == 0):
            test_buffer1=this_buffers
            print("action ")
            print(action_all)
            print("every rebuffer")
            print(test_rebuffer)
            print("every reward")
            print(rewards)
            print(" the buffer")
            print(this_buffers)

    #torch.save(value_network, 'value_lstm.pkl')
    #torch.save(actor_network, 'actor_lstm.pkl')

    print(maxReward)
    print(maxepoch)

    print("所有数据对比")
    print("train_before",train_before)
    print("train_after", train_after)
    #画图
    figure(1)
    title("lstm_with_speed_total_valuesloss")
    plot(total_valueloss)
    figure(2)
    title("total_actorloss")
    plot(total_actorloss)
    figure(3)
    title("smooth_valueloss")
    plot(smooth(total_valueloss,stride=20))
    figure(4)
    title("smooth_actorloss")
    plot(smooth(total_actorloss,stride=20))
    figure(5)
    title("reward")
    plot(test_reward)
    figure(6)
    plot(smooth(test_reward,stride=20))
    figure(7)
    plot(test_buffer1)
    figure(8)
    plot(test_buffer2)
    figure(9)
    plot(test_data)
    figure(10)
    plot(choose_action)
    figure(11)
    plot(train_before)
    plot(train_after)

    show()

if __name__ =='__main__':
    main()


