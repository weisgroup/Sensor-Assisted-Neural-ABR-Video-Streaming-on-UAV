from helperwithspeed import *
from pylab import *
total_times=41

class BufferBased :
    def __init__(self,train_throughput,total_times=total_times):
        self.total_times=total_times
        self.train_throughput=train_throughput
        self.last_action=0 #假设为上一时刻的action
        self.this_action=0 #此刻的action
        self.buffer=0 #初始buffer为0
        self.all_action=[] #记录所有action的值
        self.all_buffer=[] #记录所有的buffer值
        self.all_reward=[] #记录所有的reward

    def choose_action(self):#base only on buffer
        action  =0
        if(self.buffer<1):
            action = 0
        elif(self.buffer<=5):
            action =1
        elif(self.buffer<12):
            action =2
        else:
            action =3
        return action

    def play(self): #播放一个完整的视频

        for j in range(self.total_times):
            self.this_action =self.choose_action()#这个时刻选择的action
            #print(self.this_action)
            self.all_action.append( self.this_action)
            throughput=self.train_throughput[j+9]
            self.buffer,rebuffering =updateBuffer(self.buffer,self.this_action,throughput)
            self.all_buffer.append(self.buffer)
            reward =Reward(self.this_action,self.last_action,rebuffering)
            self.all_reward.append(reward)
            self.last_action=self.this_action

    def print_data(self):
        #print(self.all_reward)
        print(sum(self.all_reward))
        #print(self.all_buffer)
        return sum(self.all_reward)

    def display_data(self):
        figure(1)
        title("action")
        plot(self.all_action)
        ylim(-1,4)
        figure(2)
        title("buffer")
        plot(self.all_buffer)
        ylim(-1,25)
        figure(3)
        title("throughput")
        plot(self.train_throughput)
        show()

def test(all=None):
   if all==None:
        all_best=[]
        for i in range(95):

            this_throughput, _, _, _ = getThroughputData(i)
            #print(len(this_throughput))
            video = BufferBased(this_throughput)
            video.play()
            all_best.append(video.print_data())


        print("the average is ",sum(all_best)/len(all_best))
        figure(1)
        title("all the sum reward of all throughput")
        plot(all_best)
        show()
   else:
        # test a data
        j = all
        this_throughput, _, _, _ = getThroughputData(j)
        # print(len(this_throughput))
        video = BufferBased(this_throughput)
        video.play()
        video.print_data()
        # all_best.append(video.print_data())
        video.display_data()
if __name__=="__main__":
    test()
