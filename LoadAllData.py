import numpy as np
max_tobe=2.5
min_tobe=0.4

N=20000
M=95

pre_throughput=np.loadtxt('training data/throughput.txt')
speed =np.loadtxt('training data/speed.txt')
distance =np.loadtxt('training data/distance.txt')
acce=np.loadtxt('training data/acce.txt')
#缩小
'''
transform function :
y=(b-a)*x/(m-n)+(a*m-b*n)/(m-n）
while:m->a,n->b
'''
throughput=np.zeros([M,50])
for i in range(len(throughput)):
    min_data=np.min(pre_throughput[i])
    max_data=np.max(pre_throughput[i])
    for j in range(len(throughput[0])):
        throughput[i][j] = (max_tobe-min_tobe)*pre_throughput[i][j]/(max_data -min_data) +(min_tobe*max_data-max_tobe*min_data)/(max_data - min_data)

train_throughput=np.zeros([N,50])
train_speed=np.zeros([N,50])
train_distance=np.zeros([N,50])
train_acce=np.zeros([N,50])

for  i in range(len(train_throughput)):
    train_throughput[i]=throughput[i%M]
    train_speed[i]=speed[i%M]
    train_distance[i]=distance[i%M]
    train_acce[i] =acce[i%M]

