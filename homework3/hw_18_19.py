#coding:utf-8
#逻辑回归
#这里的行列数，1000和21这种，感觉弄成常量可扩展性高一些。不过这里小作业觉得没关系，以后竞赛可能得这么弄
import numpy as np
import pandas as pd
def sign(x):
    return (x>0)*2-1

def sigmoid(X):
    return 1/(1+np.exp(-X))

#梯度下降
#W是要学的参数，N是训练集一共有多少个数据
def gradecent(X,Y,W,N):
    # there0=-Y*np.dot(X,W)
    # there1=Y*sigmoid(there0)
    # there2=np.dot(np.transpose(X),there1)
    # there3=-there2/N
    #return there3
    return -np.dot(np.transpose(X),Y*sigmoid(-Y*np.dot(X,W)))/N



#训练集
#X：1000行21列
#Y：1000行1列
train_data=pd.read_table("hw3_train.dat",delim_whitespace=True,names=range(21))
train_X=np.zeros((1000,21))
train_X[:,0]=1
for i in range(20):
    train_X[:,i+1]=train_data.loc[:,i]
train_Y=train_data.loc[:,20].reshape((1000,1))
#print (train_data)
#print (train_X)
#print (train_Y)

W=np.zeros((21,1))
yita=0.01
for i in range(2000):
    W=W-yita*gradecent(train_X,train_Y,W,1000)
    #print(W)
    #train_result = sigmoid(np.dot(train_X, W))
    #print(np.sum(train_Y != train_result) / 1000)


train_result=sign(sigmoid(np.dot(train_X,W))*2-1)
print ("train:")
print (np.sum(train_Y!=train_result)/1000)

test_data=pd.read_table("hw3_test.dat",delim_whitespace=True,names=range(21))
#print (test_data)
test_X=np.zeros((3000,21))
test_X[:,0]=1
for i in range(20):
    test_X[:,i+1]=test_data.loc[:,i]
test_Y=test_data.loc[:,20].reshape((3000,1))
test_result=sign(sigmoid(np.dot(test_X,W))*2-1)
print ("test:")
print (np.sum(test_Y!=test_result)/3000)
#梯度下降中：yita值：
#0.01是0.220
#0.001是0.475



