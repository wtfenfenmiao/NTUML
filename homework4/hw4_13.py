#coding:utf-8
import numpy as np
import pandas as pd

#这个题直接算就行，线性回归

def sign(X):
    return 2*(X>0)-1

#X：n行d+1列
#Y：n行1列

#训练集获取
name=["1","2","label"]
train_data=pd.read_table("hw4_train.dat",delim_whitespace=True,names=name)
#print (train_data)
train_X=train_data.loc[:,["1","2"]]
train_X=train_X.reindex(columns=["0","1","2"])
train_X.loc[:,["0"]]=1      #这两行可以用来加为1的那一列
#print (train_X)
train_Y=train_data.loc[:,["label"]]
#print (train_Y)

#测试集获取
test_data=pd.read_table("hw4_test.dat",delim_whitespace=True,names=name)
#print (test_data)
test_X=test_data.loc[:,["1","2"]]
test_X=test_X.reindex(columns=["0","1","2"])
test_X.loc[:,["0"]]=1
#print (test_X)
test_Y=test_data.loc[:,["label"]]
#print (test_Y)

Lambda=10
I=np.eye(3)     #注意！！！这个Lambda必须*这个I，公式推导的时候要小心，之前就理解错了
w=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_X),train_X)+Lambda*I),np.transpose(train_X)),train_Y)

print (w)


Ein=np.sum(sign(np.dot(train_X,w))!=train_Y)/train_Y.size
Eout=np.sum(sign(np.dot(test_X,w))!=test_Y)/test_Y.size
print ("Ein:")
print (Ein)
print ("Eout:")
print (Eout)
