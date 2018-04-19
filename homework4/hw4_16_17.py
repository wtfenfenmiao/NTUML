#coding:utf-8
import numpy as np
import pandas as pd

#这个题直接算就行，线性回归

def sign(X):
    return 2*(X>0)-1

#X：n行d+1列
#Y：n行1列


name=["1","2","label"]
train_data=pd.read_table("hw4_train.dat",delim_whitespace=True,names=name)
# #validation_data=validation_data.reset_index(drop=True)   #一行没啥用的代码，可以重建索引，去掉原来的，让索引从0开始
# #这个划分验证集和训练集的操作太棒了
# validation_data=train_data.sample(n=80,axis=0)    #随机取样，取出来验证集
# #print (validation_data)
# train_data=train_data.append(validation_data)    #把验证集加到训练集里
# #print (train_data)
# train_data=train_data.drop_duplicates(keep=False)   #去重，训练集就出来了
# #print (train_data)

#这里为了做题就直接用前120后80了。讲道理应该用上面的。
validation_data=train_data.iloc[120:200,:]
train_data=train_data.iloc[0:120,:]
print (validation_data)
print (train_data)

#训练集获取
train_X=train_data.loc[:,["1","2"]]
train_X=train_X.reindex(columns=["0","1","2"])
train_X.loc[:,["0"]]=1      #这两行可以用来加为1的那一列
#print (train_X)
train_Y=train_data.loc[:,["label"]]
#print (train_Y)

#验证集获取
validation_X=validation_data.loc[:,["1","2"]]
validation_X=validation_X.reindex(columns=["0","1","2"])
validation_X.loc[:,["0"]]=1      #这两行可以用来加为1的那一列
#print (validation_X)
validation_Y=validation_data.loc[:,["label"]]
#print (validation_Y)

#测试集获取
test_data=pd.read_table("hw4_test.dat",delim_whitespace=True,names=name)
#print (test_data)
test_X=test_data.loc[:,["1","2"]]
test_X=test_X.reindex(columns=["0","1","2"])
test_X.loc[:,["0"]]=1
#print (test_X)
test_Y=test_data.loc[:,["label"]]
#print (test_Y)


I=np.eye(3)     #注意！！！这个Lambda必须*这个I，公式推导的时候要小心，之前就理解错了
mi=2
Etrain16=[0,1,1,1]
Eval17=[0,1,1,1]
while(mi>=-10):
    if(mi>=0):
        Lambda=np.power(10,mi)
    else:
        Lambda=np.power(0.1,-mi)
    w=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_X),train_X)+Lambda*I),np.transpose(train_X)),train_Y)
    Etrain = np.sum(sign(np.dot(train_X, w)) != train_Y) / train_Y.size
    Eval = np.sum(sign(np.dot(validation_X,w)) != validation_Y) / validation_Y.size
    Eout = np.sum(sign(np.dot(test_X, w)) != test_Y) / test_Y.size
    Etrain=Etrain["label"]
    Eval=Eval["label"]
    Eout=Eout["label"]
    if(Etrain<Etrain16[1]):
        Etrain16[0]=mi
        Etrain16[1]=Etrain
        Etrain16[2]=Eval
        Etrain16[3]=Eout
    if(Eval<Eval17[2]):
        Eval17[0]=mi
        Eval17[1]=Etrain
        Eval17[2]=Eval
        Eval17[3]=Eout
    print ("log10Lambda:{}".format(mi))
    print("Etrain:{}".format(Etrain))
    print("Eval:{}".format(Eval))
    print("Eout:{}".format(Eout))
    print ("")
    mi = mi - 1
print (Etrain16)
print (Eval17)




