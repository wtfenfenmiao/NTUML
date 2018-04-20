#coding:utf-8
import numpy as np
import pandas as pd

#给很多个Lambda，划分交叉验证集，去选择Lambda
def sign(X):
    return (X>0)*2-1

name=["1","2","label"]
train_data=pd.read_table("hw4_train.dat",delim_whitespace=True,names=name)
train_data=train_data.reindex(columns=["0","1","2","label"])
train_data.loc[:,["0"]]=1

test_data=pd.read_table("hw4_test.dat",delim_whitespace=True,names=name)
test_data=test_data.reindex(columns=["0","1","2","label"])
test_data.loc[:,["0"]]=1

train_X=train_data.loc[:,["0","1","2"]]
train_Y=train_data.loc[:,["label"]]

test_X=test_data.loc[:,["0","1","2"]]
test_Y=test_data.loc[:,["label"]]
#fold1到fold5是交叉验证集的划分
#fold1
cv_1_X=train_data.loc[0:39,["0","1","2"]]
cv_1_Y=train_data.loc[0:39,["label"]]
train_1_X=train_data.loc[40:199,["0","1","2"]]
train_1_Y=train_data.loc[40:199,["label"]]
#print (cv_1_X)
#print (train_1_X)

#fold2
cv_2_X=train_data.loc[40:79,["0","1","2"]]
cv_2_Y=train_data.loc[40:79,["label"]]
train_2_X=train_data.loc[0:39,["0","1","2"]].append(train_data.loc[80:199,["0","1","2"]])
train_2_Y=train_data.loc[0:39,["label"]].append(train_data.loc[80:199,["label"]])
#print (cv_2_X)
#print (train_2_X)

#fold3
cv_3_X=train_data.loc[80:119,["0","1","2"]]
cv_3_Y=train_data.loc[80:119,["label"]]
train_3_X=train_data.loc[0:79,["0","1","2"]].append(train_data.loc[120:199,["0","1","2"]])
train_3_Y=train_data.loc[0:79,["label"]].append(train_data.loc[120:199,["label"]])

#fold4
cv_4_X=train_data.loc[120:159,["0","1","2"]]
cv_4_Y=train_data.loc[120:159,["label"]]
train_4_X=train_data.loc[0:119,["0","1","2"]].append(train_data.loc[160:199,["0","1","2"]])
train_4_Y=train_data.loc[0:119,["label"]].append(train_data.loc[160:199,["label"]])

#fold5
cv_5_X=train_data.loc[160:199,["0","1","2"]]
cv_5_Y=train_data.loc[160:199,["label"]]
train_5_X=train_data.loc[0:159,["0","1","2"]]
train_5_Y=train_data.loc[0:159,["label"]]

mi=2
I=np.eye(3)
Lambda=0
out19=[0,1]

while (mi>=-10):
    if(mi>=0):
        Lambda=np.power(10,mi)
    else:
        Lambda=np.power(0.1,-mi)
    w1 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_1_X), train_1_X) + Lambda * I), np.transpose(train_1_X)),
               train_1_Y)
    w2 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_2_X), train_2_X) + Lambda * I), np.transpose(train_2_X)),
                train_2_Y)
    w3 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_3_X), train_3_X) + Lambda * I), np.transpose(train_3_X)),
                train_3_Y)
    w4 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_4_X), train_4_X) + Lambda * I), np.transpose(train_4_X)),
                train_4_Y)
    w5 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_5_X), train_5_X) + Lambda * I), np.transpose(train_5_X)),
                train_5_Y)
    Ecv1 = np.sum(sign(np.dot(cv_1_X, w1)) != cv_1_Y) / cv_1_Y.size
    Ecv2 = np.sum(sign(np.dot(cv_2_X, w2)) != cv_2_Y) / cv_2_Y.size
    Ecv3 = np.sum(sign(np.dot(cv_3_X, w3)) != cv_3_Y) / cv_3_Y.size
    Ecv4 = np.sum(sign(np.dot(cv_4_X, w4)) != cv_4_Y) / cv_4_Y.size
    Ecv5 = np.sum(sign(np.dot(cv_5_X, w5)) != cv_5_Y) / cv_5_Y.size

    Ecv=(Ecv1+Ecv2+Ecv3+Ecv4+Ecv5)/5
    Ecv=Ecv["label"]

    if(Ecv<out19[1]):
        out19[0]=mi
        out19[1]=Ecv
    mi=mi-1

print ("19result:")
print (out19)

#用交叉验证找到合适的Lambda之后，再把所有的train放一起训模型
if(out19[0]>=0):
    Lambda=np.power(10,out19[0])
else:
    Lambda=np.power(0.1,-out19[0])

w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_X), train_X) + Lambda * I), np.transpose(train_X)),
               train_Y)

Ein = np.sum(sign(np.dot(train_X, w)) != train_Y) / train_Y.size
Ein=Ein["label"]
Eout = np.sum(sign(np.dot(test_X, w)) != test_Y) / test_Y.size
Eout=Eout["label"]
print ("Ein:{}".format(Ein))
print ("Eout:{}".format(Eout))
