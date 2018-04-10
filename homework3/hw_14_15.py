import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sign(x):
    return (x>0)*2-1

#生成X和Y
#X:1000行6列，（1,x1,x2,x1*x2,x1*x1,x2*x2）
#Y:1000行1列
#W:6行一列

X=np.zeros((1000,6))
X[:,0]=1
X[:,1]=np.random.random(size=1000)*2-1
X[:,2]=np.random.random(size=1000)*2-1
X[:,3]=X[:,1]*X[:,2]
X[:,4]=X[:,1]*X[:,1]
X[:,5]=X[:,2]*X[:,2]
noise=np.random.random(size=1000)
Y=sign(X[:,4]+X[:,5]-0.6)
Y[noise<0.1]=-Y[noise<0.1]


# 可视化一下数据，证明数据生成对了
# plt.scatter(X[Y==1,1],X[Y==1,2],c='blue',alpha=1,marker='o')
# plt.scatter(X[Y==-1,1],X[Y==-1,2],c='red',alpha=1,marker='x')
# plt.show()
Y.reshape((1000,1))

W=np.zeros((6,1))
W=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),Y)
print (W)

# result=sign(np.dot(X,W))
# print (np.sum(result==Y)/1000)      #14选第二个W

# #15要做Eout，所以要再生成新数据，然后1000次平均算Eout。这里的Eout是 out-of-sample error
re=0.0
for i in range(1000):
    X=np.zeros((1000,6))
    X[:,0]=1
    X[:,1]=np.random.random(size=1000)*2-1
    X[:,2]=np.random.random(size=1000)*2-1
    X[:,3]=X[:,1]*X[:,2]
    X[:,4]=X[:,1]*X[:,1]
    X[:,5]=X[:,2]*X[:,2]
    Y=sign(X[:,4]+X[:,5]-0.6)
    noise=np.random.random(size=1000)
    Y[noise<0.1]=-Y[noise<0.1]
    result=sign(np.dot(X,W))
    thisre=np.sum(result!=Y)/1000
    print (thisre)
    re=re+thisre
re=re/1000
print ("结果：")
print (re)     #答案是0.1左右