import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#写这种代码，别给自己找麻烦，推导的时候矩阵行列什么样，这里就什么样！可以先把数据生成出来，最后再凑格式，凑格式用reshape就行......
#还有这个matrix真的是用的不熟，还是用array吧......array转置求逆啥的也都有，matrix实在是不方便啊
def sign(x):
    return (x>0)*2-1
#先生成1000个横坐标在[-1,1]之间，纵坐标在[-1,1]之间的点，生成数据，x和y
#x:1000行3列 （N行d+1列）
#y:1000行1列  （N行1列）
x=np.zeros((1000,3))
x[:,0]=1
x[:,1]=2*np.random.random(size=1000)-1    #随机生成x1
x[:,2]=2*np.random.random(size=1000)-1    #随机生成x2
#注意！这个mat直接*默认是内积，multiply才是按位乘；array直接乘默认是按位乘，np.dot才是内积
y=sign(x[:,1]*x[:,1]+x[:,2]*x[:,2]-0.6)                   #随机生成y
#print (y)
noise=np.random.random(size=1000)     #噪音，加百分之十的噪音
#print (np.array(noise<0.1))
#print (noise<0.1)
y[noise<0.1]=-y[noise<0.1]               #这种选条件的，[]结果里的条件得是没有[]的，[1,2,3,4,]这种
#y=y.reshape((1000,1))
#print (y)
#print (x)


#下面三行把数据可视化了一下
plt.scatter(x[y==1,1],x[y==1,2],c='blue',alpha=1,marker='o')  #这个前面筛完了条件的要是一行数组才行，所以上面1行1000列的y可以，但是1000行1列的y放这里就不行
plt.scatter(x[y==-1,1],x[y==-1,2],c='red',alpha=1,marker='+')
plt.show()

#把y整成1000行1列
y=y.reshape((1000,1))

#初始化w，3行1列
w=np.zeros((3,1))

#线性回归，推明白了w直接算就行
w=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.transpose(x)),y)
print (w)
#题里说的Ein是in-sample error，就是这个train的错误率
result=sign(np.dot(x,w))      #得到的结果
#print (result)
print (np.sum(result!=y)/1000)     #大概是0.5

#print (np.sum((np.dot(x,w)-y)*(np.dot(x,w)-y))/1000)    #这个是平方误差的公式，这里要的不是这个