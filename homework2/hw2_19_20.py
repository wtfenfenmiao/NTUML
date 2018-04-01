#coding:utf-8
import numpy as np
import pandas as pd
name=["1","2","3","4","5","6","7","8","9","label"]   #一共有9维
data=pd.read_table("hw2_train.dat",delim_whitespace=True,names=name)    #非常神奇了，给的数据集train比test少......
#print (data)

def sign(x):
    return (x>0)*2-1
def h(x,s,theta):
    return s*sign(x-theta)


#data_row_size=y_label.size    #一共有100行
#print (data_row_size)
data_row_size=100
finalEin=2
finalS=1
finalDim=0
finalTheta=0
for i in range(9):
    this=data.iloc[:,[i,9]]
    #print (this)
    this=this.sort_values(by=name[i],axis=0,ascending=True)
    #print (this)
    this_x=np.array(this.iloc[:,0])
    y_label=np.array(this.iloc[:,1])
    #print (this_x)
    #print (y_label)
    theta=[]
    theta.append(this_x[0])
    for j in range(data_row_size-1):
        theta.append((this_x[j]+this_x[j+1])/2)
    #print (theta)
    #print (len(theta))
    this_Ein=2
    this_s=0
    this_theta=0
    for the in theta:
        this_pos_y=h(this_x,1,the)
        #print (this_pos_y)
        #print (y_label)
        this_the_Ein=sum(this_pos_y!=y_label)/data_row_size
        this_the_s=1
        if(this_the_Ein>0.5):
            this_the_Ein=1-this_the_Ein
            this_the_s=-1
        if(this_Ein>this_the_Ein):
            this_Ein=this_the_Ein
            this_theta=the
            this_s=this_the_s
    #print (this_Ein)
    #print (this_s)
    #print (this_theta)
    if(this_Ein<finalEin):
        finalEin = this_Ein
        finalS = this_s
        finalDim = i
        finalTheta = this_theta


    #break
print (finalEin)                #0.25有选项
print (finalS)
print (finalDim)
print (finalTheta)

data_test=pd.read_table("hw2_test.dat",delim_whitespace=True,names=name)
#print (data_test)
data_test_x=data_test.iloc[:,finalDim]
#print (data_test_x)
data_test_y=data_test.iloc[:,9]
#print (data_test_y)
data_test_x=np.array(data_test_x)
#print (data_test_x)
data_test_size=data_test_x.size
#print (data_test_size)
data_com_y=h(data_test_x,finalS,finalTheta)
#print (data_com_y)
print (sum(data_com_y!=data_test_y)/data_test_size)    #0.355，有选项
