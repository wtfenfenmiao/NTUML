#coding:utf-8
#f的部分
#[-1,1]之间生成20个x
#y用sign(x)+20%的噪声来弄，噪声就是有20%的y要把正的变成负的，负的变成正的。
#这个噪声，可以用生成随机数的办法弄，就是生成一个0到1之间的随机数，然后如果这个数小于0.2，就要把结果反转

#h的部分
#h=s*sign(x-theta),s等于1或者-1（双向的，同样一个分界线，可能左边都是正的，右边都是负的，也可能左边都是负的，右边都是正的）
#theta可以取，两个相邻x的中值，因为两个相邻的x中间的数效果都是一样的
#然后Ein，就用这个h硬算就行，就是每一种theta，都用这个h算出来预测的结果，然后对预测错的比例
#Eout的话，看那个推导的公式，就是第16题的选项

import numpy as np
import pandas as pd

def sign(x):
    return (x>0)*2-1     #分成+1和-1
def h(x,s,theta):
    return s*sign(x-theta)

def eachRound():
    f_x = np.random.rand(1, 20) * 2 - 1  # 可以把这个20改成5看一下情况，看看输出的对不对
    f_x.sort()
    #print(f_x)
    f_noise = np.random.rand(1, 20)
    #print(f_noise)
    f_y_before_noise = sign(f_x)
    #print(f_y_before_noise)
    f_y_after_noise = f_y_before_noise
    f_y_after_noise[f_noise < 0.2] *= -1
    #print(f_y_after_noise)

    theta = []
    theta.append(-1)  # 最左边的一条“竖线”。最右边就不用了，效果一样的
    for i in range(19):  # i从0到18,f_x的下标是从0到19
        theta.append((f_x[0][i] + f_x[0][i + 1]) / 2)
    #print(theta)

    Ein = 2  # 随便设一个比1大的就行
    Ein_theta = 0
    Ein_s = 0
    for the in theta:
        h_y_positive = h(f_x, 1, the)
        #print (h_y_positive)
        Ein_this_pos = sum(h_y_positive[0] != f_y_after_noise[0]) / 20
        #print (Ein_this_pos)
        Ein_this = Ein_this_pos
        s_this = 1
        if (Ein_this_pos > 0.5):
            Ein_this = 1 - Ein_this_pos
            s_this = -1
        if (Ein_this < Ein):
            Ein = Ein_this
            Ein_s = s_this
            Ein_theta = the
    Eout=0.5+0.3*Ein_s*(abs(Ein_theta)-1)
    return Ein,Eout,Ein_theta,Ein_s

if __name__=="__main__":
    Ein_sum=0
    Eout_sum=0
    for i in range(5000):
        result=eachRound()
        #print (result)
        #print (result[0])
        Ein_sum+=result[0]
        Eout_sum+=result[1]
        #break
    Ein_sum/=5000
    Eout_sum/=5000
    print (Ein_sum)     #0.17左右，有选项
    print (Eout_sum)    #0.25左右，有选项











