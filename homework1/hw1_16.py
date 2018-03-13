import pandas as pd
import numpy as np

def signMy(core):
    if (core <= 0):
        return -1;
    else:
        return 1;

#空格分隔，name是dataframe的列名
name=["w1","w2","w3","w4","label"]
data=pd.read_table("hw1_15_train.txt",delim_whitespace=True,names=name)
#增加w0这一列
data.insert(0,"w0",1)
#数据大小，这里是400
datasize=len(data)
#print (datasize)
update_sum=0
for i in range(2000):
    parameter = np.random.randn(5)
    #print (parameter)
    flag = True
    count = 0
    update = 0
    while (count == 0) or (not flag):
        # print (parameter)
        #print(count)
        flag = True
        for ix in data.index:
            # print ((data.loc[ix]).values[0:-1]) #data.loc[ix]是找到每一行
            # print (parameter)
            core = np.dot((data.loc[ix]).values[0:-1], parameter)
            # print (core)
            if (signMy(core) != (data.loc[ix])["label"]):
                update += 1
                flag = False
                parameter = parameter + (data.loc[ix])["label"] * (data.loc[ix]).values[0:-1]
        count += 1
    print (update)
    update_sum+=update
print (update_sum)
print (update_sum/2000)







