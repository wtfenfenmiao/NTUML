import pandas as pd
import numpy as np

def signMy(core):
    return (core>0)*2-1

#空格分隔，name是dataframe的列名
name=["w1","w2","w3","w4","label"]
data=pd.read_table("hw1_18_train.txt",delim_whitespace=True,names=name)
datatest=pd.read_table("hw1_18_test.txt",delim_whitespace=True,names=name)
#增加w0这一列
data.insert(0,"w0",1)
datatest.insert(0,"w0",1)
#数据大小，这里是500
datasize=len(data)
datatestsize=len(datatest)
#print (data.loc[:,["w0","w1","w2","w3","w4"]])
errfinal=0
labelreal=np.array(data["label"]).reshape(datasize,1)
labelrealtest = np.array(datatest["label"]).reshape(datatestsize, 1)
for i in range(2000):
    update=50
    plaparameter = np.random.randn(5,1)
    pkparameter = plaparameter
    labeltemp=np.dot(data.loc[:,["w0","w1","w2","w3","w4"]],plaparameter)
    #print (labeltemp)
    plalabel=signMy(labeltemp)
    #print(plalabel)
    plaTrue = np.sum(plalabel == labelreal)
    #print(plaTrue)
    pkTrue = plaTrue
    while update>0:
        update -= 1
        sample = data[plalabel != labelreal].sample(1)
        #print(sample)
        #print (np.array(sample)[0][0:-1].reshape(5,1))
        #print (plaparameter)
        #print (np.array(sample)[0][-1])
        plaparameter=plaparameter+np.array(sample)[0][-1]*np.array(sample)[0][0:-1].reshape(5,1)
        #print (newparameter)
        plalabel=signMy(np.dot(data.loc[:,["w0","w1","w2","w3","w4"]],plaparameter))
        plaTrue=np.sum(plalabel==labelreal)
        #print ("plaTrue:")
        #print (plaTrue)
        #print ("pkTrue:")
        #print (pkTrue)
        if plaTrue>pkTrue:
            pkTrue=plaTrue
            pkparameter=plaparameter

            #print (update)
        #break
    #break
    labeltest = signMy(np.dot(datatest.loc[:, ["w0", "w1", "w2", "w3", "w4"]], plaparameter))
    falsetest=np.sum(labeltest!=labelrealtest)
    err=falsetest/datatestsize
    print (err)
    errfinal+=err
errfinal=errfinal/2000
print (errfinal)







