import pandas as pd
import numpy as np
import hw7_13_15
import matplotlib.pyplot as plt
from hw7_13_15 import sign
from hw7_13_15 import prune_CART_addNode
from hw7_13_15 import predict

if __name__ =="__main__":
    train_data = pd.read_table("hw7_train.dat", header=None, delim_whitespace=True)
    test_data = pd.read_table("hw7_test.dat", header=None, delim_whitespace=True)

    rootlist=[]
    Ein_gt=[]
    Ein_Gt=[]
    Eout_Gt=[]
    Ein_Gt_predict=pd.DataFrame(index=range(len(train_data)),columns=['predict'])
    Ein_Gt_predict["predict"]=0
    Eout_Gt_predict=pd.DataFrame(index=range(len(test_data)),columns=['predict'])
    Eout_Gt_predict["predict"]=0
    for T in range(30000):    #30000个树
        root=prune_CART_addNode(train_data.sample(len(train_data),replace=True).reset_index(drop=True))
        rootlist.append(root)

        train_data["predict"]=0
        test_data["predict"]=0
        predict(train_data,root,"predict")
        predict(test_data, root, "predict")

        this_gt = sum(train_data["predict"] != train_data[2]) / len(train_data)

        Ein_Gt_predict["predict"]+=train_data["predict"]
        Eout_Gt_predict["predict"]+=test_data["predict"]
        this_Ein_Gt=sum(sign(Ein_Gt_predict["predict"])!=train_data[2])/len(train_data)
        this_Eout_Gt=sum(sign(Eout_Gt_predict["predict"])!=test_data[2])/len(test_data)

        Ein_gt.append(this_gt)
        Ein_Gt.append(this_Ein_Gt)
        Eout_Gt.append(this_Eout_Gt)

        train_data.drop(["predict"],axis=1,inplace=True)
        test_data.drop(["predict"],axis=1,inplace=True)

        print ("")
        print ("iteration:",T)
        print ("gt:",this_gt)
        print ("Ein_Gt:",this_Ein_Gt)
        print ("Eout_Gt:",this_Eout_Gt)

    print("final:")


    plt.title("1920_Ein_gt")    #这个没啥规律，但是比16_18的效果差多了，也对，因为树分支只有一个，很粗糙
    plt.hist(Ein_gt)
    plt.show()

    plt.title("1920_Ein_Gt")    #越来越小，也是没有16_18效果好
    plt.plot(Ein_Gt)
    plt.show()

    plt.plot("1920_Eout_Gt")    #越来越小，没有16_18效果好，但是比一棵树效果也是好了。臭皮匠和诸葛亮，恩，糟心的有策略的合起来也能不糟心......
    plt.plot(Eout_Gt)
    plt.show()


