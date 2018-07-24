#coding utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sign(x):
    return (x>0)*2-1

def get_theta(train):
    theta=[]
    for feature in range(len(train.columns)-2):
        train_sort_by_feature=train.sort_values(by=[feature])
        train_feature_col=train_sort_by_feature[feature]
        train_feature_col=train_feature_col.reset_index(drop=True)
        theta.append([(train_feature_col[j]+train_feature_col[j+1])/2 for j in range(len(train_feature_col)-1)])
    return theta

def desicion_stump(train,theta):   #这里train不用return，这个train是全局更改的
    #print (train)
    positive=sum(train["u"][train[2]==1])
    #print (positive)
    s_final=1
    Err_final=1-positive
    if(1-positive>positive):    #theta用左边的和所有两两中间的，最右边的那个不要了，左右留一个就行
        s_final=-1
        Err_final=positive
    feature_final=train.columns[0]
    #print (feature_final)
    theta_final=train.sort_values(by=[train.columns[0]])[train.columns[0]].reset_index(drop=True)[0]-1
    for feature in range(len(train.columns)-2):
        for the in theta[feature]:
            result = sign(train[feature] - the)
            Err_1=np.sum(train["u"][result!=train[2]])
            Err_neg_1=np.sum(train["u"][result == train[2]])    #注，按照ppt上的公式，严格对

            #print (Err)
            if(Err_1<Err_final):
                s_final=1
                theta_final=the
                feature_final=feature
                Err_final=Err_1
            if(Err_neg_1<Err_final):
                s_final=-1
                theta_final=the
                feature_final=feature
                Err_final=Err_neg_1
    result_final = s_final*sign(train[feature_final] - theta_final)
    epsilon_final = np.sum(train["u"][result_final != train[2]]) / np.sum(train["u"])  # epsilon
    Ein_final = np.sum(result_final != train[2]) / len(result_final)
    diamond=np.sqrt((1-epsilon_final)/epsilon_final)
    #print(diamond)
    alpha=np.log(diamond)
    # print(alpha)
    # print (s_final)
    # print (theta_final)
    # print (Err_final)
    # print (feature_final)
    # print (epsilon_final)
    # print (Ein_final)
    #print (train[2])
    #print (train["u"])

    train_pos_u=train["u"][s_final*sign(train[feature_final]-theta_final)==train[2]]/diamond
    #print (train_pos_u)
    train_neg_u=train["u"][s_final*sign(train[feature_final]-theta_final)!=train[2]]*diamond
    #print (train_neg_u)
    train["u"]=train_pos_u.append(train_neg_u)
    #print (train["u"])
    #print (train)
    #print("")
    return s_final,theta_final,feature_final,Err_final,epsilon_final,Ein_final,alpha


if __name__=="__main__":
    train = pd.read_table("hw6_adaboost_train.dat", delim_whitespace=True, header=None)
    test = pd.read_table("hw6_adaboost_test.dat", delim_whitespace=True, header=None)
    train["u"] = 1 / (len(train))
    Theta=get_theta(train)
    s_print=[]
    theta_print=[]
    feature_print=[]
    Err_print=[]
    alpha_print=[]
    epsilon_print=[]
    Ein_print=[]
    Eoutgt=[]
    Gt=0
    Gtout=0
    EGt=[]
    EGtout=[]
    U=[]
    U.append(np.sum(train["u"]))
    for itnum in range(320):
        print (str(itnum)+":")
        s,theta,feature,Err,epsilon,Ein,alpha=desicion_stump(train,Theta)
        Gt+=alpha*s*sign(train[feature]-theta)
        Eoutgt.append(np.sum(s*sign(test[feature]-theta)!=test[2])/len(test))
        Gtout+=alpha*s*sign(test[feature]-theta)
        #print (sum(sign(Gt)==train[2])/len(train))
        EGt.append(sum(sign(Gt)!=train[2])/len(train))
        EGtout.append(sum(sign(Gtout)!=test[2])/len(test))
        #print (Gt)
        U.append(np.sum(train["u"]))
        s_print.append(s)
        theta_print.append(theta)
        feature_print.append(feature)
        Err_print.append(Err)
        alpha_print.append(alpha)
        epsilon_print.append(epsilon)
        Ein_print.append(Ein)
    print (s_print)
    print (theta_print)
    print (feature_print)
    print (Err_print)
    print (alpha_print)
    print (epsilon_print)
    print (EGt)
    print (U)

    print (Err_print[0])   #0.24
    print (alpha_print[0])  #0.57
    print (EGt[299])  #0
    print (U[1])   #0.85
    print (U[299])  #0.0054
    print(Eoutgt[0])  #0.29
    print(EGtout[299]) #0.132

    plt.title("Egt")
    plt.plot(Err_print)     #13题，在decreasing
    plt.show()
    plt.title("Ein")
    plt.plot(Ein_print)  # 这个应该才对，Ein应该是不带u的。13题，锯齿状的
    plt.show()
    plt.title("EGt")
    plt.plot(EGt)  # 14,也是decreasing
    plt.show()
    plt.title("Ut")   #也是decreasing
    plt.plot(U)  # 15
    plt.show()
    plt.title("epsilon")
    plt.plot(epsilon_print)  # 16，在增加
    plt.show()
    plt.title("Egtout")
    plt.plot(Eoutgt)  # 17  锯齿
    plt.show()
    plt.title("EGtout")
    plt.plot(EGtout)  # 18 先下降再微微上升
    plt.show()



    #12：Ein(g1)是0.24，alpha1是0.57





