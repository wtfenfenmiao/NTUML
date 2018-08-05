#coding:utf-8
import pandas as pd
import numpy as np


con = 0

class Node:
    def __init__(self,feature=None,theta=None,s=None,left=None,right=None,value=None):
        self.feature=feature
        self.theta=theta
        self.s=s
        self.left=left   #s，theta，feature算得是负，就去左边。叶子结点这个就是空的   #划分的时候也是，正1得那面要分到左面，-1的那面分到右面
        self.right=right  #s，theta，feature算得是正，就去右边。叶子结点这个就是空的
        self.value=value  #如果是上一个得左分支，就是-1。右分支就是1。



def sign(x):
    return (x>0)*2-1

#train_data是没排序的
#theta是把这个数据按照这个特征排序之后中间的那个分界
#算纯度，哪个特征纯度大就用哪个



def find_feature_theta_s(train_data):
    features=train_data.columns[0:-1]
    label=train_data.columns[-1]
    train_len=len(train_data)
    final_theta=0
    final_s=1
    final_feature=features[0]
    final_impurity=train_len
    for feature in features:
        train=train_data.sort_values(by=[feature]).reset_index(drop=True)
        theta_left_new_index=int((train_len-1)/2)
        theta_right_new_index=theta_left_new_index+1
        theta=(train.iloc[theta_left_new_index,feature]+train.iloc[theta_right_new_index,feature])/2
        #算个s，要正确率高的
        s=1
        if(sum(sign(train[feature]-theta)==train[label])/train_len<0.5):
            s=-1
        #算纯度
        left_class_positive_1=sum(train.iloc[0:theta_right_new_index,label]==1)/len(train.iloc[0:theta_right_new_index,label])   #算这个的纯度
        left_class_negtive_1=1-left_class_positive_1
        right_class_positive_1=sum(train.iloc[theta_right_new_index:, label]==1)/len(train.iloc[theta_right_new_index:, label])
        right_class_negtive_1=1-right_class_positive_1
        impurity_left=len(train.iloc[0:theta_right_new_index,label])*(1-(left_class_positive_1)*(left_class_positive_1)-(left_class_negtive_1)*(left_class_negtive_1))
        impurity_right=len(train.iloc[theta_right_new_index:, label])*(1-(right_class_positive_1)*(right_class_positive_1)-(right_class_negtive_1)*(right_class_negtive_1))
        impurity=impurity_left+impurity_right
        if(impurity<final_impurity):
            final_impurity=impurity
            final_theta=theta
            final_s=s
            final_feature=feature

    return final_theta,final_s,final_feature


def prune_CART_addNode(train_data):    #19-20用的，剪枝的树，只有一层。
    theta, s, feature = find_feature_theta_s(train_data)
    root = Node(feature=feature, theta=theta, s=s, value=0)
    root.left=Node(value=-1)
    root.right=Node(value=1)
    return root



def CART_addNode(train_data,pre_value):   #建树的算法，叶子结点只有value，别的啥都没有。中间的节点，每个都有s theta feature，根据这个划分左右子树。
    if(len(train_data)==1):    #x分不开的情况（每次算theta都是对半劈的，这样应该没事）
        root=Node(value=pre_value)
        return root


    label = train_data.columns[-1]     #y分不开的情况
    if (sum(train_data.iloc[:, label] == 1) == 0) or (sum(train_data.iloc[:, label] == -1) == 0):
        root = Node(value=pre_value)
        return root

    theta,s,feature=find_feature_theta_s(train_data)
    root=Node(feature=feature,theta=theta,s=s,value=pre_value)
    train = train_data.sort_values(by=[feature]).reset_index(drop=True)
    theta_right_index = int((len(train) + 1) / 2)
    if s==1:   #保证左负右正
        left_data=train.iloc[0:theta_right_index,:].reset_index(drop=True)
        right_data=train.iloc[theta_right_index:,:].reset_index(drop=True)
    else:     #保证左负右正
        right_data = train.iloc[0:theta_right_index, :].reset_index(drop=True)
        left_data = train.iloc[theta_right_index:, :].reset_index(drop=True)

    root.left=CART_addNode(left_data,-1)   #这个1和-1.只有叶子结点有用，根节点没用
    root.right=CART_addNode(right_data,1)
    return root


#写一个test的，就是用决策树的函数。test_data进来的时候要有预测值这一列，没有值没关系
def predict(test_data,root,label_predict):
    #print ("start:")
    #print (test_data)
    if len(test_data)==0:
        return test_data
    if root.left==None and root.right==None:
        test_data[label_predict]=root.value
        return test_data
    else:
        #print ("root value:")
        #print (root.s,root.feature,root.theta)
        val=(root.s)*sign(test_data[root.feature]-root.theta)
        #print (val)
        test_data[val == 1]=predict(test_data[val==1],root.right,label_predict)
        test_data[val == -1]=predict(test_data[val==-1],root.left,label_predict)
        return test_data

#遍历树的,前序遍历
def traversal_tree(root):
    global con
    con += 1
    if root.left!=None and root.right!=None:
        print(root.s,root.theta,root.feature,con)
        traversal_tree(root.left)
        traversal_tree(root.right)
    else:
        print("leaf",root.value,con)




if __name__=="__main__":
    train_data = pd.read_table("hw7_train.dat", header=None, delim_whitespace=True)
    test_data = pd.read_table("hw7_test.dat",header=None,delim_whitespace=True)

    print("training the tree:")
    root=CART_addNode(train_data, 0)

    print("traversal tree:")
    traversal_tree(root)

    print("predicting:")
    train_data["predict"]=0
    test_data["predict"]=0

    #print (train_data)
    predict(train_data,root,"predict")
    predict(test_data,root,"predict")


    #print (train_data)
    #print (test_data)
    Ein=sum(train_data[2]!=train_data["predict"])/len(train_data)
    Eout=sum(test_data[2]!=test_data["predict"])/len(test_data)
    print("Ein:",Ein)    #0
    print("Eout:",Eout)   #0.156














