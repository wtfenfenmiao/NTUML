#这个是用牛顿法优化，5次之后错误函数E降到了2.361，这个说明了一下牛顿法比梯度下降法降得快，收敛的快
import numpy as np
def E(u,v):
    return np.exp(u) + np.exp(2 * v) + np.exp(u * v) + u * u - 2 * u * v + 2 * v * v - 3 * u - 2 * v
def gradu(u,v):
    return np.exp(u)+v*np.exp(u*v)+2*u-2*v-3
def gradv(u,v):
    return 2*np.exp(2*v)+u*np.exp(u*v)-2*u+4*v-2
def graduu(u,v):
    return np.exp(u)+v*v*np.exp(u*v)+2
def gradvv(u,v):
    return 4*np.exp(2*v)+u*u*np.exp(u*v)+4
def graduv(u,v):
    return np.exp(u*v)+u*v*np.exp(u*v)-2
def HessianReverse(u,v):
    hessian=np.zeros((2,2))
    hessian[0][0]=graduu(u,v)
    hessian[0][1]=graduv(u,v)
    hessian[1][0]=graduv(u,v)
    hessian[1][1]=gradvv(u,v)
    return np.mat(hessian).I     #np.mat是矩阵，可以进行转置或者求逆，np.array是数组，没办法进行转置或者求逆这些操作
def neplaE(u,v):
    re=np.zeros((2,1))
    re[0]=gradu(u,v)
    re[1]=gradv(u,v)
    return np.mat(re)

if __name__=="__main__":
    to_update=np.mat([[0],[0]])
    test=np.mat([[4],[5]])
    #print (test)
    #print (test[0][0])
    #print (test[1][0])
    for i in range(5):
        u=to_update[0]
        v=to_update[1]
        to_update=to_update-np.dot(HessianReverse(u,v),neplaE(u,v))
        #print (HessianReverse(u,v))
        #print (neplaE(u,v))
    print (E(to_update[0],to_update[1]))   #2.361


