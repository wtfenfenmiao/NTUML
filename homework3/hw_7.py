#这个是用梯度下降法优化，5次之后错误函数E降到了2.825
import numpy as np
def E(u,v):
    return np.exp(u)+np.exp(2*v)+np.exp(u*v)+u*u-2*u*v+2*v*v-3*u-2*v

def gradu(u,v):
    return np.exp(u)+v*np.exp(u*v)+2*u-2*v-3

def gradv(u,v):
    return 2*np.exp(2*v)+u*np.exp(u*v)-2*u+4*v-2

if __name__=="__main__":
    u=0
    v=0
    yita=0.01
    for i in range(5):
        u=u-yita*gradu(u,v)
        v=v-yita*gradv(u,v)
    print (E(u,v))     #2.825