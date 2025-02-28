import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


X,y = make_blobs(n_samples = 100,n_features = 2,centers = 2 , random_state=0)
y=y.reshape((y.shape[0],1))

print('demension de X',X.shape)
print('demension de y',y.shape)

plt.scatter(X[:,0] , X[:,1] ,c=y ,cmap='summer')
# plt.show()



def initialisation(X):
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn(1)
    
    return(W,b)


def Model(X,W,b): 
    Z = np.dot(X , W) + b
    A = 1/(1+np.exp(-Z))
    
    return A


def log_loss(A,y):
    return 1/len(y) * np.sum(-y * np.log(A) - (1-y) * np.log(1-A))


def gradient(A,X,y):
    dw = 1/len(y) * np.dot(X.T , A-y)
    db = 1/len(y) * np.sum(A-y)
    return (dw , db)



W,b = initialisation(X)
A=Model(X,W,b)
L = log_loss(A,y)
# print(A)
# print(L)
dw ,db = gradient(A,X,y)
print(dw)
print(db)
