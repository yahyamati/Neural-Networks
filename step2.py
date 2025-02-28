import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score



X,y = make_blobs(n_samples = 100,n_features = 5,centers = 2 , random_state=0)
y=y.reshape((y.shape[0],1))


# print('demension de X',X.shape)
# print('demension de y',y.shape)

# plt.scatter(X[:,0] , X[:,1] ,c=y ,cmap='summer')
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


def update(W,b,dw,db,learning_rate):
    W = W - learning_rate * dw
    b = b - learning_rate * db
    
    return (W,b)



def predict(X,W,b):
    A = Model(X,W,b)
    print("predict point==>",A)
    return A>=0.5
    


def artificial_neuron(X,y,learning_rate=0.1,nb_iteration = 100):
    #initialisation
    W,b = initialisation(X)
    
    loss = []
    for i in range(nb_iteration):
        A = Model(X,W,b)
        loss.append(log_loss(A,y))
        dw,db =gradient(A,X,y)
        #update
        W,b = update(W,b,dw,db,learning_rate)
        
    y_pred = predict(X,W,b)
    print(f"accuracy_score=>{accuracy_score(y,y_pred)*100}%")
        
    
    # plt.plot(loss)
    # plt.show()
    return (W,b)
    
    
W,b = artificial_neuron(X,y) 
print("the best w=>",W)
print("the best b=>",b)   





#predict a new point 
new_plant = np.array([1,1,1,2,3])

#tracer la frontiere de decision
x0 = np.linspace(-1,4,100)
x1 =(-W[0]*x0-b)/ W[1]

plt.scatter(X[:,0] , X[:,1] ,c=y ,cmap='summer')
plt.scatter(new_plant[0] , new_plant[1] , c='r')
#tracer la frontiere de decision
plt.plot(x0,x1,c='b',lw=3)
plt.show()
a = predict(new_plant,W,b)
print("the predict of the new point =>",a)







