from utilities import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from tqdm import tqdm


X_train , y_train , X_test , y_test = load_data()



print("X_train shape before reshape:", X_train.shape)
print("X_test shape before reshape:", X_test.shape)


X_train = X_train.reshape(X_train.shape[0], -1)  # Shape (num_samples, 4096)
X_test = X_test.reshape(X_test.shape[0], -1)  # Shape (num_samples, 4096)

print("X_train shape after reshape:", X_train.shape)
print("X_test shape after reshape:", X_test.shape)

#//////////////////////////////////////////////////////////////////////////////////////////////////////


print("Train set before normalizing=> min:", X_train.min(), "max:", X_train.max())
print("Test set before normalizing=> min:", X_test.min(), "max:", X_test.max()) 

#normalize the train and test sets (scaling pixel values from 0-255 to 0-1)
X_train = X_train / X_train.max() # /255
X_test = X_test / X_test.max() # /255

print("Train set after normalizing=> min:", X_train.min(), "max:", X_train.max())
print("Test set after normalizing=> min:", X_test.min(), "max:", X_test.max()) 

#//////////////////////////////////////////////////////////////////////////////////////////////////////


# print("y_train shape before flattening:", y_train.shape)
# print("y_test shape before flattening:", y_test.shape)
# y_train =y_train.flatten()
# y_test = y_test.flatten()
# print("y_train shape after flattening:", y_train.shape)
# print("y_test shape after flattening:", y_test.shape)


#//////////////////////////////////////////////////////////////////////////////////////////////////////


# plt.figure(figsize=(16,8))
# for i in range(1,10):
#     plt.subplot(4,5,i)
#     plt.imshow(X_train[i],cmap='gray')
#     plt.title(y_train[i])
#     plt.tight_layout()
# plt.show()



def initialisation(X_train):
    W = np.random.randn(X_train.shape[1],1)
    b = np.random.randn(1)
    
    return(W,b)


def Model(X_train,W,b): 
    Z = np.dot(X_train , W) + b
    A = 1/(1+np.exp(-Z))
    
    return A


# def log_loss(A,y_train):
#     return 1/len(y_train) * np.sum(-y_train * np.log(A) - (1-y_train) * np.log(1-A))


def gradient(A,X_train,y_train):
    dw = 1/len(y_train) * np.dot(X_train.T , A-y_train)
    db = 1/len(y_train) * np.sum(A-y_train)
    return (dw , db)


def update(W,b,dw,db,learning_rate):
    W = W - learning_rate * dw
    b = b - learning_rate * db
    
    return (W,b)



def predict(X_train,W,b):
    A = Model(X_train,W,b)
    # print("predict point==>",A)
    return A>=0.5
    


def artificial_neuron(X_train,y_train,learning_rate=0.01,nb_iteration = 10000):
    # y_train = y_train.reshape(-1, 1)
    #initialisation
    W,b = initialisation(X_train)
    
    loss = []
    acc= []
    for i in tqdm(range(nb_iteration)):
        A = Model(X_train,W,b)
        # print("A==>",A.shape)
        # print("y_train==>",y_train.shape)
        
        if i%10==0:
            #calcul du cout
            loss.append(log_loss(y_train, A ))
            
            #calcul de l'accuracy
            y_pred = predict(X_train,W,b)
            acc.append(accuracy_score(y_train,y_pred))
            # print(f"accuracy_score=>{acc*100}%")
        
        #update
        dw,db =gradient(A,X_train,y_train)
        W,b = update(W,b,dw,db,learning_rate)
        
        
        
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(loss)
    plt.subplot(1,2,2)
    plt.plot(acc)
    # plt.xlabel("Iterations")
    # plt.ylabel("Log Loss")
    # plt.title("Loss Curve")
    plt.show()
    return (W,b)
    
    
W,b = artificial_neuron(X_train,y_train) 
print("the best w=>",W)
print("the best b=>",b)   





# #predict a new point 
# y_pred_test = predict(X_test, W, b)

# #tracer la frontiere de decision
# x0 = np.linspace(-1, 4, 100)  # X-axis range
# x1 = (-W[0] * x0 - b) / W[1]  # Decision boundary equation

# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='summer', label="Train Data")
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap='coolwarm', marker='s', label="Test Predictions")
# plt.plot(x0, x1, c='b', lw=3, label="Decision Boundary")
# plt.legend()
# plt.show()














