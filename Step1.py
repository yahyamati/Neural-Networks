

# # weights1 = [0.2,0.8,-0.5,1.0]
# # weights2 = [0.5,-0.91,0.26,-0.5]
# # weights3 = [-0.26,-0.27,0.17,0.87]
# weights = [[0.2,0.8,-0.5,1.0],
#            [0.5,-0.91,0.26,-0.5],
#            [-0.26,-0.27,0.17,0.87]]
# # bias1=2
# # bias2=3
# # bias3=0.5
# biases = [2,3,0.5]  


# weights2 = [[0.1,-0.14,-0.5],
#            [-0.5,-0.12,-0.33],
#            [-0.44,0.73,-0.13]]
# # bias1=2
# # bias2=3
# # bias3=0.5
# biases2 = [-1,2,-0.5]  
 

# layer1_output = np.dot(inputs, np.array(weights).T) + biases
# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
 

# print(layer1_output)
# print(layer2_output)

import numpy as np

X = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]] 



class Layer_Dense:
    def __init__(self,n_inputs,n_neurons): 
        self.weights = 0.10*np.random.rand(n_inputs,n_neurons)
        self.biases = np.zeros((1 , n_neurons))
    def forward(self,inputs):
        self.outputs = np.dot(inputs , self.weights) + self.biases
        
        
layer1 = Layer_Dense(4,5)    
layer2 = Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.outputs)
print(layer1.outputs)
print(layer2.outputs)

    
