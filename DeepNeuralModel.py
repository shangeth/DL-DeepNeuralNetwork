import numpy as np
import pickle
import os
import h5py
import glob
import cv2
from keras.preprocessing import image
import matplotlib.pyplot as plt


#-------------------------------------------------------------------------------------------
class DeepNeuralNetwork:
    def __init__(self,layers_dims, initialization='random'):
        self.layers_dims = layers_dims
        self.initialization = initialization
        
    
    def initialize_parameters_deep(self, layer_dims):
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)
        for l in range(1, L):
            if self.initialization == 'he':
                factor = np.sqrt(2/layer_dims[l-1])
            elif self.initialization == 'xavier':
                factor = np.sqrt(1/layer_dims[l-1])
            else :
                factor = 0.01
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * factor
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))   
        return parameters
    
    
    
    def linear_forward(self, A, W, b):
        Z = np.dot(W,A) + b
        cache = (A, W, b)
        return Z, cache
    
    def sigmoid(self, Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache

    def relu(self, Z):
        A = np.maximum(0,Z)
        cache = Z 
        return A, cache


    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True) 
        dZ[Z <= 0] = 0
        return dZ

    def sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ


    
    
    def linear_activation_forward(self, A_prev, W, b, activation):   
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        cache = (linear_cache, activation_cache)
        return A, cache
    
    
    
    
    def L_model_forward(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) //2   
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            caches.append(cache)
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)     
        return AL, caches
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -(np.dot(Y,np.log(AL.T))+np.dot((1-Y),np.log(1-AL).T))/m
        cost = np.squeeze(cost)    
        return cost
    
    
    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = np.dot(dZ,A_prev.T)/m
        db = np.mean(dZ,axis=1,keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        return dA_prev, dW, db
    
    
    
    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    
    
    
    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp    
        return grads

    def update_parameters(self, parameters, grads, learning_rate):   
        L = len(parameters) //2
        for l in range(L):
            # print(parameters["W" + str(l+1)])
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW"+ str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db"+ str(l+1)]
        return parameters 
    
    
    def fit(self, X_train, Y_train, num_iterations=2000, learning_rate=0.01, print_cost=True):
        np.random.seed(1)
        self.costs = []                         
        parameters = self.initialize_parameters_deep(self.layers_dims)        
        # (gradient descent)
        for i in range(0, num_iterations):
            AL, caches = self.L_model_forward(X_train, parameters)
            cost = self.compute_cost(AL, Y_train)
            grads = self.L_model_backward(AL, Y_train, caches)
            self.parameters = self.update_parameters(parameters, grads, learning_rate)

            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                self.costs.append(cost)
        Y_predict_train = self.predict(X_train)
        print ("Accuracy of Training Dataset : {} %".format(100-np.mean(np.abs(Y_predict_train - Y_train)) * 100))        

        # plot the cost
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()


    def predict(self, X):
        m = X.shape[1]
        n = len(self.parameters) // 2
        p = np.zeros((1,m))
        probas, caches = self.L_model_forward(X, self.parameters)
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0            
        return p    


    def predict_image(self,X):
        Y_predict = None
        w = self.parameters["w"]
        b = self.parameters["b"] 
        w = w.reshape(X.shape[0], 1)
        A = self.sigmoid(np.dot(w.T, X) + b)
        for i in range(A.shape[1]):
            if A[0, i] <= 0.5:
                Y_predict = 0
            else:
                Y_predict = 1
        return Y_predict


    def predict_my_images(self, test_img_paths, image_size, train_path):
        for test_img_path in test_img_paths:
            img_to_show    = cv2.imread(test_img_path, -1)
            img            = image.load_img(test_img_path, target_size=image_size)
            x              = image.img_to_array(img)
            x              = x.flatten()
            x              = np.expand_dims(x, axis=1)
            predict1        = self.predict(x)
            predict_label  = ""
            train_labels = os.listdir(train_path)
            if predict1 == 0:
                predict_label = str(train_labels[0])
            else:
                predict_label = str(train_labels[1])
            cv2.putText(img_to_show, predict_label, (30,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            plt.imshow(img_to_show)
            plt.show() 


#-----------------------------------------------------------------------------------------        
def save_model(model_obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model_obj, f)
    print("Model saved to {}".format(str(file_name)))    

#----------------------------------------------------------------------------------------- 
def open_saved_model(file_name):
    with open(file_name, 'rb') as f:
        clf = pickle.load(f)
    return clf    
 
#-----------------------------------------------------------------------------------------         
def prepare_image_dataset(train_path, test_path, image_size, num_train_images, num_test_images, num_channels=3):    
    train_labels = os.listdir(train_path)
    test_labels  = os.listdir(test_path)
    train_x = np.zeros(((image_size[0]*image_size[1]*num_channels), num_train_images))
    train_y = np.zeros((1, num_train_images))
    test_x  = np.zeros(((image_size[0]*image_size[1]*num_channels), num_test_images))
    test_y  = np.zeros((1, num_test_images))

    #train dataset
    count = 0
    num_label = 0
    for i, label in enumerate(train_labels):
        cur_path = train_path + "/" + label
        for image_path in glob.glob(cur_path + "/*.jpg"):
            img = image.load_img(image_path, target_size=image_size)
            x   = image.img_to_array(img)
            x   = x.flatten()
            x   = np.expand_dims(x, axis=0)
            train_x[:,count] = x
            train_y[:,count] = num_label
            count += 1
        num_label += 1

    count = 0 
    num_label = 0 
    for i, label in enumerate(test_labels):
        cur_path = test_path + "/" + label
        for image_path in glob.glob(cur_path + "/*.jpg"):
            img = image.load_img(image_path, target_size=image_size)
            x   = image.img_to_array(img)
            x   = x.flatten()
            x   = np.expand_dims(x, axis=0)
            test_x[:,count] = x
            test_y[:,count] = num_label
            count += 1
        num_label += 1

    train_x = train_x/255.
    test_x  = test_x/255.

    print ("train_labels : " + str(train_labels))
    print ("train_x shape: " + str(train_x.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x shape : " + str(test_x.shape))
    print ("test_y shape : " + str(test_y.shape))
    
    return train_x, test_x, train_y, test_y

#----------------------------------------------------------------------------------------- 
def accuracy(y_pred, y_true):
    return 100-np.mean(np.abs(y_pred - y_true)) * 100

#----------------------------------------------------------------------------------------- 
