import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


train_dataset = h5py.File('train_catvnoncat.h5',"r")
test_dataset = h5py.File('test_catvnoncat.h5',"r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_set_y = np.array(train_dataset["train_set_y"][:]).reshape((1,209)) # your train set labels
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_set_y = np.array(test_dataset["test_set_y"][:]).reshape((1,50)) # your test set labels
classes = np.array(test_dataset["list_classes"][:])


m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
x_train_flatten = train_set_x_orig.reshape(m_train,-1).T
x_test_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
test_set_x = x_test_flatten/255
train_set_x = x_train_flatten/255

def sigmoid(z):
    return 1/(1+np.exp(-z))
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(9.2) = " + str(sigmoid(9.2)))

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim,1))
    return w,b

def propagate(w,x,y,b):
    m = x.shape[0]
    z = np.dot(w.T,x) + b
    a = sigmoid(z)
    J = -np.sum(y * np.log(a) + ((1-y) * np.log(1-a)))/m
    db = np.sum(a-y)/m
    dw = np.dot(x,(a-y).T)/m
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(J)
    assert(cost.shape == ())
    grads = {"dw":dw,"db":db}
    return grads, cost

def optimize(w,b,x,y,num_iter,alpha):
    costs = []
    for i in range(num_iter):
        grads,cost = propagate(w,x,y,b)
        w = w - grads["dw"]*alpha
        b = b - grads["db"]*alpha
        if i%100 == 0:
            costs.append(cost)
    params = {"w":w,"b":b}
    grad = {"w":grads["dw"],"b":grads["db"]}
    return params, grad, costs


def predict(w,x,b):
    z_cap = np.dot(w.T,x) + b
    y_cap = sigmoid(z_cap)
    y_pred = np.zeros((1,x.shape[1]))
    for i in range(y_cap.shape[1]):
        if y_cap[0,i]>0.5:
            y_pred[0][i]=1
        else:
            y_pred[0,i]=0
    assert(y_pred.shape == (1,x.shape[1]))
    return y_pred


def model(x_train,y_train,x_test,y_test,num_iters=2000,alpha=0.5):
    w = np.zeros((x_train.shape[0],1))
    b = 0
    params,grads,costs = optimize(w,b,x_train,y_train,num_iters,alpha)
    w = params["w"]
    b = params["b"]
    y_pred_test = predict(w,x_test,b)
    y_pred_train = predict(w,x_train,b)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": y_pred_test, 
         "Y_prediction_train" : y_pred_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : alpha,
         "num_iterations": num_iters}
    
    return d
    
d = model(train_set_x,train_set_y,test_set_x,test_set_y)
my_image = "cat.jpg" #name of the image you want to test
fname = "/home/destroyer/MLimages/" + my_image # here enter the path to your image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(d["w"], my_image,d["w"])

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", the algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
