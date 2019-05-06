import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import math


test_data = []
file =  open('train.csv', mode='r') 
raw_data = np.asarray(list(csv.reader(file, delimiter=',')))

#index of photo in images folder corresponding to a sample
photo_indices = raw_data[1:,0].astype(int)

#string describing species of each sample
leaf_labels_strings = raw_data[1:,1]
num_samples_total = len(leaf_labels_strings)

#list of unique leaf names (the classes)
class_names = np.unique(leaf_labels_strings)
num_classes = len(class_names)

print((num_samples_total, num_classes))
#label matrix, rows are one hot vectors, that is labels[a,b]==1 iff sample a is class b
labels = np.zeros((num_samples_total, num_classes))
for i in range(num_samples_total):
    for j in range(num_classes):
        if (leaf_labels_strings[i] == class_names[j]):
            labels[i,j] = 1
        else:
            labels[i,j] = 0

#raw training data
data = raw_data[1:,2:].astype(float)
num_features = data.shape[1]

# 60/29/20 split between train data, valid data, test data
train_data = data[:(3*num_samples_total)//5 ,:]
train_labels =labels[:(3*num_samples_total)//5 ,:]

valid_data = data[(3*num_samples_total)//5 : (4*num_samples_total)//5,:]
valid_labels = labels[(3*num_samples_total)//5 : (4*num_samples_total)//5,:]

test_data = data[(4*num_samples_total)//5 :,:]
test_labels = labels[(4*num_samples_total)//5 :,:]

train_data_pad = np.concatenate( ( np.ones((train_data.shape[0], 1)), train_data ), axis = 1 )
test_data_pad = np.concatenate( ( np.ones((test_data.shape[0], 1)), test_data ), axis = 1 )
valid_data_pad = np.concatenate( ( np.ones((valid_data.shape[0], 1)), valid_data ), axis = 1 )

def pen_loglikelihood(W, X, Y, alpha=0): 
    #compute loglikelihood for current w, b, given the data X, Y
    #W is a p*c matrix, b is a scalar, X is a n*p matrix and Y is a n * c matrix
    ll = 0
    grad = np.zeros(Y.shape[1])
    mus = np.zeros(Y.shape)
    xw = np.matmul(X,W)
    exp_xw = np.exp(xw)
    for i in range(Y.shape[0]):
        expsum = np.sum(np.exp(xw[i,:]))
        logsum = np.log(expsum)
        mus[i,:] = exp_xw[i,:] / expsum
        temp = xw[i:,] - logsum
        ll+= np.sum(np.multiply(Y[i,:], temp))

    ll_penalty = alpha* np.sum(np.square(W)) /2
        # for c in range(Y.shape[1]):

        #     mus[i,c] = np.exp(np.dot(X[i,:], W[:,c])) / expsum
        #     # logsum = np.lo(np.argmax(pred, axis=0)==np.argmax(valid_labels, axis=0)g(np.sum([np.dot(X[i],W[k]) for k in range (Y.shape[1])]))
        #     ll += Y[i,c] * (np.dot(X[i,:],W[:,c]) - logsum)
    
    grad = np.zeros(W.shape)
    for c in range(grad.shape[1]):
        gradval = 0
        for i in range(grad.shape[0]):
            gradval += (Y[i,c] -mus[i,c]) * X[i,:]
        gradPenalty = -alpha * W[:,c]
        gradPenalty[0] = 0
        grad[:,c] =  gradval + gradPenalty

    return ll - ll_penalty, grad
    # X = X.T #X becomes a p*n matrix so the gradVal can be compute straight-forwardly.
    # gradVal = np.dot(X,Y*prob)
    # penalty = alpha/2.*np.sum(W[1:]**2)
    # gradPenalty = -alpha * W
    # gradPenalty[0,:] = 0
    # return -np.sum( np.log( tmp ) ) - penalty, gradVal + gradPenalty

def grad_check(f, xy0, delta=1e-6,tolerance=1e-7):
    _, g0 = f(xy0)
    p = xy0.shape
    finite_diff = np.zeros(p)
    gradient_correct = True
    for xy0 in xy0:
        xy1 = np.copy(xy0)
        xy2 = np.copy(xy0)
        xy1[i] = xy1[i] - 0.5*delta
        xy2[i] = xy2[i] + 0.5*delta
        f1,_ = f(xy1)
        f2,_ = f(xy2)
        finite_diff = (f2 - f1)/(delta)
        if (abs(finite_diff - g0[i])>tolerance):
            print("Broken partial",i," Finite Diff: ",finite_diff," Partial: ",g0[i])
            gradient_correct = False
    return gradient_correct

w_init = np.random.randn(num_features + 1, num_classes )*0.001
w_init[0,:] = 0

def gradient_ascent(f, w, init_step,iterations):  
    f_val,grad = f(w)                           # compute function value and gradient 
    f_vals = [f_val]
    for it in range(iterations):                # iterate for a fixed number of iterations
        print('iteration %d' % it)
        done = False                            # initial condition for done
        line_search_it = 0                      # how many times we tried to shrink the step
        step = init_step                        # reset step size to the initial size
        while not done and line_search_it<100:  # are we done yet?
            new_w = w + step*grad               # take a step along the gradient
            # print(new_w)
            new_f_val,new_grad = f(new_w)       # evaluate function value and gradient
            if new_f_val<f_val:                 # did we go too far?
                step = step*0.95                # if so, shrink the step-size
                line_search_it += 1             # how many times did we shrank the step
            else:
                done = True                     # better than the last x, so we move on
        
        if not done:                            # did not find right step size
            print("Line Search failed.")
        else:
            f_val = new_f_val                   # ah, we are ok, accept the new w
            w = new_w
            grad = new_grad
            f_vals.append(f_val)
        plt.plot(f_vals)
    plt.xlabel('Iterations')
    plt.ylabel('Function value')
    return f_val, w

def optimizeFn( init_step, iterations, alpha, w):
    g = lambda xy0: pen_loglikelihood(xy0, train_data_pad, train_labels, alpha)
    f_val, update_w = gradient_ascent( g, w, init_step, iterations )
    return f_val, update_w

# print ('This should take about 6 seconds.')
# start = time.time()
# f_val, update_w=optimizeFn( init_step = 1e-4
#                            , iterations=100, alpha=0, w = w_init) #set init_step to 1e-4, 1e-5, 1e-6
# end = time.time()
# print ('Time elapsed (seconds):', end-start)
# print ('final log-likelihood = %f\n' % (f_val))

def prediction(w, validData ):
    prob = 1./(1+np.exp(np.matmul(validData,w)))
    print(prob)
    print("prob shape is ", prob.shape)
    res = np.zeros(prob.shape)
    res = res -1 
    for i in range(res.shape[0]):
        res[i,np.argmax(prob[i,:])] = 1
    return res

alphas = [0, 50, 100, 500, 1000, 2000, 1000]
fvals = [0] *(len(alphas))
for i in range(len(alphas)):
    w_init = np.random.randn(num_features + 1, num_classes )*0.001
    w_init[0,:] = 0
    fvals[i], update_w=optimizeFn( init_step = 1e-5, iterations=1000, alpha=alphas[i], w=w_init) #try different alphas [1000, 2000, 3000]

print ('final log-likelihood =', fvals)
    
pred = prediction(update_w, valid_data_pad)
print(pred.shape)
print(valid_labels.shape)
print((np.argmax(pred, axis=1), np.argmax(valid_labels, axis=  1)))
print( 'accuracy on the validation set {:.2f}%'.format( 100.*np.mean(np.argmax(pred, axis=1)==np.argmax(valid_labels, axis=1)) ))

pred = prediction(update_w, test_data_pad)
print( 'accuracy on the test set {:.2f}%'.format( 100.*np.mean(np.argmax(pred, axis=1)==np.argmax(test_labels, axis=1)) ))