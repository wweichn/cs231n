import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_training = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_training):
        scores = X[i].dot(W)
        scores = scores - np.max(scores)
        exp_sum = np.sum(np.exp(scores))
        correct_class_score = scores[y[i]]
        loss -= np.log(np.exp(scores[y[i]])/exp_sum)
        for j in xrange(num_classes):
            if j != y[i]:              
                dW[:,j] += (np.exp(scores[j])/exp_sum) * X[i] 
            else:
                dW[:,j] += (np.exp(scores[j])/exp_sum - 1) * X[i]
  loss = loss / num_training + 0.5 * reg * np.sum(W * W)
  dW = dW / num_training + reg * W
  return loss, dW


 

  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  scores = X.dot(W)
  num_training = X.shape[0]
  num_classes = W.shape[1]
  scores = scores - np.max(scores,axis = 1).reshape(-1,1)
  exp_sum = np.sum(np.exp(scores),axis = 1)   # n_training * 1 
  exp_score = np.exp(scores[np.arange(num_training),y])
  loss = - np.sum(np.log(np.true_divide(exp_score,exp_sum))) / num_training + 0.5 *reg * np.sum(W * W)
  
  
  prob = np.true_divide(np.exp(scores) , exp_sum.reshape(-1,1))
  prob[np.arange(num_training),y] -= 1
  dW = np.transpose(X).dot(prob) / num_training +  reg * W
  return loss,dW

  
  ##########################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

