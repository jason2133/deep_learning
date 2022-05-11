"""
Implements linera regression with regularization as neural networks in PyTorch
"""
# Subcodes version 0.1
import torch
import random
import math
import matplotlib.pyplot as plt

class LinearRegress:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def train_analytic(self, lam=0):
        """
        Parameters
        ----------
        X : A torch tensor for the data. (N, D) 
        y : A torch tensor for the reseponse. (N,1)
        lam : penalty parameter for ridge


        """
        beta = None
        yhat_train = None
        #################################
        # TODO: fit linear regression using the analtyic soltuoin
        # you can use normal_equation function defined below
        # note that your X does not include one-vector for the intercept terms
        # Do not change self.X 
        #################################
        # Replace "pass" statement with your code        
        ones_col = torch.ones((self.X.shape[0], 1))
        X = torch.cat([ones_col, self.X], axis=1)
        beta = normal_equation(self.y,X,lam)
        yhat_train = X @ beta
        #################################
        self.beta = beta
        self.yhat_train = yhat_train

    def predict(self, Xtest):
        """
        Assume that you already trained the model once using whatever optimizer. So you already have self.beta

        Parameters
        ----------
        Xtest : A torch tensor for data to be predicted (N_test, D)
        
        Returns
        ----------
        Yhat_test: A torch tensor for the predicted test set (N_test,1)
        """
        yhat_test = None
        #################################
        # Replace "pass" statement with your code        
        ones_col = torch.ones((Xtest.shape[0], 1))
        X = torch.cat([ones_col, Xtest], axis=1)
        yhat_test = X @ self.beta
        #################################
        return yhat_test
    
    def train_gd(self, niter, lr=1e-4):
        """
        

        Parameters
        ----------
        lr : learning rate
        niter: number of iterations

        Returns(no return/ just change attributes)
        -------
        self. loss_history: A list of losses for each iteration (niter,)
        self.beta, self.yhat_train
        """
        loss_history = []
        beta = None
        yhat_train = None
        
        # initialize beta
        N = self.X.shape[0]
        D = self.X.shape[1]
        reset_seed(0)
        beta = torch.randn(D+1,1)
        
        # use bias trick
        ones_col = torch.ones((self.X.shape[0], 1))
        X = torch.cat([ones_col, self.X], axis=1)
        
        # iterate
        for t in range(niter):            
            #################################
            # TODO: fill out gradient descent algorithm for the linear regression
            #################################
            # Replace "pass" statement with your code
            
            grad = 2/N * X.T.mm(X @ beta - self.y)
            beta -= lr * grad
            yhat_train = X @ beta
            loss = (yhat_train - self.y).square().mean()
            loss_history.append(loss)
            #################################
        
        self.beta = beta
        self.yhat_train = yhat_train
        self.loss_history = loss_history
    
    def train_adam(self, niter, lr=1e-3, beta1=0.9, beta2=0.999):
        """

        Parameters
        ----------
        lr : learning rate
        niter: number of iterations
        beta1, beta2: hyperparameters in adam

        Returns(no return/ just change attributes)
        -------
        self. loss_history: A list of losses for each iteration (niter,)
        self.beta, self.yhat_train

        """
        
        loss_history = []
        beta = None
        yhat_train = None
        
        # initialize beta
        N = self.X.shape[0]
        D = self.X.shape[1]
        reset_seed(0)
        beta = torch.randn(D+1,1)
        moment1 = 0
        moment2 = 0
        
        # use bias trick
        ones_col = torch.ones((self.X.shape[0], 1))
        X = torch.cat([ones_col, self.X], axis=1)
        
        for t in range(niter):
            #################################
            # TODO: fill out gradient descent algorithm for the linear regression
            #################################
            # Replace "pass" statement with your code
            t=t+1
            grad = 2/N * X.T.mm(X @ beta - self.y)
            moment1 = beta1 * moment1 + (1-beta1) * grad
            moment2 = beta2 * moment2 + (1-beta2) * grad * grad
            moment1_unbias = moment1 / (1-beta1**t)
            moment2_unbias = moment2 / (1-beta2**t)
            
            beta -= lr * moment1_unbias / (moment2_unbias.sqrt()+1e-7)
            yhat_train = X @ beta
            loss = (yhat_train - self.y).square().mean()
            loss_history.append(loss)
            #################################
        self.beta = beta
        self.yhat_train = yhat_train
        self.loss_history = loss_history
        
def reset_seed(number):
  """
  Reset random seed to the specific number

  Inputs:
  - number: A seed number to use
  """
  random.seed(number)
  torch.manual_seed(number)
  return

def normal_equation(y_true, X, lam=0):
    """Computes the normal equation for the linear regression
    
    Args:
        y_true: A torch tensor for the response. (N,1)
        X: A torch tensor for the data. (N, D+1) (first column is one-vector)
    Outputs:
        (x^Tx+ \lambda I)^{-1}x^Ty
    """
    p = X.shape[1]
    XTX_inv = (X.T.mm(X)+lam*torch.diag(torch.ones(p))).inverse()
    XTy = X.T.mm(y_true)
    beta = XTX_inv.mm(XTy)
    return beta
  
def gen_linear(N=100,D=20,beta=None, sparse=False, multico=False, dtype=torch.float32,device='cpu'):
    '''
    Inputs:
        N: # of observations
        D: dimension of the input
        beta (optional): `dtype` tensor of shape (D+1, )   # +1; intercept
          If None, give random parameters
        sparse (optional)D>5: sparseness of beta- first 5 betas are zero
        multico (optional)D>7: impose multicollinearity
    Outputs:
        X: `dtype` tensor of shape (N, D) giving data points
        y: `dtype` tensor of shape (N, 1) giving continuous numbers on the real line
        beta: `dtype` tensor of shape (D, 1) 
    '''

    # Generate some random parameters, storing them in a dict
    if beta is None:
      beta = 10*torch.rand(D+1) - 5
      if sparse:
        beta[1:6]=0
    
    X = 2 * torch.rand(N, D)
    if multico:
      X[:,1:4] = X[:,6:8] + 0.01* torch.randn(N,2)
    ones_col = torch.ones((X.shape[0], 1), dtype=dtype)
    Xb = torch.cat([ones_col, X], axis=1)
    y = torch.mv(Xb, beta) + torch.randn(N)
    y = torch.unsqueeze(y,1)
    beta = torch.unsqueeze(beta,1)
    return X, y, beta
