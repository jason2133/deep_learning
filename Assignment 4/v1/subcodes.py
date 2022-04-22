# Subcodes version 0.1
import torch
import random
import math
import matplotlib.pyplot as plt

def reset_seed(number):
  """
  Reset random seed to the specific number

  Inputs:
  - number: A seed number to use
  """
  random.seed(number)
  torch.manual_seed(number)
  return

def subsample(x,y,ratio=0.2,ran=False):
    ''' 
    Input:
        x : (n, ...)
        y : (n,)
        ratio: validation ratio
        ran : True-random split, False - first:validation/other: training
    Output:
        data_dict: 'X_val', 'y_val', 'X_train', 'y_train'
    
    '''
    n = x.shape[0]
    num_training = int(n*(1-ratio))
    num_validation = n - num_training
    
    if ran:
        ind = torch.randperm(n)
    else:
        ind = torch.arange(0,n)
    data_dict = {}
    data_dict['X_val'] = x[ind[num_training:num_training + num_validation]]
    data_dict['y_val'] = y[ind[num_training:num_training + num_validation]]
    data_dict['X_train'] = x[ind[0:num_training]]
    data_dict['y_train'] = y[ind[0:num_training]]
    return data_dict

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-7):
  """
  Utility function to perform numeric gradient checking. We use the centered
  difference formula to compute a numeric derivative:
  
  f'(x) =~ (f(x + h) - f(x - h)) / (2h)

  Rather than computing a full numeric gradient, we sparsely sample a few
  dimensions along which to compute numeric derivatives.

  Inputs:
  - f: A function that inputs a torch tensor and returns a torch scalar
  - x: A torch tensor giving the point at which to evaluate the numeric gradient
  - analytic_grad: A torch tensor giving the analytic gradient of f at x
  - num_checks: The number of dimensions along which to check
  - h: Step size for computing numeric derivatives
  """
  # fix random seed to 0 
  reset_seed(0)
  for i in range(num_checks):
    
    ix = tuple([random.randrange(m) for m in x.shape])
    
    oldval = x[ix].item()
    x[ix] = oldval + h # increment by h
    fxph = f(x).item() # evaluate f(x + h)
    x[ix] = oldval - h # increment by h
    fxmh = f(x).item() # evaluate f(x - h)
    x[ix] = oldval     # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error_top = abs(grad_numerical - grad_analytic)
    rel_error_bot = (abs(grad_numerical) + abs(grad_analytic) + 1e-12)
    rel_error = rel_error_top / rel_error_bot
    msg = 'numerical: %f analytic: %f, relative error: %e'
    print(msg % (grad_numerical, grad_analytic, rel_error))

def compute_numeric_gradient(f, x, h=1e-7):
  """ 
  Compute the numeric gradient of f at x using a finite differences
  approximation. We use the centered difference:

  df    f(x + h) - f(x - h)
  -- ~= -------------------
  dx           2 * h
  
  Inputs:
  - f: A function that inputs a torch tensor and returns a torch scalar
  - x: A torch tensor giving the point at which to compute the gradient

  Returns:
  - grad: A tensor of the same shape as x giving the gradient of f at x
  """ 
  fx = f(x) # evaluate function value at original point
  flat_x = x.contiguous().view(-1)
  grad = torch.zeros_like(x)
  flat_grad = grad.view(-1)
  # iterate over all indexes in x
  for i in range(flat_x.shape[0]):
    oldval = flat_x[i].item() # Store the original value
    flat_x[i] = oldval + h    # Increment by h
    fxph = f(x).item()        # Evaluate f(x + h)
    flat_x[i] = oldval - h    # Decrement by h
    fxmh = f(x).item()        # Evaluate f(x - h)
    flat_x[i] = oldval        # Restore original value

    # compute the partial derivative with centered formula
    flat_grad[i] = (fxph - fxmh) / (2 * h)

  return grad


def rel_error(x, y, eps=1e-10):
  """
  Compute the relative error between a pair of tensors x and y,
  which is defined as:

                          max_i |x_i - y_i]|
  rel_error(x, y) = -------------------------------
                    max_i |x_i| + max_i |y_i| + eps

  Inputs:
  - x, y: Tensors of the same shape
  - eps: Small positive constant for numeric stability

  Returns:
  - rel_error: Scalar giving the relative error between x and y
  """
  """ returns relative error between x and y """
  top = (x - y).abs().max().item()
  bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
  return top / bot


################# Visualizations #################

def plot_stats(stat_dict):
  # Plot the loss function and train / validation accuracies
  plt.subplot(1, 2, 1)
  plt.plot(stat_dict['loss_history'], 'o')
  plt.title('Loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')

  plt.subplot(1, 2, 2)
  plt.plot(stat_dict['train_acc_history'], 'o-', label='train')
  plt.plot(stat_dict['val_acc_history'], 'o-', label='val')
  plt.title('Classification accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')
  plt.legend()

  plt.gcf().set_size_inches(14, 4)
  plt.show()


def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 4D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (N, H, W, C)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (N, H, W, C) = Xs.shape
  # print(Xs.shape)
  grid_size = int(math.ceil(math.sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = torch.zeros((grid_height, grid_width, C), device=Xs.device)
  next_idx = 0
  y0, y1 = 0, H
  for y in range(grid_size):
    x0, x1 = 0, W
    for x in range(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = torch.min(img), torch.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  return grid


# Visualize the weights of the network
def show_net_weights(net):
  W1 = net.params['W1']
  W1 = W1.reshape(3, 32, 32, -1).transpose(0, 3)
  plt.imshow(visualize_grid(W1, padding=3).type(torch.uint8).cpu())
  plt.gca().axis('off')
  plt.show()


def plot_acc_curves(stat_dict):
  plt.subplot(1, 2, 1)
  for key, single_stats in stat_dict.items():
    plt.plot(single_stats['train_acc_history'], label=str(key))
  plt.title('Train accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')

  plt.subplot(1, 2, 2)
  for key, single_stats in stat_dict.items():
    plt.plot(single_stats['val_acc_history'], label=str(key))
  plt.title('Validation accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')
  plt.legend()

  plt.gcf().set_size_inches(14, 5)
  plt.show()