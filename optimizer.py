from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math
import time

#### Weights calculation ####

def rbf_kernel(distance_matrix, gamma):
    """This function calculates the RBF kernel of a given distance matrix, considering as input also the gamma value"""
    kernel_matrix = np.exp(-gamma * distance_matrix)
    return kernel_matrix

def normalize_data(points):
  """This function uses a standard scaler to normalize vectors and matrices"""
  scaler = StandardScaler()
  points_normalized = scaler.fit_transform(points)
  
  return points_normalized

def weights_extraction(points_unlabeled, points_labeled, gamma = 3):
  """THis function extract the weights from a list of unlabeled and labeled points, according to the logic of the problem.
  The points are at first normalized, then the eucliean distance is calculated and finally the weights are extracted using
  the RBF kernel function"""
  # Normalize the data
  points_unlabeled_normalized, points_labeled_normalized = normalize_data(points_unlabeled), normalize_data(points_labeled)
  
  # Calculate Euclidean distances
  distances_labeled_unlabeled = euclidean_distances(points_labeled_normalized, points_unlabeled_normalized, squared = True)
  distances_unlabeled_unlabeled = euclidean_distances(points_unlabeled_normalized, points_unlabeled_normalized, squared = True)

  # Calculate the RBF kernel
  weights_labeled_unlabeled = rbf_kernel(distances_labeled_unlabeled, gamma)
  weights_unlabeled_unlabeled = rbf_kernel(distances_unlabeled_unlabeled, gamma)

  return weights_labeled_unlabeled, weights_unlabeled_unlabeled

#### Lipschitz constant calculation ####

def hessian_matrix(weights_labeled_unlabeled, weights_unlabeled_unlabeled, n_unlabeled, alpha = 0.01):
  """This function calculates the Hessian matrix of the loss function"""
  # Diagonal elements calculation
  diag_elements = 2 * (np.sum(weights_labeled_unlabeled, axis = 0) + np.sum(weights_unlabeled_unlabeled, axis = 0) - np.diagonal(weights_unlabeled_unlabeled))
  H_diag = np.diag(diag_elements)

  # Off-diagonal elements calculation
  off_diag_elements = -2 * (weights_unlabeled_unlabeled - np.identity(n_unlabeled))

  # Final sum
  H = off_diag_elements + H_diag

  # Regularization to get a well-conditioned matrix
  H += alpha * np.eye(n_unlabeled)

  return H

def lipschitz_constant(hessian_matrix):
  """This function calculates the Lipschitz constant starting from the hessian matrix"""
  eigenvalues = np.linalg.eigvals(hessian_matrix)
  eigenvalues = np.real(eigenvalues)
  L = np.max(eigenvalues) # Keeping the maximum
  return L

#### Gradient descent ####

def loss(y_unlabeled_init, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled, n_labeled, n_unlabeled):
  """This function calculates the loss function"""
  sq_diff_1 = (y_unlabeled_init.reshape(1, n_unlabeled) - y_labeled.reshape(n_labeled, 1))**2
  term_1 = np.sum(weights_labeled_unlabeled * sq_diff_1)

  sq_diff_2 = (y_unlabeled_init.reshape(n_unlabeled, 1) - y_unlabeled_init.reshape(1, n_unlabeled))**2
  term_2 = np.sum(weights_unlabeled_unlabeled * sq_diff_2)

  return term_1 + 0.5 * term_2

def gradient(y_unlabeled_init, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled):
  """This function calculates the gradient"""
  term_1 = np.sum(weights_labeled_unlabeled * (y_unlabeled_init.reshape(1,- 1) - y_labeled.reshape(-1, 1)), axis=0)
  term_2 = np.sum(weights_unlabeled_unlabeled * (y_unlabeled_init.reshape(1, -1) - y_unlabeled_init.reshape(-1, 1)), axis=0)
  y_j = 2 * term_1 + 2 * term_2

  return y_j.reshape(-1, 1)

def gradient_descent(weights_labeled_unlabeled, weights_unlabeled_unlabeled, y_unlabeled_optimized, y_unlabeled, y_labeled, n_labeled, n_unlabeled, num_iter, learning_rate, verbose = True):
  """This function perform the gradient descent returning an updated version of the input data, a list containing all the
  accuracies and the losses for all the iterations and the toal times needed for the calculation"""
  losses = list()
  accuracies = list()

  start = time.time()

  for i in range(num_iter):
    # Updating weights (with time calculation)

    grad = gradient(y_unlabeled_optimized, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled)
    y_unlabeled_optimized = y_unlabeled_optimized - learning_rate * grad.reshape(n_unlabeled)
    current_loss = loss(y_unlabeled_optimized, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled, n_labeled, n_unlabeled)
    losses.append(current_loss) # Collect losses

    current_accuracy = np.sum(np.round(y_unlabeled_optimized) == y_unlabeled) / n_unlabeled
    accuracies.append(current_accuracy) # Collect accuracies

    # Print execution percentage steps and current loss
    
    progress = i / num_iter * 100
    if verbose and progress % 10 == 0:
      print("Execution progress: {} %, Accuracy: {}".format(int(progress), round(current_accuracy, 4)))

  if verbose:
      print("Final accuracy: {}".format(round(current_accuracy, 4)))

  # Total time
  end = time.time()
  tot_time = end - start

  return y_unlabeled_optimized, losses, tot_time, accuracies

#### Block Coordinate Gradient Descent ####

def block_gradient(y_unlabeled_init, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled, start_idx, end_idx):
  """This function calculates the block version of the gradient"""
  term_1 = np.sum(weights_labeled_unlabeled[:, start_idx:end_idx] * (y_unlabeled_init[start_idx:end_idx].reshape(1, -1) - y_labeled.reshape(-1, 1)), axis=0)
  term_2 = np.sum(weights_unlabeled_unlabeled[start_idx:end_idx, start_idx:end_idx] * (y_unlabeled_init[start_idx:end_idx].reshape(1, -1) - y_unlabeled_init[start_idx:end_idx].reshape(-1, 1)), axis=0)
  y_j = 2 * term_1 + 2 * term_2
  return y_j.reshape(-1, 1)

def block_coordinate_gradient_descent(weights_labeled_unlabeled, weights_unlabeled_unlabeled, y_unlabeled_optimized, y_labeled, y_unlabeled, n_labeled, n_unlabeled, num_iter, learning_rate, block_size, method, verbose=True):
  """This function perform the BCGD returning an updated version of the input data, a list containing all the
  accuracies and the losses for all the iterations and the toal times needed for the calculation"""
  losses = list()
  accuracies = list()
  start = time.time()

  n_blocks = math.ceil(n_unlabeled / block_size)

  # Gauss-Southwell method
  if method == "gauss_southwell":
    block_indices = np.arange(n_blocks)

    gradients = dict()
    norms = dict()

    # This first loop build the gradient vector and the vector of the norms of the gradients, while performing the first iteration
    for block_idx in block_indices:
      start_idx = block_idx * block_size
      end_idx = min(start_idx + block_size, n_unlabeled)

      grad_block = block_gradient(y_unlabeled_optimized, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled, start_idx, end_idx)
      gradients[start_idx] = grad_block
      norms[start_idx] = np.linalg.norm(grad_block)

    max_start = max(norms, key = lambda k : abs(norms[k]))
    max_end = min(max_start + block_size, n_unlabeled)
    grad = gradients[max_start]

    # This second loop perform the remaining iterations updating only the gradient value with the biggest norm in abs
    # With this dynamic programming-based approach the GS method can be quite faster
    for i in range(num_iter - 1):
      grad_block = block_gradient(y_unlabeled_optimized, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled, max_start, max_end)
      gradients[max_start] = grad_block
      norms[max_start] = np.linalg.norm(grad_block)

      max_start = max(norms, key = lambda k : abs(norms[k]))
      max_end = min(max_start + block_size, n_unlabeled)
      grad = gradients[max_start]

      y_unlabeled_optimized[int(max_start):int(max_end)] -= learning_rate * grad.reshape(-1)

      current_loss = loss(y_unlabeled_optimized, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled, n_labeled, n_unlabeled)
      losses.append(current_loss)

      current_accuracy = np.sum(np.round(y_unlabeled_optimized) == y_unlabeled) / n_unlabeled
      accuracies.append(current_accuracy)

      if verbose: # Status print
        progress = i / num_iter * 100
        if progress % 10 == 0:
          print("Execution progress: {} %, Accuracy: {}".format(int(progress), round(current_accuracy, 4)))

    if verbose:
      print("Final accuracy: {}".format(round(current_accuracy, 4)))

  # Random permutation method
  elif method == "random_permutation":
    for i in range(num_iter):
      block_indices = np.random.permutation(n_blocks)

      for block_idx in block_indices:
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, n_unlabeled)
        
        grad_block = block_gradient(y_unlabeled_optimized, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled, start_idx, end_idx)
        y_unlabeled_optimized[start_idx:end_idx] -= learning_rate * grad_block.reshape(-1)

      current_loss = loss(y_unlabeled_optimized, y_labeled, weights_labeled_unlabeled, weights_unlabeled_unlabeled, n_labeled, n_unlabeled)
      losses.append(current_loss)

      current_accuracy = np.sum(np.round(y_unlabeled_optimized) == y_unlabeled) / n_unlabeled
      accuracies.append(current_accuracy)
      
      if verbose: # Status print
        progress = i / num_iter * 100
        if progress % 10 == 0:
          print("Execution progress: {} %, Accuracy: {}".format(int(progress), round(current_accuracy, 4)))

    if verbose:
      print("Final accuracy: {}".format(round(current_accuracy, 4)))

  end = time.time()
  tot_time = end - start
  
  return y_unlabeled_optimized, losses, tot_time, accuracies
