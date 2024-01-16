from sklearn.datasets import make_blobs, make_circles, make_moons

def make_data(n_samples = 1000, type = "blobs", random_state = None):
  """This function makes synthetic random data, using giving the number of examples, the shape type and the randon_state"""
  if type == "blobs": # Blobs
    points, labels = make_blobs(n_samples = n_samples,
                                n_features = 2,
                                centers = 2,
                                cluster_std = 2.0,
                                center_box = (-10.0, 10.0),
                                shuffle = True,
                                random_state = random_state,
                                return_centers = False)
  elif type == "circles": # Concentric circles
    points, labels = make_circles(n_samples = n_samples,
                                  shuffle = True,
                                  noise = 0.05,
                                  random_state = random_state,
                                  factor = 0.5)
  elif type == "moons": # Moons
    points, labels = make_moons(n_samples = n_samples,
                                shuffle = True,
                                noise = 0.05,
                                random_state = random_state)

  return points, labels

