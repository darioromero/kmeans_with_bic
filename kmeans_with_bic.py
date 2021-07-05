import numpy as np
import pandas as pd

import sklearn.datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class kmeans_clustering:
  
  def __init__():
    pass
  
  def get_kmeans(n_components, X):
    km = KMeans(
      n_clusters=n_components, 
      random_state=123,
      init='k-means++',
      max_iter=300,
      n_init=5,
      tol=0.0001
    )
    km.fit(X)
    return km
  
  def get_bic_aic(n_components, X):
    """
    return BIC, AIC scores
    Note: (BIC, AIC) the lower score the better
    """
    gmm = GaussianMixture(n_components=n_components, init_params='random')
    gmm.fit(X)
    return gmm.bic(X), gmm.aic(X)
  
  
    
      
