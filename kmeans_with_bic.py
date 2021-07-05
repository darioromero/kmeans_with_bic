import numpy as np
import pandas as pd

import sklearn.datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class kmeans_clustering:

    km = None
    bic = None
    aic = None

    def __init__(self, n_components, X):
        self.km = self.get_kmeans(n_components, X)
        self.bic, self.aic = self.get_bic_aic(n_components, X)
        return

    def get_kmeans(self, n_components, X):
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

    def get_bic_aic(self, n_components, X):
        """
        return BIC, AIC scores
        Note: (BIC, AIC) the lower score the better
        """
        gmm = GaussianMixture(n_components=n_components, init_params='random')
        gmm.fit(X)
        return gmm.bic(X), gmm.aic(X)
    
    def get_metrics(self):
        return self.km, self.bic, self.aic

# Test class kmeans_with_bic
from sklearn.datasets import load_sample_images, make_multilabel_classification

X, y = make_multilabel_classification(n_samples=1000, allow_unlabeled=False, random_state=123)

km, bic, aic = kmeans_clustering(3, X).get_metrics()
print(km, bic, aic)
