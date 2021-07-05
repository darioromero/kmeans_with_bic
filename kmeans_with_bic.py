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

# generate some data
X, y = make_multilabel_classification(n_samples=1000, allow_unlabeled=False, random_state=123)

from sklearn.preprocessing import MinMaxScaler
# scale data X
scaler = MinMaxScaler()
Xt = scaler.fit_transform(X)

# reduce dimensionality by applying PCA
pca = PCA(n_components=20, whiten=True, random_state=123)
# fit & transform scaled feature array
components = pca.fit_transform(Xt)

# cummulative explained variance
cum_pca_exp_var = np.cumsum(pca.explained_variance_ratio_)

# taking only upto 90% cummulative variance
pca_lt90 = cum_pca_exp_var[np.where(cum_pca_exp_var < 0.91)]

# prepare names for PCA columns
cols = ['PC'+str(num+1) for num in range(len(pca_lt90))]

# save cummulative variance, pca components
pca_df = pd.DataFrame(components[:, :len(cols)], columns=cols)

Xpca = pca_df.values

km, bic, aic = kmeans_clustering(3, Xpca).get_metrics()
print(km, bic, aic)
