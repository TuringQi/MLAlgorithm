#coding=utf-8
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# X = pd.read_excel(
#     io='G:/Molecular_Descriptor_Test.xlsx',
#     sheet_name=0,
#     header=None,
#     skiprows=1,
#     # skipcols=0
# )
# X = np.array(X)
# y = pd.read_excel(
#     io='G:/ADMET_Test.xlsx',
#     sheet_name=0,
#     header=None,
#     skiprows=1,
#     # skipcols=0
# )
# y = np.array(y).ravel()
# # 标准化
# X = StandardScaler().fit_transform(X)

def generate_data(n_samples, dataset, noise):
    if dataset == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)
    elif dataset == "circles":
        return datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=1
        )

    elif dataset == "linear":
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1,
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )

X, y = generate_data(n_samples=2000, dataset='moons', noise=0.2)
X = StandardScaler().fit_transform(X)
