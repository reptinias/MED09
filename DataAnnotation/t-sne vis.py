import numpy as np
import pandas as pd
import cv2
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

datasetPath = "TestDataset - Kopi.csv"

tempdf = pd.read_csv(datasetPath, sep=";")
tempdf["Class"] = tempdf["Class"].fillna(0)
X = tempdf["File"]
y = tempdf["Class"]

images = pd.DataFrame([])

for img in X:
    rgbImg = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    flattenImg = pd.Series(rgbImg.flatten())
    images = images.append(flattenImg, ignore_index=True)

feat_cols = ['pixel'+str(i) for i in range(images.shape[1])]
df = pd.DataFrame(images.values.tolist(), columns=feat_cols)
df["y"] = y
df["label"] = df["y"].apply(lambda i: str(i))
X, y = None, None

np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

N = 893
df_subset = df.loc[rndperm[:N],:].copy()

data_subset = df_subset[feat_cols].values

visString = "pca-t-sne"

if visString == "pca":
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_subset)

    df_subset['pca-one'] = pca_result[:,0]
    df_subset['pca-two'] = pca_result[:,1]
    df_subset['pca-three'] = pca_result[:,2]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.show()

if visString == "t-sne":
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,4))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
    )
    plt.show()

if visString == "pca-t-sne":
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)

    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
    time_start = time.time()

    tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
    df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
    df_subset['tsne-pca50-three'] = tsne_pca_results[:,2]

    plt.figure(figsize=(16,4))
    sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
    )
    plt.show()

    sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-three",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
    )
    plt.show()

    sns.scatterplot(
        x="tsne-pca50-two", y="tsne-pca50-three",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
    )
    plt.show()