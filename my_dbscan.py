import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd


path = "dataset/Apple_fixation_dataset"
txt_files = glob.glob(path + "/*.txt")
img = plt.imread("dataset/APPLE_segmented.png")

# names=["FixationIndex", "Timestamp", "FixationDuration", "MappedFixationPointX", "MappedFixationPointY", "StimuliName"]

all_data_frames = []

for file in txt_files:
    if os.stat(file).st_size == 0:
        print("Empty file:", file)
        continue
    print('Location:', file)
    print('File Name:', file.split("\\")[-1])
    data = pd.read_csv(file, sep="\t")
    all_data_frames.append(data)
print("Total files:", len(all_data_frames))
complete_data = pd.concat(all_data_frames)

X_train = complete_data[["MappedFixationPointX", "MappedFixationPointY"]]
X = np.array(X_train)
print(X)

eps_values = [10, 12.5, 15, 20, 25, 30]
min_samples = [5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20, 25]

#Good Values
eps_values = [30]
min_samples = [25]

for i in range(len(eps_values)):
    for j in range(len(min_samples)):
        clusterer = DBSCAN(eps=eps_values[i], min_samples=min_samples[j], metric='euclidean')
        y_pred = clusterer.fit_predict(X)
        cluster = clusterer.labels_
        print(f'clusters: {len(set(cluster))}, results: {cluster}')

        plt.figure(figsize=(12, 9))
        plt.imshow(img)
        plt.annotate('', xy=(0.03, 0.95), xycoords='axes fraction')
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='Dark2')
        plt.title('DBSCAN')
        plt.show()

        print("Eps:", eps_values[i], "Min_samples:", min_samples[j])