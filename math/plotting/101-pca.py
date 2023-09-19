#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Complete code for plotting the 3D scatter plot of PCA-reduced Iris dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=labels, cmap='plasma')

# Set labels and title
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.title('PCA of Iris Dataset')

# Add colorbar
cbar = fig.colorbar(scatter)
cbar.set_label('Species (0: Setosa, 1: Versicolor, 2: Virginica)')

# Display the plot
plt.show()