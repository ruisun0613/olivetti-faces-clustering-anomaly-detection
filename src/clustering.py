from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import numpy as np
import matplotlib.pyplot as plt
# Hierarchical Agglomerative Clustering
def hierarchical_clustering(X_train_pca):
    n_clusters = 40

    clustering = AgglomerativeClustering(n_clusters = n_clusters, metric = 'euclidean', linkage = 'ward')

    labels_train = clustering.fit_predict(X_train_pca)

    print("Agglomerative clustering completed.")
    print("Cluster labels (first 10 samples):", labels_train[:10])

    # Plot dendrogram for 100 samples
    subset = X_train_pca[:100]
    linked = linkage(subset, method = 'ward', metric = 'euclidean')

    plt.figure(figsize = (15,8))
    dendrogram(linked, truncate_mode = 'level', p = 6, color_threshold = 25)
    plt.title("Hierarchical Clustering Dendrogram(ward linkage, 100 samples)")
    plt.xlabel("Sample Index or (Cluster Size)")
    plt.ylabel("Distance")
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.show()

    return linked

# Visualize Clusters
def visualize_clusters(X_train, linked):
    cluster_labels = fcluster(linked, t = 40, criterion = 'maxclust')
    print(f"Number of clusters formed: {len(np.unique(cluster_labels))}\n")

    unique_clusters = np.unique(cluster_labels)
    selected_clusters = np.random.choice(unique_clusters, 10, replace = False)
    subset_indices = np.arange(len(cluster_labels))

    fig,axes = plt.subplots(10, 5, figsize = (10,20))
    fig.suptitle("Sample Faces from Selected Clusters", fontsize = 16)

    for i, cluster_id in enumerate(selected_clusters):
        cluster_indices = subset_indices[cluster_labels == cluster_id]
        for j in range(5):
            if j < len(cluster_indices):
                idx = int(cluster_indices[j])
                axes[i, j].imshow(X_train[idx], cmap = 'gray')
                axes[i, j].set_title(f"Cluster {cluster_id}", fontsize = 10)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis("off")
    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.show()