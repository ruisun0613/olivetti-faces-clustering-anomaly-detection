import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# random seed
RANDOM_SEED = 42

def apply_pca(data):
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    # Flatten images
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # Fit PCA to retain 99% variance
    pca = PCA(n_components = 0.99, svd_solver = 'full', random_state = RANDOM_SEED)
    pca.fit(X_train_flat)

    print(f"Number of components to retain 99% variance: {pca.n_components_}\n")

    # Transform datasets
    X_train_pca = pca.transform(X_train_flat)
    X_val_pca = pca.transform(X_val_flat)
    X_test_pca = pca.transform(X_test_flat)

    # result
    print("Transformed shapes:")
    print(f"Train: {X_train_pca.shape}")
    print(f"Validation: {X_val_pca.shape}")
    print(f"Test: {X_test_pca.shape}\n")

    # Plot PCA variance curve
    plt.figure(figsize = (8,6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axhline(y = 0.99, color = 'r', linestyle = '--', label = '99% Variance Threshold')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.show()

    return pca, X_train_pca, X_val_pca, X_test_pca