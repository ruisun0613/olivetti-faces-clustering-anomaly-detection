"""
@Author  : Rui Sun
@File    : Exercise1.py
@Software: PyCharm
@Time    : 2025-10-15 10:19 p.m.
@Description: PCA + Agglomerative + GMM Clustering + Anomaly Detection on Olivetti Faces
"""

# ----------------------------- Import Required Libraries -----------------------------
import os
os.environ["OMP_NUM_THREADS"] = "2"  # Prevent MKL multi-threading issue on Windows
import tensorflow as tf
import imageio
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from skimage import exposure, transform

# ----------------------------- 1. Load and Visualize Dataset -----------------------------
print("\n====================== Step 1: Load and Visualize Dataset ======================\n")

data = fetch_olivetti_faces(shuffle=True,)
X_Rui = data.images
y_Rui = data.target

print("Dataset shapes:")
print(f"X (images): {X_Rui.shape}")
print(f"y (labels): {y_Rui.shape}\n")

# Convert to TensorFlow tensor (optional)
X_Rui_tf = tf.convert_to_tensor(X_Rui, dtype=tf.float32)
y_Rui_tf = tf.convert_to_tensor(y_Rui, dtype=tf.float32)

# Display a few sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_Rui[i], cmap='gray')
    ax.set_title(f"Label: {y_Rui[i]}")
    ax.axis('off')
plt.suptitle("Sample Olivetti Faces", fontsize=16)
plt.show()

# ----------------------------- 2. Split Dataset into Train / Validation / Test -----------------------------
print("\n====================== Step 2: Train / Validation / Test Split ======================\n")

X_train, X_temp, y_train, y_temp = train_test_split(
    X_Rui, y_Rui,
    test_size=0.3,
    stratify=y_Rui,
    random_state=42
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

print(f"Training set : {X_train.shape}, Validation set : {X_val.shape}, Test set : {X_test.shape}\n")

# Visualize label distribution
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
sns.countplot(x=y_train, ax=axes[0], color="steelblue")
axes[0].set_title("Training Set Distribution")

sns.countplot(x=y_val, ax=axes[1], color="darkorange")
axes[1].set_title("Validation Set Distribution")

sns.countplot(x=y_test, ax=axes[2], color="seagreen")
axes[2].set_title("Test Set Distribution")

plt.suptitle("Class Distribution Across Train/Validation/Test Sets", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ----------------------------- 3. PCA Dimensionality Reduction -----------------------------
print("\n====================== Step 3: Apply PCA (99% Variance Retained) ======================\n")

# Flatten images
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)
print(f"Flattened shape: {X_train_flat.shape}\n")

# Fit PCA to retain 99% variance
pca = PCA(n_components=0.99, svd_solver='full', random_state=42)
pca.fit(X_train_flat)

print(f"Number of components to retain 99% variance: {pca.n_components_}\n")

# Transform datasets
X_train_pca = pca.transform(X_train_flat)
X_val_pca = pca.transform(X_val_flat)
X_test_pca = pca.transform(X_test_flat)

print("Transformed shapes:")
print(f"Train: {X_train_pca.shape}")
print(f"Validation: {X_val_pca.shape}")
print(f"Test: {X_test_pca.shape}\n")

# Plot PCA variance curve
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.99, color='r', linestyle='--', label='99% Variance Threshold')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------- 4. Hierarchical Agglomerative Clustering -----------------------------
print("\n====================== Step 4: Hierarchical Agglomerative Clustering ======================\n")

n_clusters = 40
clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
labels_train = clustering.fit_predict(X_train_pca)

print("Agglomerative clustering completed.")
print("Cluster labels (first 10 samples):", labels_train[:10])

# Plot dendrogram for 100 samples
subset = X_train_pca[:100]
linked = linkage(subset, method='ward', metric='euclidean')

plt.figure(figsize=(15, 8))
dendrogram(linked, truncate_mode='level', p=6, color_threshold=25)
plt.title("Hierarchical Clustering Dendrogram (ward linkage, 100 samples)")
plt.xlabel("Sample Index or (Cluster Size)")
plt.ylabel("Distance")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------- 5. Visualize Clusters -----------------------------
print("\n====================== Step 5: Visualize Selected Clusters ======================\n")

cluster_labels = fcluster(linked, t=40, criterion='maxclust')
print(f"Number of clusters formed: {len(np.unique(cluster_labels))}\n")

unique_clusters = np.unique(cluster_labels)
selected_clusters = np.random.choice(unique_clusters, 10, replace=False)
subset_indices = np.arange(len(cluster_labels))

fig, axes = plt.subplots(10, 5, figsize=(10, 20))
fig.suptitle("Sample Faces from Selected Clusters", fontsize=16)

for i, cluster_id in enumerate(selected_clusters):
    cluster_indices = subset_indices[cluster_labels == cluster_id]
    for j in range(5):
        if j < len(cluster_indices):
            idx = cluster_indices[j]
            axes[i, j].imshow(X_train[idx], cmap='gray')
            axes[i, j].set_title(f"Cluster {cluster_id}", fontsize=10)
            axes[i, j].axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ----------------------------- 6. Gaussian Mixture Model (GMM) Clustering -----------------------------
print("\n====================== Step 6: Gaussian Mixture Model (GMM) Clustering ======================\n")

cov_types = ["full", "tied", "diag", "spherical"]
k_range = range(20, 61, 5)
aic_scores = {cv: [] for cv in cov_types}
bic_scores = {cv: [] for cv in cov_types}

# Fit multiple GMMs and record AIC/BIC
for cv in cov_types:
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type=cv, random_state=42)
        gmm.fit(X_train_pca)
        aic_scores[cv].append(gmm.aic(X_val_pca))
        bic_scores[cv].append(gmm.bic(X_val_pca))

# Plot BIC and AIC
plt.figure(figsize=(14, 6))
for cv in cov_types:
    plt.plot(list(k_range), bic_scores[cv], marker='o', label=f"BIC - {cv}")
plt.yscale('log')
plt.xlabel("Number of clusters (k)")
plt.ylabel("BIC Score (lower is better)")
plt.title("GMM Model Selection by BIC on Validation Set")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
for cv in cov_types:
    plt.plot(list(k_range), aic_scores[cv], marker='o', label=f"AIC - {cv}")
plt.yscale('log')
plt.xlabel("Number of clusters (k)")
plt.ylabel("AIC Score (lower is better)")
plt.title("GMM Model Selection by AIC on Validation Set")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Find best model
best_cv, best_k, best_score = None, None, float('inf')
for cv in cov_types:
    for k, score in zip(k_range, bic_scores[cv]):
        if score < best_score:
            best_score = score
            best_cv = cv
            best_k = k

print(f"[GMM] Best by BIC -> covariance_type={best_cv}, n_components={best_k}, BIC={best_score:.2f}\n")

# Fit best GMM
X_trainval_pca = np.vstack([X_train_pca, X_val_pca])
best_gmm = GaussianMixture(n_components=best_k, covariance_type=best_cv, random_state=42)
best_gmm.fit(X_trainval_pca)

train_labels = best_gmm.predict(X_trainval_pca)
train_probs = best_gmm.predict_proba(X_trainval_pca)

# ----------------------------- 7. Hard & Soft Cluster Assignments -----------------------------
print("\n====================== Step 7: Hard & Soft Cluster Assignments ======================\n")

hard_assignments = best_gmm.predict(X_test_pca)
soft_assignments = best_gmm.predict_proba(X_test_pca)

print("=== Hard Cluster Assignments (first 10 samples) ===")
print(hard_assignments[:10])
print("\n=== Soft Cluster Assignments (first 10 samples) ===")
print(soft_assignments[:10])

# Save results
df_hard = pd.DataFrame({"Image_Index": np.arange(len(hard_assignments)), "Hard_Cluster": hard_assignments})
df_soft = pd.DataFrame(soft_assignments, columns=[f"Cluster_{i}" for i in range(best_k)])
df_soft.insert(0, "Image_Index", np.arange(len(soft_assignments)))
df_hard.to_csv("GMM_Hard_Assignments.csv", index=False)
df_soft.to_csv("GMM_Soft_Assignments.csv", index=False)
print("\n[INFO] Saved hard & soft clustering results to CSV files.\n")

# ----------------------------- 8. Generate Synthetic Faces Using GMM -----------------------------
print("\n====================== Step 8: Generate Synthetic Faces ======================\n")

num_new_faces = 10
new_faces_pca, new_labels = best_gmm.sample(num_new_faces)
new_faces = pca.inverse_transform(new_faces_pca).reshape(-1, 64, 64)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Synthetic Faces Generated by GMM", fontsize=16)
for i, ax in enumerate(axes.flat):
    ax.imshow(new_faces[i], cmap='gray')
    ax.set_title(f"Sample {i + 1}")
    ax.axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

for i, face in enumerate(new_faces):
    imageio.imwrite(f"Synthetic_Face_{i + 1}.png", (face * 255).astype(np.uint8))
print("[INFO] Synthetic faces saved as 'synthetic_face_X.png'\n")

# ----------------------------- 9. Create Anomalous Images -----------------------------
print("\n====================== Step 9: Apply Transformations to Create Anomalies ======================\n")

sample_faces = X_test[:5]
rotated_faces = [transform.rotate(face, angle=30, mode='wrap') for face in sample_faces]
flipped_faces = [np.fliplr(face) for face in sample_faces]
darkened_faces = [exposure.adjust_gamma(face, gamma=2) for face in sample_faces]
anomalous_faces = rotated_faces + flipped_faces + darkened_faces

# Plot anomalies
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle("Transformed / Anomalous Faces", fontsize=16)
titles = ["Rotated (30°)", "Flipped", "Darkened"]
for row in range(3):
    for col in range(5):
        idx = row * 5 + col
        axes[row, col].imshow(anomalous_faces[idx], cmap='gray')
        axes[row, col].set_title(titles[row] if col == 0 else "")
        axes[row, col].axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Predict anomalies
anomalous_flat = np.array([f.reshape(-1) for f in anomalous_faces])
anomalous_pca = pca.transform(anomalous_flat)
anom_labels = best_gmm.predict(anomalous_pca)
anom_probs = best_gmm.predict_proba(anomalous_pca)

print("=== GMM Cluster Predictions for Anomalous Images ===")
print(anom_labels)
print("\n=== Mean Soft Probabilities (uncertainty indicator) ===")
print(np.mean(np.max(anom_probs, axis=1)))

# ----------------------------- 10. Anomaly Detection Using Log-Likelihood -----------------------------
print("\n====================== Step 10: Detect Anomalies Using Log-Likelihood ======================\n")

normal_scores = best_gmm.score_samples(X_test_pca)
anom_scores = best_gmm.score_samples(anomalous_pca)

print("=== Average Log-Likelihood ===")
print(f"Normal images:   {np.mean(normal_scores):.2f}")
print(f"Anomalous images:{np.mean(anom_scores):.2f}\n")

# Plot histogram comparison
plt.figure(figsize=(8, 5))
plt.hist(normal_scores, bins=20, alpha=0.7, label="Normal Faces", color="green")
plt.hist(anom_scores, bins=20, alpha=0.7, label="Anomalous Faces", color="red")
plt.title("Comparison of Log-Likelihood Scores")
plt.xlabel("Log-Likelihood (higher = more normal)")
plt.ylabel("Number of Images")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()