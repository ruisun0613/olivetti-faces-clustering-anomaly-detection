from skimage import transform, exposure

import numpy as np
import matplotlib.pyplot as plt

# Create Anomalous Images
def generate_anomalous_images(X_test,pca,best_gmm):
    sample_faces = X_test[:5]
    rotated_faces = [transform.rotate(face, angle = 30, mode = 'wrap') for face in sample_faces]
    flipped_faces = [np.fliplr(face) for face in sample_faces]
    darkened_faces = [exposure.adjust_gamma(face, gamma = 2) for face in sample_faces]
    anomalous_faces = rotated_faces + flipped_faces + darkened_faces

    # Plot anomalies
    fig, axes = plt.subplots(3, 5, figsize = (15, 9))
    fig.suptitle("Transformed / Anomalous Faces", fontsize = 16)
    titles = ["Rotated(30°)", "Flipped", "Darkened"]

    for row in range(3):
        for col in range(5):
            idx = row * 5 + col
            axes[row, col].imshow(anomalous_faces[idx], cmap = "gray")
            axes[row, col].set_title(titles[row] if col == 0 else "")
            axes[row, col].axis('off')
    plt.tight_layout(rect = [0, 0, 1, 0.95])
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

    return anomalous_pca

# Anomaly Detection Using Log-Likelihood
def anomaly_detection(best_gmm, X_test_pca, anomalous_pca):
    normal_scores = best_gmm.score_samples(X_test_pca)
    anom_scores = best_gmm.score_samples(anomalous_pca)

    print("=== Average Log-Likelihood ===")
    print(f"Normal images: {np.mean(normal_scores):.2f}")
    print(f"Anomalous images: {np.mean(anom_scores):.2f}\n")

    # Plot histogram comparison
    plt.figure(figsize = (8,5))
    plt.hist(normal_scores, bins = 20, alpha = 0.7, label = "Normal Faces", color = "green")
    plt.hist(anom_scores, bins = 20, alpha = 0.7, label = "Anomalous Faces", color = "red")
    plt.title("Comparison of Log-Likelihood Scores")
    plt.xlabel("Log-Likelihood (higher = more normal)")
    plt.ylabel("Number of Images")
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.show()