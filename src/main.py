from data import load_and_visualize_data, train_validation_test
from pca import apply_pca
from clustering import hierarchical_clustering, visualize_clusters
from gmm import gaussian_mixture_model, generate_synthetic_faces, hard_soft_cluster
from anomaly import generate_anomalous_images, anomaly_detection

def main():
    # Load data
    X,y = load_and_visualize_data()
    
    # Split train / validation / test sets
    data = train_validation_test(X,y)

    # PCA
    pca, X_train_pca, X_val_pca, X_test_pca = apply_pca(data)

    # Clustering
    linked = hierarchical_clustering(X_train_pca)

    # visualize clusters
    visualize_clusters(X_train = data["X_train"], linked = linked)

    # GMM
    best_gmm, best_k = gaussian_mixture_model(X_train_pca, X_val_pca)

    # Hard / Soft Cluster
    hard_soft_cluster(best_gmm, best_k, X_test_pca)

    # Generate Synthetic Faces Using GMM
    generate_synthetic_faces(best_gmm, pca)

    # Generate Anomalous Images
    anomalous_pca = generate_anomalous_images(X_test = data["X_test"], pca = pca, best_gmm = best_gmm)

    # Anomaly Detection
    anomaly_detection(best_gmm, X_test_pca, anomalous_pca)

# Run
if __name__ == "__main__":
    main()