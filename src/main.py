from data import load_and_visualize_data, train_validation_test
from pca import apply_pca

# Run
if __name__ == "__main__":
    # Load data
    X,y = load_and_visualize_data()
    
    # Split train / validation / test sets
    data = train_validation_test(X,y)

    # PCA
    apply_pca(data)
