import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# random seed
RANDOM_SEED = 42

# Load and Visualize Dataset
def load_and_visualize_data():
    data = fetch_olivetti_faces(shuffle = True)

    X = data.images
    y = data.target

    print("Dataset shapes:")
    print(f"X (images): {X.shape}")
    print(f"y (target): {y.shape}")

    # Display a few sample images
    fig, axes = plt.subplots(2, 5, figsize = (10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i], cmap = "gray")
        ax.set_title(f"Label: {y[i]}")
        ax.axis('off')
    plt.suptitle("Sample Olivetti Faces:", fontsize = 16)
    plt.show()

    return X,y
# Split Dataset into Train / Validation / Test
def train_validation_test(X,y):
    # Train / Validation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size = 0.3, stratify = y, random_state = RANDOM_SEED
    ) 

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size = 0.5, stratify = y_temp, random_state = RANDOM_SEED
    )

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}\n")

    # Validation label distribution
    fig, axes = plt.subplots(1, 3, figsize = (20, 8))
    sns.countplot(x = y_train, ax = axes[0], color = "steelblue")
    axes[0].set_title("Training Set Distribution")

    sns.countplot(x = y_val, ax = axes[1], color = "darkorange")
    axes[1].set_title("Validation Set Distribution")

    sns.countplot(x = y_test, ax = axes[2], color = "seagreen")
    axes[2].set_title("Test Set Distribution")

    plt.suptitle("Class Distribution Across Train/Validation/Test Sets", fontsize = 16)
    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.show()

    return {
        "X_train" : X_train,
        "X_val" : X_val,
        "X_test" : X_test,
        "y_train" : y_train,
        "y_val" : y_val,
        "y_test" : y_test
    }

# Test
# if __name__ == "__main__":
#     X, y = load_and_visualize_data()
#     data = train_validation_test(X, y)