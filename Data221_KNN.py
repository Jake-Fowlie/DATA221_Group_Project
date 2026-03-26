import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Function to load and preprocess images
def load_images_from_folder(folder_path, label_map, image_size=128):
    features = []  # Will store image data (X)
    labels = []    # Will store corresponding labels (y)

    # Loop through each class folder (glioma, meningioma, etc.)
    for class_name, class_label in label_map.items():
        class_folder = os.path.join(folder_path, class_name)

        # Loop through each image in the class folder
        for file_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, file_name)

            # Read image in grayscale (reduces complexity)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                # Resize image to a fixed size (128x128)
                image = cv2.resize(image, (image_size, image_size))

                # Normalize pixel values (0–255 → 0–1)
                image = image / 255.0

                # Flatten image into 1D vector for KNN
                image = image.flatten()

                # Store processed image and its label
                features.append(image)
                labels.append(class_label)

    # Convert lists to numpy arrays for model compatibility
    return np.array(features), np.array(labels)


# Main function
def main():

    # Mapping class names to numerical labels
    label_map = {
        "glioma": 0,
        "meningioma": 1,
        "pituitary": 2,
        "notumor": 3
    }

    # Used for readable output in classification report
    class_names = ["glioma", "meningioma", "pituitary", "notumor"]

    # Get absolute path of current file (avoids path errors)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define training and testing paths
    train_path = os.path.join(base_dir, "Data", "Training")
    test_path = os.path.join(base_dir, "Data", "Testing")

    # Debugging prints to verify paths
    print("Current working directory:", os.getcwd())
    print("Train path:", train_path)
    print("Test path:", test_path)
    print("Training exists:", os.path.exists(train_path))
    print("Testing exists:", os.path.exists(test_path))

    # Load and preprocess training data
    print("Loading training data...")
    X_train, y_train = load_images_from_folder(train_path, label_map)

    # Load and preprocess testing data
    print("Loading testing data...")
    X_test, y_test = load_images_from_folder(test_path, label_map)

    # Print dataset shapes
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # KNN Model

    # Initialize KNN with k = 5 neighbors
    knn_model = KNeighborsClassifier(n_neighbors=5)

    # Train model on training data
    knn_model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = knn_model.predict(X_test)


    # Evaluation Metrics
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Results Output
    print("\n=== KNN Results ===")
    print("Accuracy:", accuracy)
    print("Macro Precision:", precision)
    print("Macro Recall:", recall)
    print("Macro F1 Score:", f1)

    # Display confusion matrix
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Detailed classification report per class
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))


# Run the program
if __name__ == "__main__":
    main()