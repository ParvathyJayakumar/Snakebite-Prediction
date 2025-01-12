import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Function to evaluate the model
def evaluate_model(model_path, test_data_dir, batch_size=32, img_size=(224, 224)):
    # Load the trained model
    model = load_model(model_path)

    # Prepare test data using ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # Evaluate the model on the test data
    print("Evaluating the model...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Predict the class labels for the test data
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.round(predictions).astype(int).flatten()

    # Get the true labels
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Generate confusion matrix and classification report
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_labels)

if __name__ == '__main__':
    # Hardcoded values for the model evaluation
    model_path="./best_model.keras"   # Replace with your model path
    test_data_dir = "./dataset/test"       # Replace with your test data directory
    batch_size = 32                            # You can modify this if needed
    img_size = (224, 224)                      # Modify if your images need different size

    # Call the evaluation function
    evaluate_model(model_path, test_data_dir, batch_size, img_size)
