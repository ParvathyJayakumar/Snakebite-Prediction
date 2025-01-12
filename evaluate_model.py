from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_data import val_generator  # Ensure your validation data generator is correctly imported

# Load the trained model
model = load_model('best_model.keras')  # Load the best model saved during training

# Evaluate the model on the validation data
loss, accuracy = model.evaluate(val_generator, steps=val_generator.samples // val_generator.batch_size)

# Print the evaluation results
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# If you want to get the predicted labels and true labels for further analysis
y_true = []
y_pred = []

# Use an iterator to loop over the generator
for i, (x_batch, y_batch) in enumerate(val_generator):
    print(f"Processing batch {i + 1}")
    
    # Check if the batch is empty
    if len(x_batch) == 0 or len(y_batch) == 0:
        print(f"Warning: Empty batch at index {i}")
        continue  # Skip this batch if it's empty

    # True labels (integer form)
    y_true.extend(np.argmax(y_batch, axis=1))  # Get the true labels as integers
    
    # Predictions (probabilities)
    y_pred_batch = model.predict(x_batch)

    # Check if predictions were made
    if len(y_pred_batch) == 0:
        print(f"Warning: Model did not generate predictions for batch {i}")
        continue  # Skip this batch if no predictions are made

    # Predicted labels (integer form)
    y_pred.extend(np.argmax(y_pred_batch, axis=1))  # Get the predicted labels as integers

# Ensure both y_true and y_pred have the same length
if len(y_true) != len(y_pred):
    raise ValueError(f"Mismatch in length of true labels and predicted labels: {len(y_true)} != {len(y_pred)}")

# Make sure y_true and y_pred are not empty
if len(y_true) == 0 or len(y_pred) == 0:
    raise ValueError("y_true or y_pred is empty!")

# Classification report (precision, recall, F1-score)
print("\nClassification Report:")
target_names = list(val_generator.class_indices.keys())  # Get class labels from the generator
print(classification_report(y_true, y_pred, target_names=target_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices, yticklabels=val_generator.class_indices)
plt.title('Confusion Matrix')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
