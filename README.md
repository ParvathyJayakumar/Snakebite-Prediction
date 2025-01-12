# Snakebite Prediction Using Deep Learning  

## Project Overview  
This project leverages deep learning to predict and classify snakebite cases using image data. Snakebites pose a significant medical challenge, especially in rural and under-resourced areas. Early detection and classification of snakebite wounds or the responsible snake species can play a vital role in delivering timely and appropriate treatment, potentially saving lives.  

The model employs a **Convolutional Neural Network (CNN)** architecture, optimized for high accuracy in image classification tasks. The system processes image datasets to classify inputs into two categories: **snakebite detected** or **no snakebite detected**, enabling healthcare professionals or non-experts to utilize this tool effectively.  

---

## Key Features  
- **Deep Learning with Transfer Learning**: Leverages pretrained **EfficientNetB0** for robust feature extraction and binary classification.  
- **Image Classification**: Processes input images to classify them as venomous or non-venomous snakebites.  
- **Model Optimization**: Includes dropout, data augmentation, and learning rate adjustments to improve model generalization and prevent overfitting.  
- **Evaluation Metrics**: Generates confusion matrices, classification reports, and visualizations to assess performance.  
- **Automation Potential**: The model can serve as a foundation for mobile or web applications for wider accessibility.  

---

## Model Details  

### Base Model Architecture  
The model architecture incorporates:  
1. **Transfer Learning**:  
   - **Base Model**: Pretrained **EfficientNetB0** with frozen layers for initial training.  
   - **Fine-tuning**: Additional training on specific layers for improved accuracy on the snakebite dataset.  

2. **Custom Layers**:  
   - Global Average Pooling for feature reduction.  
   - Fully Connected Dense Layers for classification.  
   - Dropout Layers to prevent overfitting.  

3. **Output Layer**:  
   - Activation: Sigmoid for binary classification.  

### Configuration  
- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam with an adaptive learning rate.  
- **Metrics**: Accuracy, Precision, Recall, and F1 Score.  

---

## Technologies Used  
- **Deep Learning Framework**: TensorFlow/Keras  
- **Programming Language**: Python 3.11  
- **Visualization**: Matplotlib, Seaborn  

---

## Installation  

### Prerequisites  
1. Python 3.8+  
2. TensorFlow 2.x and additional libraries (see `requirements.txt`).  

### Steps  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/yourusername/snakebite-prediction.git  
   cd snakebite-prediction  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

---

## Usage  

1. **Data Preparation**:  
   Organize your dataset with two categories: `venomous` and `non-venomous`. Update the paths in `train_model.py` to point to your dataset directory.  

2. **Training the Model**:  
   Run the training script to train the model on your dataset:  
   ```bash  
   python train_model.py  
   ```  

3. **Evaluation**:  
   Evaluate the trained model using the test dataset:  
   ```bash  
   python evaluate_model.py  
   ```  

4. **Prediction**:  
   Use the saved model for predicting new images:  
   ```bash  
   python predict.py --image_path /path/to/image.jpg  
   ```  
