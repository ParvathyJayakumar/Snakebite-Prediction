import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0  # EfficientNetB0 is a smaller variant

def build_model():
    # Load EfficientNet with pre-trained ImageNet weights, exclude top layers (classification layers)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model to retain pre-trained weights
    base_model.trainable = False
    
    # Build a custom model on top of EfficientNet
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary output: Venomous vs Non-Venomous
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
