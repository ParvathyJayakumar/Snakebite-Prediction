from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from buildmodel import build_model
from prepare_data import train_generator, val_generator

# Build the model
model = build_model()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Checkpoint to save the best model based on validation loss
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Adjust based on your dataset
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping, checkpoint]  # Add both early stopping and checkpoint callbacks
)

# Save the final model after training
model.save('model.keras')  # Save the entire model (including architecture and weights)
