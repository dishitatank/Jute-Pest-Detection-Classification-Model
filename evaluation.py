import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
train_dir = "/Users/dishita.tank/Desktop/Jute_Pest_Dataset/train"
val_dir = "/Users/dishita.tank/Desktop/Jute_Pest_Dataset/val"
test_dir = "/Users/dishita.tank/Desktop/Jute_Pest_Dataset/test"

# Hyperparameters
batch_size = 32
image_size = (150, 150)
learning_rate = 0.0001
epochs = 10
num_classes = 17  # Based on your dataset

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pooling layer
x = Dense(1024, activation='relu')(x)  # Fully connected layer
predictions = Dense(num_classes, activation='softmax')(x)  # Output layer

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size
)

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important to keep order for correct confusion matrix calculation
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Get the true labels and predictions for confusion matrix
test_labels = test_generator.classes
test_predictions = model.predict(test_generator)
test_predictions = np.argmax(test_predictions, axis=1)

# Confusion Matrix and Classification Report for plotting
cm = confusion_matrix(test_labels, test_predictions)
cr = classification_report(test_labels, test_predictions, target_names=test_generator.class_indices.keys())

# Print Classification Report (Precision, Recall, F1-Score)
print("Classification Report:")
print(cr)

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))  # Adjust figure size as needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 8}, 
            xticklabels=test_generator.class_indices.keys(), 
            yticklabels=test_generator.class_indices.keys())  # Adjust annotation font size
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()