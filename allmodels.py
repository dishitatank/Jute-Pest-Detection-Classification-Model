import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import (
    MobileNet, DenseNet121, InceptionV3, ResNet50, EfficientNetB0, VGG16
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

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
datagen_params = dict(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
train_datagen = ImageDataGenerator(**datagen_params)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical'
)

# Function to load pre-trained models
def load_model(model_name):
    if model_name == "MobileNet":
        base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    elif model_name == "DenseNet":
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    elif model_name == "GoogleNet":
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    elif model_name == "ResNet":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    elif model_name == "EfficientNet":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    elif model_name == "VGGNet":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    else:  # Custom CNN
        return create_custom_cnn()

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)

# Custom CNN
def create_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model

# Training and Evaluation Pipeline
def train_and_evaluate(model_name):
    print(f"\n--- Training {model_name} ---")
    model = load_model(model_name)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=val_generator.samples // batch_size
    )
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"{model_name} Test Accuracy: {test_accuracy:.2f}\n")
    return history, test_accuracy

# Models to train
models = ["MobileNet", "DenseNet", "GoogleNet", "ResNet", "EfficientNet", "VGGNet", "CustomCNN"]
results = {}

for model_name in models:
    history, test_accuracy = train_and_evaluate(model_name)
    results[model_name] = test_accuracy

print("\n--- Final Test Accuracies ---")
for model, acc in results.items():
    print(f"{model}: {acc:.2f}")