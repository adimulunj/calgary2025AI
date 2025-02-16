import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Define paths
training_images_path = "./training_data"
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
VALIDATION_SPLIT = 0.1  # 10% validation split

# Function to load data using tf.data API
def load_data():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        training_images_path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=42
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        training_images_path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=42
    )

    # Get class names automatically
    class_names = train_dataset.class_names
    print("Detected classes:", class_names)
    return train_dataset, val_dataset, class_names

# Normalize images with a closure that captures num_classes
def normalize(num_classes):
    def norm(image, label):
        image = tf.cast(image, tf.float32) / 255.0  # Scale pixels to [0,1]
        label = tf.one_hot(label, depth=num_classes)  # One-hot encoding of labels
        return image, label
    return norm

# Data augmentation function
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    return image, label


# Create the CNN model
def create_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
    )
    base_model.trainable = False  # Freeze pretrained layers

    # Add custom classification head
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def main():
    # Load datasets and infer class names
    train_data, val_data, class_names = load_data()
    num_classes = len(class_names)

    # Apply augmentation only on training data
    train_data = train_data.map(augment)

    # Normalize both datasets using the closure with num_classes
    norm_fn = normalize(num_classes)
    train_data = train_data.map(norm_fn)
    val_data = val_data.map(norm_fn)

    # Optimize the input pipeline
    train_data = train_data.shuffle(1000).cache().prefetch(tf.data.AUTOTUNE)
    val_data = val_data.cache().prefetch(tf.data.AUTOTUNE)

    # Create and compile the model
    model = create_model(num_classes)
    initial_learning_rate = 0.001

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,  
        decay_rate=0.9,    # Reduce LR by 10% every 1000 steps
        staircase=True     # Apply in discrete steps
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    # Evaluate model on validation data
    val_loss, val_acc = model.evaluate(val_data)
    print(f"Validation Accuracy: {val_acc:.2f}")

    # Save the model
    os.makedirs("./models", exist_ok=True)
    model.save('./models/cnn_model(0).h5')

    # Plot training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
