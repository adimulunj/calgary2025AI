import tensorflow as tf
import numpy as np
import cv2
import os

# Constants
MODEL_PATH = "./models/cnn_model(4).h5"
IMAGE_SIZE = (128, 128)  # Match your training size

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Get class names (from training dataset)
CLASS_NAMES = sorted(os.listdir("./training_data"))  # Ensure same order as training

def preprocess_image(image_path):
    """Loads and preprocesses an image for the model."""
    if not os.path.exists(image_path):  # Check if the file exists
        print(f"Error: File '{image_path}' not found.")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load '{image_path}'. Check the format and path.")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, IMAGE_SIZE)  # Resize to model's input size
    image = image.astype("float32") / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def predict(image_path, top_k=5):
    """Runs inference and prints the top K predicted classes with confidence scores."""
    image = preprocess_image(image_path)

    if image is None:
        print("Skipping prediction due to image loading error.")
        return

    predictions = model.predict(image)[0]  # Get predictions for the single image

    # Get the indices sorted by confidence (highest first)
    sorted_indices = np.argsort(predictions)[::-1]

    print("\nPredictions (Sorted by Confidence):")
    for i in range(min(top_k, len(CLASS_NAMES))):
        class_index = sorted_indices[i]
        confidence = predictions[class_index] * 100
        predicted_class = CLASS_NAMES[class_index]
        print(f"{i+1}. {predicted_class} - {confidence:.2f}%")


# Example Usage
if __name__ == "__main__":
    test_image = "D:\Programming\Hackathon\calgary2025\AI\image.png"  # Change this to your image path
    predict(test_image)
