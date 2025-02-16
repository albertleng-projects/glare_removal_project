import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import base64
import logging
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from flask import Flask, request, jsonify
from models.autoencoder.auto import Auto
import torchvision.transforms as transforms
from pathlib import Path

# Add the parent directory (glare_removal_project) to the Python path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load your model here
MODEL_PATH = Path("../models/final_glare_removal_autoencoder.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize the model
model = Auto().to(device)

# Load the trained model
if os.path.exists(MODEL_PATH):
    logger.info("Model file exists!")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Set the model to evaluation mode
    logger.info("✅ Model loaded successfully.")
else:
    logger.error("Model file not found at: %s", MODEL_PATH)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512
    transforms.ToTensor(),  # Convert image to tensor
])


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})


@app.route('/infer', methods=['POST'])
def infer():
    # Check if the 'image' is part of the request
    if 'image' not in request.files:
        logger.error("No image file provided")
        return jsonify({"error": "No image file provided"}), 400

    # Get the uploaded image from the request
    file = request.files['image']

    try:
        # Open the image using PIL and convert to RGB (to handle both grayscale and colored inputs)
        img = Image.open(file.stream).convert("RGB")
        logger.info("Image successfully loaded for inference.")
    except Exception as e:
        logger.error("Error processing the image: %s", e)
        return jsonify({"error": f"Error processing the image: {e}"}), 400

    # Preprocess the image to a tensor
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Ensure the image is on the correct device (GPU or CPU)
    img_tensor = img_tensor.to(device)
    model.to(device)

    # Run inference
    with torch.no_grad():
        enhanced_img_tensor = model(img_tensor)

    # Post-process the output tensor
    enhanced_img = enhanced_img_tensor.squeeze(0).cpu().numpy()  # Remove batch dimension

    # Ensure the image is in the expected dimensions (512x512)
    logger.info("Enhanced image shape before processing: %s", enhanced_img.shape)

    if enhanced_img.shape[0] == 3:  # If it's RGB (3 channels)
        enhanced_img = np.transpose(enhanced_img, (1, 2, 0))  # Change to HxWxC format (512, 512, 3)
    elif enhanced_img.shape[0] == 1:  # If grayscale (1 channel)
        enhanced_img = enhanced_img.squeeze(axis=0)  # Remove channel dimension, (512, 512)

    # Ensure the values are in the range [0, 255] for uint8 format
    enhanced_img = np.clip(enhanced_img * 255, 0, 255).astype('uint8')

    # Convert the output array to a PIL image
    enhanced_img_pil = Image.fromarray(enhanced_img)

    # Convert the enhanced image to bytes
    img_byte_arr = BytesIO()
    enhanced_img_pil.save(img_byte_arr, format='PNG')

    # Encode the image to base64 to send it as JSON
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Save the enhanced image locally (optional)
    save_image_locally(img_base64)

    # Send back the base64-encoded image in a JSON response
    return jsonify({"image": img_base64})


def save_image_locally(image_base64):
    try:
        # Decode the base64 string to binary data
        img_data = base64.b64decode(image_base64)

        # Load the binary data as an image
        img = Image.open(BytesIO(img_data))

        # Save the image locally
        img.save('enhanced_image.png')
        logger.info("Image saved locally as 'enhanced_image.png'")
    except Exception as e:
        logger.error("Error saving the image locally: %s", e)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000)
