import os
import base64
import logging
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torchvision.transforms as transforms

# ===========================
# 🔹 Set Up Logging
# ===========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ===========================
# 🔹 Define Image Save Path
# ===========================
STATIC_FOLDER = os.path.join(os.getcwd(), 'static')
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# ===========================
# 2️⃣ Define Generator (U-Net)
# ===========================
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ===========================
# 🔹 Load Model Checkpoint
# ===========================
MODEL_PATH = Path("../models/checkpoints/final_model_epoch_5.pth") if Path(
    "../models/checkpoints/final_model_epoch_5.pth").exists() else Path("./models/final_model_epoch_5.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize U-Net Generator model
model = UNetGenerator().to(device)

if MODEL_PATH.exists():
    logger.info("✅ Model file exists! Loading checkpoint...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Set to evaluation mode
    logger.info("✅ Model loaded successfully.")
else:
    logger.error("❌ Model file not found at: %s", MODEL_PATH)

# ===========================
# 🔹 Define Image Transformations (Grayscale)
# ===========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to Grayscale
    transforms.Resize((512, 512)),  # Resize to 512x512
    transforms.ToTensor(),  # Convert image to tensor
])

# ===========================
# 🔹 API Endpoints
# ===========================

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})

@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        logger.error("❌ No image file provided")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert("L")  # Convert to grayscale
        logger.info("✅ Image successfully loaded for inference.")
    except Exception as e:
        logger.error("❌ Error processing the image: %s", e)
        return jsonify({"error": f"Error processing the image: {e}"}), 400

    # Preprocess image
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)

    # Run inference
    with torch.no_grad():
        enhanced_img_tensor = model(img_tensor)

    # Post-process the output tensor
    enhanced_img = enhanced_img_tensor.squeeze(0).cpu().numpy()
    enhanced_img = enhanced_img.squeeze(axis=0)  # Remove the channel dimension

    enhanced_img = np.clip(enhanced_img * 255, 0, 255).astype('uint8')
    enhanced_img_pil = Image.fromarray(enhanced_img, mode="L")  # Convert back to grayscale image

    # Convert to base64
    img_byte_arr = BytesIO()
    enhanced_img_pil.save(img_byte_arr, format='PNG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Save locally
    save_image_locally(img_base64)

    return jsonify({"image": img_base64})

def save_image_locally(image_base64):
    try:
        img_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_data))
        img.save(os.path.join(STATIC_FOLDER, 'enhanced_image.png'))
        logger.info("✅ Image saved locally as 'enhanced_image.png'")
    except Exception as e:
        logger.error("❌ Error saving the image locally: %s", e)

@app.route('/static/<filename>')
def serve_static_file(filename):
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/list_files', methods=['GET'])
def list_files():
    files = [{"filename": file, "url": f"/static/{file}"} for file in os.listdir(STATIC_FOLDER)]
    return jsonify({"files": files})

# ===========================
# 🔹 Run Flask App
# ===========================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000)
