# Glare Removal API Service

This project provides a **RESTful API** for glare removal from images using a trained **UNet-GAN model**.  
The API is built with **Flask** and can be **containerized using Docker** for easy deployment.

---

## 📌 Features

- ✅ Removes glare from images using a **trained UNet-GAN model**
- ✅ Accepts **PNG (.png) images**
- ✅ Converts images to **grayscale before processing**
- ✅ Provides a REST API with **Flask**
- ✅ Deployable using **Docker**

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/albertleng-projects/glare_removal_project
cd glare_removal_project/api
``` 

### 2. Build the Docker Image

To build the Docker image, make sure you're in the project directory (same directory as the Dockerfile) and run:

```bash
docker build -t glare-removal-api .
```

### 3. Run the Docker Container

After building the image, run the Docker container with the following command:

```bash
docker run -p 4000:4000 glare-removal-api
```

### 4. Test the API

The API should now be running on `http://localhost:4000`. You can test the API using a tool like Postman or cURL.
Here are some sample cURL commands to test the API:

#### a. GET `/ping` endpoint

```bash
curl http://localhost:4000/ping
```

You should get a response like this:

```json
{
  "message": "pong"
}
```

#### b. POST `/infer` endpoint

1. Prepare an image that you want to enhance.
2. Use curl:

```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:4000/infer
```

Make sure to replace `/path/to/your/image.jpg` with the actual path to the image you want to enhance.

Response:

```bash
{
    "image": "<binary_value_of_the_image>"
}
```

#### c. GET `/list_files` endpoint

List the enhanced images in the server. Currently, the server only stores the last enhanced image.

```bash
curl http://localhost:4000/list_files
```

Response:

```bash
{
  "files": [
    {
      "filename": "enhanced_image.png",
      "url": "/static/enhanced_image.png"
    }
  ]
}
```

#### d. GET `/static/<filename>` endpoint

Retrieve the enhanced image from the server. Currently, the server only stores the last enhanced image. Hence, you can
access the enhanced image using the following URL:

```bash
http://localhost:4000/static/enhanced_image.png
``` 

The response will be the enhanced image.

### 🛠 Running Without Docker

If you want to run the app without Docker, follow these steps:

#### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2️⃣ Run the Flask App

```bash
python app.py
```

#### 3️⃣ Test the API

The API will be available at: 📌 http://127.0.0.1:4000

### 📝 Notes & Troubleshooting

#### 💡 1. Model Not Found Error

If you see this error:

```bash
❌ Model file not found at: models/final_model_epoch_5.pth
```

**Solution**: Make sure the model file exists in `models/` directory. If it's missing, place the correct
`final_model_epoch_5.pth` file in:

```bash
glare_removal_project/models/final_model_epoch_5.pth
```

#### 💡 2. Container Already Running

If you see this error:

```bash
docker: Error response from daemon: Conflict. The container name "/glare-removal-api" is already in use.
```

**Solution**: Stop and remove the existing container before restarting:

```bash
docker rm -f glare-removal-api
docker run -p 4000:4000 glare-removal-api
```

#### 💡 3. Flask App Not Starting

If the app **fails to start**, try running it manually and check for missing dependencies:

```bash
python app.py
```

Ensure all dependencies are installed using:

```bash
pip install -r requirements.txt
```