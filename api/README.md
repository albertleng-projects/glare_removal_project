# Glare Removal API Service

This project provides an API for glare removal from images using a trained autoencoder model. The API is built with
Flask and exposed as a REST service. The application can be containerized using Docker for easy deployment.

## Steps to Build and Run the Docker Container

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


