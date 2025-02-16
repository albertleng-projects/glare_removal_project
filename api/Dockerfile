# Use the official Python image from Docker Hub
FROM python:3.12.6-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory content into the container at /app
COPY . /app

# Copy the models folder from the parent directory into the container
COPY ../models /app/models

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 4000 for the Flask API
EXPOSE 4000

# Set the command to run the Flask app
CMD ["python", "app.py"]
