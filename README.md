# Glare Removal Project

This repository contains an implementation of a **glare removal model** using an **autoencoder**. The project is
structured into two main components:

1. **`api/`** – A Flask-based API for model inference.
2. **`notebooks/`** – Jupyter notebooks for data preprocessing, model training, evaluation, and inference.

The model is trained using **an autoencoder** located in `models/autoencoder/`. The **goal** is to process images
affected by glare and generate an enhanced, glare-free version.

---

## 📂 Repository Structure

```graphql
glare_removal_project/
│── api/                # Flask API for model inference
│── data/               # Data used for training (ignored in Git)
│── models/             # Trained models & autoencoder architecture
│── notebooks/          # Jupyter notebooks for training and evaluation
│── Dockerfile          # Containerization setup for the API
│── README.md           # Documentation
│── requirements.txt    # Python dependencies
└── .gitignore          # Files to be ignored in version control
```

---

## 🚀 How to Use This Repository

### 1️⃣ Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/albertleng-projects/glare_removal_project.git
cd glare_removal_project
```

### 2️⃣ Set Up a Virtual Environment

It’s recommended to use a virtual environment to manage dependencies.

For Windows (PowerShell):

```bash
python -m venv venv
venv\Scripts\activate
```

For Mac/Linux:

```bash 
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

Once inside the virtual environment, install the required packages:

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Jupyter Notebooks (For Training & Evaluation)

If you want to explore the data preprocessing, model training, and evaluation, start Jupyter Notebook:

```bash
jupyter notebook
```

## 📝 Notebooks Overview

### 1️⃣ **Data Preprocessing ([01_data_preprocessing.ipynb](notebooks/01_data_preprocessing.ipynb))**

- Loads the dataset from `data/` (not included in the repo due to large size).
- Applies transformations such as resizing, normalization, and augmentation.
- Splits data into **training** and **validation** sets.

### 2️⃣ **Model Training ([02_model_training.ipynb](notebooks/02_model_training.ipynb))**

- Defines the **autoencoder architecture** ([models/autoencoder/auto.py](models/autoencoder/auto.py)).
- Trains the model using **L1 loss** (Mean Absolute Error).
- Uses **Adam optimizer** for weight updates.
- Monitors **training loss** and **validation loss** per epoch.

### 3️⃣ **Model Evaluation ([03_model_evaluation.ipynb](notebooks/03_model_evaluation.ipynb))**

- Loads the trained model.
- Computes evaluation metrics (**L1 loss** on test images).
- Displays **sample predictions** (before and after glare removal).
- Identifies potential areas for improvement.

### 4️⃣ **Model Inference ([04_inference.ipynb](notebooks/04_inference.ipynb))**

- Loads a saved model ([models/final_glare_removal_autoencoder.pth](models/final_glare_removal_autoencoder.pth)).
- Runs inference on new input images.
- Saves and displays the enhanced (glare-free) images.

---

## 🏗️ Model Details

- The project uses a **convolutional autoencoder** for glare removal.
- The encoder extracts features from the input image.
- The decoder reconstructs the glare-free image.
- The model is trained using an **L1 loss function**, as it is well-suited for image restoration tasks.

**Training Results:**

- The final **training loss** and **validation loss** per epoch are shown in **Results**
  of [03_model_evaluation.ipynb](notebooks/03_model_evaluation.ipynb).
- Lower **L1 loss** indicates better performance.
- The **validation loss** achieved is **0.1388**, which does not meet the target of **0.06** due to constraints in time
  and resources. That would be addressed in future enhancements.

---

## 🚀 Using the API

The `api/` folder contains a Flask-based API to serve the trained model.

See the [api/README.md](./api/README.md) for detailed instructions on setting up and running the API.

## 🔮 Future Enhancements

### 1. Improve Model Performance

- Experiment with more **complex architectures** (e.g., UNet, GANs).
- Tune **hyperparameters** (learning rate, batch size, etc.).

### 2. Better Training Dataset

- Collect more **diverse glare-affected images**.
- Augment dataset with **synthetic glare effects**.

### 3. Optimize API Performance and Use GPU

- Use **FastAPI** instead of Flask for faster inference.
- Deploy model using **TensorRT** for optimized GPU execution.
- Utilize **GPU** (e.g., Google Colab) for faster model training and inference.

### 4. Deploy to Cloud

- Host API on **AWS/GCP/Azure** with **Docker & Kubernetes**.
- Use **CI/CD pipelines** for automated deployment.

