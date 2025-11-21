Breast Cancer Prediction API (Full-Stack Machine Learning SaaS)

(Note: You can replace this placeholder with a screenshot of your actual website dashboard)

🚀 Project Overview

This project is a complete, end-to-end Machine Learning Software-as-a-Service (SaaS) platform. It allows users (developers, researchers, or medical professionals) to access a powerful neural network model capable of predicting whether a breast tumor is malignant or benign with high confidence.

The project consists of three main components:

Machine Learning Model: A trained Multi-Layer Perceptron (MLP) classifier.

Backend API: A high-performance FastAPI server deployed on Render.

Frontend Portal: A responsive, user-friendly website deployed on Firebase Hosting for user management and API key generation.

🏗️ Architecture

The system follows a decoupled client-server architecture, ensuring scalability and security.

1. The "Brain" (Backend API)

Technology: Python, FastAPI, Scikit-learn, Uvicorn.

Deployment: Render.

Function:

Loads the pre-trained scikit-learn model and scaler.

Exposes a secure RESTful endpoint: POST /predict.

Security Middleware: Validates X-API-Key headers against the Firebase database before processing any request.

Usage Tracking: Automatically increments user usage stats in Firestore upon successful prediction.

2. The "Storefront" (Frontend Portal)

Technology: HTML5, Tailwind CSS, Vanilla JavaScript (ES6 modules).

Deployment: Firebase Hosting.

Function:

User Authentication: Handles Sign Up/Login using Firebase Auth.

API Key Management: Automatically generates secure, unique API keys for new users upon "payment" (demo flow).

Live Testing: Provides an interactive dashboard where users can test the model with sample data directly in the browser.

Documentation: Integrated API documentation with code snippets for Python and JavaScript.

3. The "Vault" (Database & Auth)

Technology: Firebase Authentication & Cloud Firestore.

Function:

Stores user profiles, encrypted passwords, API keys, and plan status (pro vs free).

Acts as the single source of truth for the backend to verify permissions.

🛠️ Tech Stack

Machine Learning

Python: Core language.

Scikit-learn: For model training (MLPClassifier), data splitting, and preprocessing (StandardScaler).

Pandas & NumPy: For data manipulation.

Joblib: For model serialization (saving/loading).

Backend

FastAPI: Modern, high-performance web framework for building APIs.

Firebase Admin SDK: To securely communicate with Firestore from the server.

Uvicorn: ASGI server for production deployment.

Frontend

HTML5 & JavaScript: Core web technologies.

Tailwind CSS: For rapid, responsive UI design.

Firebase SDK: For client-side authentication and database interactions.

Prism.js: For syntax highlighting in the documentation.

🤖 The Machine Learning Model

The model is trained on the Breast Cancer Wisconsin (Diagnostic) Data Set.

Input: 30 numerical features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

Examples: Radius (mean), Texture (mean), Perimeter (mean), Area (mean), Smoothness (mean), etc.

Algorithm: Multi-layer Perceptron (MLP) Neural Network.

Hidden Layers: Two layers (100 neurons, 50 neurons).

Activation Function: Logistic (Sigmoid).

Solver: Stochastic Gradient Descent (SGD).

Performance: The model achieves ~97-98% accuracy on the test set.

🔒 Security & Monetization Logic

This project implements a "Pay-to-Use" (SaaS) model:

Gatekeeping: The POST /predict endpoint is protected. It cannot be accessed without a key.

Authentication: Users must sign up to get a key.

Authorization (The "Pro" Plan):

When a user signs up, they simulate a "payment."

The system assigns them a plan: "pro" status in Firestore.

The API middleware checks this status for every request. If a user is not "pro," the API rejects the request, even if the key is valid.

Usage Limits: The system tracks the number of calls made by each user, enabling future features like rate limiting or pay-per-call billing.

🚀 How to Run Locally

Prerequisites

Python 3.9+

Node.js & NPM (for Firebase tools)

A Firebase Project (with Auth & Firestore enabled)

1. Clone the Repository

git clone [https://github.com/yourusername/breast-cancer-api.git](https://github.com/yourusername/breast-cancer-api.git)
cd breast-cancer-api


2. Backend Setup

# Install dependencies
pip install -r requirements.txt

# Run the training script (to generate .joblib files)
python breastcancer.py

# Start the API server
uvicorn api:app --reload


Note: You will need to place your serviceAccountKey.json from Firebase in the root directory.

3. Frontend Setup

# Serve the static files
python -m http.server 9000


Visit http://127.0.0.1:9000 in your browser.

📡 API Usage Example

Once you have your API Key from the dashboard, you can make predictions from your own code:

Python Example:

import requests

API_KEY = "YOUR_API_KEY"
URL = "[https://your-api-url.onrender.com/predict](https://your-api-url.onrender.com/predict)"

data = {
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}

headers = {"X-API-Key": API_KEY}
response = requests.post(URL, json=data, headers=headers)

print(response.json())
# Output: {'prediction': 'malignant', 'confidence': '1.0000'}


📝 License

This project is open-source and available under the MIT License.
