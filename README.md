# SMSGuard: AI-Based SMS Spam Detection System

SMSGuard is an AI-powered spam detection system that classifies SMS messages as either spam or ham (not spam). Built using Python, it leverages machine learning models to identify spam patterns with high accuracy.

---

## **Features**
- **Clustering:** Exploratory analysis using K-Means to understand message groupings.
- **Classification:** Naive Bayes classifier to predict spam or ham.
- **Web Interface:** Flask-based UI to input SMS messages and get predictions.
- **Visualization:** Clustering results visualized with PCA.

---

## **System Requirements**
- Python 3.8 or above
- pip or conda for package management
- Operating System: Linux (Ubuntu preferred), macOS, or Windows with WSL.

---

## **Setup and Installation**

### **Step 1: Clone the Repository**
``
bash
git clone <repository_url>
cd smsguard
``

### **Step 2: Create a Virtual Environment**
Using `venv`:
``
python3 -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
``

### **Step 3: Install Requirementst**
Install all dependencies listed in `requirements.txt`: 
``
pip install -r requirements.txt
``

---

## **Dataset Preparation**

1. Default Dataset:
   The repository includes a default dataset (`SMSSpamCollection`) in `data/raw/`.

2. Additional Dataset:
   If you have another dataset (e.g., `spam.csv`), place it in the `data/raw/` folder.

3. Combining Datasets:
   The application automatically combines datasets during preprocessing.

---

## **Running the Application**

### **Step 1: Train the Model**
To preprocess the data, train the model, and generate clustering visualizations:
`
python src/main.py
`

### **Step 2: Start the Web Application**
To launch the Flask-based web interface:
`
python src/app.py
`
Open this URL shown in the terminal in your browser to use the interface.

---

## **Commands Summary**

| **Command**                              | **Description**                                           |
|------------------------------------------|-----------------------------------------------------------|
| `python src/main.py`                     | Preprocesses data, trains the model, and saves it.        |
| `python src/app.py`                      | Runs the Flask web app for predictions.                  |

---

## **Project Structure**

smsguard/ │ ├─ data/ │ ├─ raw/ # Raw datasets (SMSSpamCollection, spam.csv, etc.) │ └─ processed/ # Processed data and visualizations │ ├─ models/ # Saved models and vectorizers │ ├─ trained_model.pkl # Trained Naive Bayes model │ └─ vectorizer.pkl # TF-IDF vectorizer │ ├─ src/ │ ├─ utils.py # Helper functions for loading data │ ├─ preprocess.py # Text cleaning and preprocessing │ ├─ feature_extraction.py # TF-IDF vectorization │ ├─ cluster_analysis.py # K-Means clustering and analysis │ ├─ train_model.py # Model training and evaluation │ ├─ main.py # Orchestrates the entire process │ └─ app.py # Flask application │ ├─ requirements.txt # Required Python libraries └─ README.md # Project documentation

---

## **References**

UCI SMS Spam Collection Dataset
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Kaggle SMS Spam Dataset
https://www.kaggle.com/uciml/sms-spam-collection-dataset

Scikit-learn Documentation
https://scikit-learn.org/stable/

Flask Documentation
https://flask.palletsprojects.com/
