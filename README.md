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
Step 1: Train the Model
To preprocess the data, train the model, and generate clustering visualizations:

