SMSGuard

System Requirements
Python 3.8 or above
pip or conda for package management
Operating System: Linux (Ubuntu preferred), macOS, or Windows with WSL.


Setup and Installation:

Step 1: Clone the Repository
Using:
bash git clone <repository_url> cd smsguard

Step 2: Create a Virtual Environment
Using venv: 
python3 -m venv env source env/bin/activate  

Step 3: Install Requirementst
Install all dependencies listed in requirements.txt: 
pip install -r requirements.txt


Dataset Preparation

1. Default Dataset: The repository includes a default dataset (SMSSpamCollection, spam.csv) in data/raw/.

2. Additional Dataset: If you have another dataset, place it in the data/raw/ folder.

3. Combining Datasets: The application automatically combines datasets during preprocessing.


Running the Application

Step 1: Train the Model
To preprocess the data, train the model, and generate clustering visualizations: 
python src/main.py

Step 2: Start the Web Application
To launch the Flask-based web interface: 
python src/app.py

Commands Summary:

python src/main.py	Preprocesses data, trains the model, and saves it.

python src/app.py	Runs the Flask web app for predictions.


Project Structure

smsguard/
│
├─ data/
│  ├─ raw/                  # Raw datasets (SMSSpamCollection, spam.csv, etc.)
│  └─ processed/            # Processed data and visualizations
│
├─ models/                  # Saved models and vectorizers
│  ├─ trained_model.pkl     # Trained Naive Bayes model
│  └─ vectorizer.pkl        # TF-IDF vectorizer
│
├─ src/
│  ├─ utils.py              # Helper functions for loading data
│  ├─ preprocess.py         # Text cleaning and preprocessing
│  ├─ feature_extraction.py # TF-IDF vectorization
│  ├─ cluster_analysis.py   # K-Means clustering and analysis
│  ├─ train_model.py        # Model training and evaluation
│  ├─ main.py               # Orchestrates the entire process
│  └─ app.py                # Flask application
│
├─ requirements.txt         # Required Python libraries
└─ README.md                # Project documentation
