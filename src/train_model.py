from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def train_and_evaluate(X, y):
    # Split data into training and testing (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam')
    rec = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')

    print("Model Performance:")
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1-score:  {f1:.2f}")

    return model

def save_model(model, vectorizer, model_path='models/trained_model.pkl', vec_path='models/vectorizer.pkl'):
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Model and vectorizer saved to {model_path} and {vec_path}")
