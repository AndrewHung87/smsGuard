from flask import Flask, request, render_template_string
import joblib

# Load the model and vectorizer
model = joblib.load('models/trained_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

app = Flask(__name__)

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>SMSGuard</title>
</head>
<body>
<h1>SMSGuard: Spam Detector</h1>
<form method="POST" action="/">
    <label for="message">Enter your SMS message:</label><br><br>
    <textarea name="message" rows="5" cols="50" placeholder="Type your message here..."></textarea><br><br>
    <input type="submit" value="Classify">
</form>

{% if prediction %}
<h2>Prediction: {{ prediction }}</h2>
{% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_message = request.form.get("message", "")
        # Preprocess should match what we did before (quick minimal preprocess)
        # Note: For simplicity, let's replicate a minimal preprocessing here:
        processed = user_message.lower()
        # Vectorize
        X_input = vectorizer.transform([processed])
        pred = model.predict(X_input)
        prediction = pred[0]

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
