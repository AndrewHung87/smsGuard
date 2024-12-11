from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_messages(messages):
    # Return both vectorized data (X) and the fitted vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(messages)
    return X, vectorizer
