import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):

    #coverts text to lowercase
    text = text.lower()

    #remove punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    #remove stopword
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

def preprocess_dataframe(df):
    df['cleaned'] = df['message'].apply(clean_text)
    return df
