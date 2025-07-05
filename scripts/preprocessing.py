import re
import string
import pandas as pd
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = re.sub(r'\d+', '', text)
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text
    else:
        return ""