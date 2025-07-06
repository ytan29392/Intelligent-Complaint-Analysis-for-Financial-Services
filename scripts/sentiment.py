from textblob import TextBlob

def get_sentiment(text):
    """
    Returns sentiment label based on TextBlob polarity score:
    - Positive
    - Neutral
    - Negative
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'