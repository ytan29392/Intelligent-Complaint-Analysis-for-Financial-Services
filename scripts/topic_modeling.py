from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def prepare_corpus(texts):
    """
    Tokenizes and creates dictionary and corpus for LDA.
    """
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    return dictionary, corpus, tokenized_texts

def build_lda_model(corpus, dictionary, num_topics=5):
    """
    Builds LDA model.
    """
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    return lda_model

def display_topics(lda_model, num_words=5):
    """
    Prints the top words for each topic.
    """
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)

def generate_wordcloud(lda_model, topic_num, dictionary):
    """
    Generates a word cloud for a specific topic.
    """
    plt.figure(figsize=(8,6))
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(lda_model.show_topic(topic_num, topn=20)))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()