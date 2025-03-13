import streamlit as st
import requests
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim import corpora, models
from bertopic import BERTopic

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("vader_lexicon")
nltk.download('punkt_tab') 

# API Key for NewsAPI (Replace with your API key)
NEWS_API_KEY = "96d70c10f98b428e8d0ec7500a35baab"
NEWS_URL = f"https://newsapi.org/v2/everything?q=stocks OR crypto OR trading&language=en&apiKey={NEWS_API_KEY}"

# Function to fetch news articles
@st.cache_data(ttl=600)  # Auto-refresh every 10 minutes
def fetch_news():
    response = requests.get(NEWS_URL)
    data = response.json()
    if "articles" in data:
        return [article["title"] + " " + article["description"] for article in data["articles"] if article["description"]]
    return []

news_articles = fetch_news()

# Preprocessing Function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

processed_articles = [preprocess_text(article) for article in news_articles]

# Topic Modeling using LDA
def apply_lda(processed_articles):
    dictionary = corpora.Dictionary(processed_articles)
    corpus = [dictionary.doc2bow(article) for article in processed_articles]
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=5)
    topic_words = {f"Topic {i}": [word.split("*")[-1].replace('"', "").strip() for word in topic.split("+")] for i, topic in topics}
    return topic_words

lda_topics = apply_lda(processed_articles)

# Topic Modeling using BERTopic
def apply_bertopic(news_articles):
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(news_articles)
    topic_words = topic_model.get_topics()
    return topic_words

bertopic_topics = apply_bertopic(news_articles)

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()

def get_topic_sentiment(words):
    scores = [sia.polarity_scores(word)["compound"] for word in words]
    avg_score = np.mean(scores)
    return "Bullish" if avg_score > 0 else "Bearish"

lda_sentiments = {topic: get_topic_sentiment(words) for topic, words in lda_topics.items()}

# Streamlit UI
st.title("📊 Real-Time Trading News & Topic Modeling")

# Sidebar
st.sidebar.header("🔍 Choose Analysis Method")
analysis_method = st.sidebar.radio("Select Topic Modeling Method:", ["LDA", "BERTopic"])

st.sidebar.header("🔄 Refresh Data")
if st.sidebar.button("Fetch Latest News"):
    st.experimental_rerun()

# Display News Articles
st.subheader("📰 Latest Trading News")
if news_articles:
    for i, article in enumerate(news_articles[:10]):
        st.write(f"**{i+1}. {article}**")
else:
    st.warning("No news articles available. Check API key.")

# Display Topics
st.subheader("📌 Extracted Topics")
if analysis_method == "LDA":
    for topic, words in lda_topics.items():
        st.write(f"**{topic}:** {', '.join(words)}")
elif analysis_method == "BERTopic":
    for topic_id, words in bertopic_topics.items():
        if words:
            st.write(f"**Topic {topic_id}:** {', '.join([word[0] for word in words[:5]])}")

# Display Sentiment Analysis
st.subheader("📈 Sentiment Analysis of Topics")
sentiment_df = pd.DataFrame(lda_sentiments.items(), columns=["Topic", "Sentiment"])

fig, ax = plt.subplots()
sns.countplot(x="Sentiment", data=sentiment_df, palette={"Bullish": "green", "Bearish": "red"}, ax=ax)
st.pyplot(fig)

# Word Clouds
st.subheader("🌤 Word Clouds for Bullish & Bearish Topics")
bullish_words = [word for topic, words in lda_topics.items() if lda_sentiments[topic] == "Bullish" for word in words]
bearish_words = [word for topic, words in lda_topics.items() if lda_sentiments[topic] == "Bearish" for word in words]

col1, col2 = st.columns(2)

with col1:
    if bullish_words:
        st.write("### Bullish Topics")
        wordcloud_bullish = WordCloud(width=600, height=400, background_color="white", colormap="Greens").generate(" ".join(bullish_words))
        st.image(wordcloud_bullish.to_array())
    else:
        st.write("No bullish topics detected.")

with col2:
    if bearish_words:
        st.write("### Bearish Topics")
        wordcloud_bearish = WordCloud(width=600, height=400, background_color="white", colormap="Reds").generate(" ".join(bearish_words))
        st.image(wordcloud_bearish.to_array())
    else:
        st.write("No bearish topics detected.")

# Auto-refresh every 10 minutes
st.sidebar.write("🔄 Data auto-refreshes every 10 minutes.")
