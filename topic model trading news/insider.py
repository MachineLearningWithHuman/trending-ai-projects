import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("insider_trading_data.csv")  # Replace with real data source
    # Create a placeholder "Filing_Text" column
    data["Filing_Text"] = (
        "Trader: " + data["Trader_Type"] + " | " +
        "Transaction: " + data["Transaction_Type"] + " | " +
        "Stock: " + data["Stock_Symbol"]
    )
    return data

data = load_data()

# Convert Transaction_Date to datetime format
data["Transaction_Date"] = pd.to_datetime(data["Transaction_Date"])

# Sidebar for user inputs
st.sidebar.title("Settings")
st.sidebar.markdown("Customize the analysis below:")

# Topic Modeling Options
# st.sidebar.subheader("Topic Modeling")
# topic_model_option = st.sidebar.selectbox(
#     "Choose a Topic Modeling Algorithm",
#     ["BERTopic", "NMF", "LDA"]
# )
#num_topics = st.sidebar.slider("Number of Topics", min_value=2, max_value=20, value=5)

# Filters
st.sidebar.subheader("Filters")
trader_type_filter = st.sidebar.multiselect("Filter by Trader Type", data["Trader_Type"].unique())
stock_symbol_filter = st.sidebar.multiselect("Filter by Stock Symbol", data["Stock_Symbol"].unique())
date_range = st.sidebar.date_input("Filter by Date Range", [data["Transaction_Date"].min(), data["Transaction_Date"].max()])

# Apply filters
filtered_data = data[
    (data["Trader_Type"].isin(trader_type_filter) )if trader_type_filter else True
]
filtered_data = filtered_data[
    (filtered_data["Stock_Symbol"].isin(stock_symbol_filter)) if stock_symbol_filter else True
]
filtered_data = filtered_data[
    (filtered_data["Transaction_Date"] >= pd.to_datetime(date_range[0])) &
    (filtered_data["Transaction_Date"] <= pd.to_datetime(date_range[1]))
]

# Page Title
st.title("Insider Trading Analysis Dashboard")
st.markdown("Explore key insights from insider trading data.")

# Summary Statistics
st.subheader("Summary Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", len(filtered_data))
with col2:
    st.metric("Average Sentiment Score", round(filtered_data["Sentiment_Score"].mean(), 2))
with col3:
    st.metric("Total Trade Volume", f"{filtered_data['Trade_Volume'].sum():,}")

# Topic Modeling
# Topic Modeling
st.subheader("Topic Modeling")

# Sidebar options for topic modeling
st.sidebar.subheader("Topic Modeling Settings")
num_topics = st.sidebar.slider("Number of Topics", min_value=2, max_value=20, value=5)
topic_model_option = st.sidebar.selectbox(
    "Choose a Topic Modeling Algorithm",
    ["BERTopic", "NMF", "LDA"]
)

# Run topic modeling
if st.sidebar.button("Run Topic Modeling"):
    st.write(f"Running {topic_model_option} with {num_topics} topics...")
    texts = filtered_data["Filing_Text"].tolist()

    if topic_model_option == "BERTopic":
        # BERTopic
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        topic_model = BERTopic(nr_topics=num_topics)
        topics, _ = topic_model.fit_transform(texts, embeddings)

        # Visualize topic frequencies
        st.write("### Topic Frequencies")
        topic_freq = topic_model.get_topic_freq()
        st.dataframe(topic_freq)

        # Visualize top topics
        st.write("### Top Topics")
        fig = topic_model.visualize_barchart(top_n_topics=num_topics)
        st.plotly_chart(fig)

        # Visualize topic word clouds
        st.write("### Topic Word Clouds")
        for topic_id in range(num_topics):
            st.write(f"#### Topic {topic_id}")
            topic_words = topic_model.get_topic(topic_id)
            wordcloud = WordCloud(background_color="white", width=800, height=400).generate_from_frequencies(dict(topic_words))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

        # Visualize topic hierarchy
        st.write("### Topic Hierarchy")
        fig = topic_model.visualize_hierarchy()
        st.plotly_chart(fig)

        # Interactive topic exploration
        st.write("### Explore Topics")
        selected_topic = st.selectbox("Select a Topic", range(num_topics))
        st.write(f"#### Words in Topic {selected_topic}")
        topic_words = topic_model.get_topic(selected_topic)
        st.write(topic_words)

        st.write(f"#### Documents in Topic {selected_topic}")
        topic_docs = [text for text, topic in zip(texts, topics) if topic == selected_topic]
        st.write(topic_docs[:5])  # Show top 5 documents

    elif topic_model_option == "NMF":
        # NMF
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        nmf = NMF(n_components=num_topics, random_state=42)
        nmf.fit(tfidf)
        feature_names = vectorizer.get_feature_names_out()

        # Visualize top words for each topic
        st.write("### Top Words per Topic (NMF)")
        for idx, topic in enumerate(nmf.components_):
            st.write(f"#### Topic {idx + 1}")
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            st.write(", ".join(top_words))

            # Word cloud for each topic
            wordcloud = WordCloud(background_color="white", width=800, height=400).generate(" ".join(top_words))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

    elif topic_model_option == "LDA":
        # LDA
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(tfidf)
        feature_names = vectorizer.get_feature_names_out()

        # Visualize top words for each topic
        st.write("### Top Words per Topic (LDA)")
        for idx, topic in enumerate(lda.components_):
            st.write(f"#### Topic {idx + 1}")
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            st.write(", ".join(top_words))

            # Word cloud for each topic
            wordcloud = WordCloud(background_color="white", width=800, height=400).generate(" ".join(top_words))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

# Network Graph
# st.subheader("Insider Trading Network")
# G = nx.Graph()
# for _, row in filtered_data.iterrows():
#     if row["Trader_Type"] == "Insider":
#         G.add_edge(row["Trader_Type"], row["Stock_Symbol"], weight=row["Trade_Volume"])
# plt.figure(figsize=(10, 6))
# nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", font_size=10)
# st.pyplot(plt)

# Network Graph with Plotly
st.subheader("Insider Trading Network")

# Create a graph
G = nx.Graph()

# Add edges for all trader types
for _, row in filtered_data.iterrows():
    G.add_edge(row["Trader_Type"], row["Stock_Symbol"], weight=row["Trade_Volume"])

# Extract node positions for visualization
pos = nx.spring_layout(G)

# Create edge traces
edge_traces = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace = go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        line=dict(width=1, color="gray"),
        hoverinfo="none",
        mode="lines"
    )
    edge_traces.append(edge_trace)

# Create node traces
node_traces = []
for node in G.nodes():
    x, y = pos[node]
    node_trace = go.Scatter(
        x=[x], y=[y],
        mode="markers+text",
        text=[node],
        textposition="top center",
        marker=dict(
            size=20,
            color="lightblue" if node in filtered_data["Trader_Type"].unique() else "orange",
            line=dict(width=2, color="darkblue")
        ),
        hoverinfo="text",
        hovertext=f"Node: {node}"
    )
    node_traces.append(node_trace)

# Create the figure
fig = go.Figure(data=edge_traces + node_traces)

# Update layout for better visualization
fig.update_layout(
    title="Insider Trading Network",
    showlegend=False,
    hovermode="closest",
    margin=dict(b=0, l=0, r=0, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="white"
)

# Display the graph
st.plotly_chart(fig)

# Word Clouds
st.subheader("Insider Trading Sentiment Analysis")
bullish_threshold = 0.5
bearish_threshold = -0.5
bullish_words = " ".join(filtered_data[filtered_data["Sentiment_Score"] > bullish_threshold]["Filing_Text"])
bearish_words = " ".join(filtered_data[filtered_data["Sentiment_Score"] < bearish_threshold]["Filing_Text"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
wordcloud_bullish = WordCloud(background_color="white", colormap="Greens").generate(bullish_words)
wordcloud_bearish = WordCloud(background_color="white", colormap="Reds").generate(bearish_words)
axes[0].imshow(wordcloud_bullish, interpolation="bilinear")
axes[0].set_title("Bullish Topics")
axes[0].axis("off")
axes[1].imshow(wordcloud_bearish, interpolation="bilinear")
axes[1].set_title("Bearish Topics")
axes[1].axis("off")
st.pyplot(fig)

# Sentiment Trend
st.subheader("Sentiment Score Over Time")
agg_data = filtered_data.groupby(["Transaction_Date", "Trader_Type"], as_index=False)["Sentiment_Score"].mean()
fig = px.line(agg_data, x="Transaction_Date", y="Sentiment_Score", color="Trader_Type", title="Sentiment Score Over Time")
st.plotly_chart(fig)

# Trade Volume Heatmap
st.subheader("Trade Volume Heatmap")
pivot_table = filtered_data.pivot_table(index="Trader_Type", columns="Stock_Symbol", values="Trade_Volume", aggfunc="sum").fillna(0)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".0f", linewidths=0.5)
plt.title("Trade Volume Heatmap")
plt.xlabel("Stock Symbol")
plt.ylabel("Trader Type")
st.pyplot(plt)

# Additional Visualizations
st.subheader("Additional Insights")

# Top Traders by Trade Volume
st.write("#### Top Traders by Trade Volume")
trader_volume = filtered_data.groupby("Trader_Type", as_index=False)["Trade_Volume"].sum().sort_values(by="Trade_Volume", ascending=False)
fig = px.bar(trader_volume, x="Trader_Type", y="Trade_Volume", title="Top Traders by Trade Volume")
st.plotly_chart(fig)

# Top Stocks by Trade Volume
st.write("#### Top Stocks by Trade Volume")
stock_volume = filtered_data.groupby("Stock_Symbol", as_index=False)["Trade_Volume"].sum().sort_values(by="Trade_Volume", ascending=False)
fig = px.bar(stock_volume, x="Stock_Symbol", y="Trade_Volume", title="Top Stocks by Trade Volume")
st.plotly_chart(fig)

# Sentiment Distribution
st.write("#### Sentiment Distribution")
fig = px.histogram(filtered_data, x="Sentiment_Score", nbins=20, title="Distribution of Sentiment Scores")
st.plotly_chart(fig)

# Price Impact Analysis
st.write("#### Price Impact Analysis")
fig = px.scatter(filtered_data, x="Trade_Volume", y="Price_Impact", color="Trader_Type", title="Trade Volume vs. Price Impact")
st.plotly_chart(fig)

# Transaction Type Distribution
st.write("#### Transaction Type Distribution")
transaction_counts = filtered_data["Transaction_Type"].value_counts()
fig = px.pie(transaction_counts, values=transaction_counts.values, names=transaction_counts.index, title="Transaction Type Distribution")
st.plotly_chart(fig)

# Correlation Heatmap
st.write("#### Correlation Heatmap")
corr_matrix = filtered_data[["Trade_Volume", "Sentiment_Score", "Price_Impact"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
st.pyplot(plt)

# Top Insider Transactions
st.write("#### Top Insider Transactions")
insider_data = filtered_data[filtered_data["Trader_Type"] == "Insider"]
top_insider_transactions = insider_data.sort_values(by="Trade_Volume", ascending=False).head(10)
st.table(top_insider_transactions[["Transaction_Date", "Stock_Symbol", "Trade_Volume", "Sentiment_Score", "Price_Impact"]])

# Interactive Data Explorer
st.subheader("Interactive Data Explorer")
st.dataframe(filtered_data)