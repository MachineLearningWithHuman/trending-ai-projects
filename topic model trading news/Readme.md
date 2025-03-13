# Financial Analysis Toolkit

This repository contains a set of Python scripts designed for financial analysis, leveraging various machine learning and natural language processing techniques. The toolkit is built using Streamlit for interactive web-based dashboards and includes modules for news analysis, hedge fund strategy simulation, and insider trading analysis.

## Files Overview

### 1. `app.py`
This script provides a real-time trading news analysis dashboard. It fetches the latest news articles related to stocks, crypto, and trading, and performs topic modeling using both LDA (Latent Dirichlet Allocation) and BERTopic. Additionally, it includes sentiment analysis to classify topics as "Bullish" or "Bearish" and visualizes the results using word clouds and bar charts.

![image](https://github.com/user-attachments/assets/2501c134-95ae-4867-9f47-f33bfe010cd6)

#### Key Features:
- **Real-time News Fetching**: Fetches news articles from NewsAPI.
- **Topic Modeling**: Uses LDA and BERTopic to extract topics from news articles.
- **Sentiment Analysis**: Classifies topics based on sentiment scores.
- **Visualizations**: Includes word clouds, bar charts, and sentiment analysis plots.

### 2. `hedge.py`
This script simulates a hedge fund strategy using sector rotation and reinforcement learning (RL). It loads earnings call transcripts, performs topic modeling using BERTopic, and trains an RL model to optimize portfolio allocation. The script also visualizes the performance of the RL model and compares it to a benchmark.

![image](https://github.com/user-attachments/assets/5aa9a61f-df15-4065-ad4f-422a4968a937)

#### Key Features:
- **Sector Rotation Strategy**: Uses topic modeling to identify key sectors.
- **Reinforcement Learning**: Trains an RL model to optimize portfolio allocation.
- **Performance Visualization**: Includes portfolio allocation pie charts, cumulative returns, and RL training performance plots.

### 3. `insider.py`
This script provides an interactive dashboard for analyzing insider trading data. It includes topic modeling using BERTopic, NMF, and LDA, and visualizes the data using network graphs, word clouds, and various other plots. The dashboard allows users to filter data by trader type, stock symbol, and date range.

![image](https://github.com/user-attachments/assets/aa4d1226-c7f5-4561-9cac-5eb310f3fcc0)

#### Key Features:
- **Topic Modeling**: Supports BERTopic, NMF, and LDA for topic extraction.
- **Network Graph**: Visualizes relationships between traders and stocks.
- **Sentiment Analysis**: Includes word clouds for bullish and bearish topics.
- **Interactive Filters**: Allows users to filter data by trader type, stock symbol, and date range.

## Installation

To run these scripts, you'll need to install the required Python packages. You can do this using `pip`:

```bash
pip install streamlit requests pandas numpy nltk matplotlib seaborn wordcloud gensim bertopic sentence-transformers stable-baselines3 gym networkx plotly scikit-learn
