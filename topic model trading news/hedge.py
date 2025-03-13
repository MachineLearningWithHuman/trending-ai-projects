import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
from stable_baselines3 import PPO
import gym
from gym import spaces

# Load Sample Data (Simulated Earnings Calls)
def load_data():
    df = pd.read_csv("earnings_call_transcripts.csv")
    return df

def preprocess_data(df):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(df["Transcript"].tolist(), show_progress_bar=True)
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(df["Transcript"].tolist(), embeddings)
    df["Topic"] = topics
    return df, topic_model

# Reinforcement Learning Trading Environment
class PortfolioEnv(gym.Env):
    def __init__(self, df):
        super(PortfolioEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.action_space = spaces.Box(low=0, high=1, shape=(len(df['Sector'].unique()),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(df['Sector'].unique()),), dtype=np.float32)
    
    def step(self, action):
        reward = np.random.randn()  # Simulated reward based on market conditions
        self.current_step += 1
        done = self.current_step >= len(self.df)
        return np.random.randn(len(self.df['Sector'].unique())), reward, done, {}

    def reset(self):
        self.current_step = 0
        return np.random.randn(len(self.df['Sector'].unique()))

# Streamlit UI
def main():
    st.title("Hedge Fund Strategy: Sector Rotation with Topic Modeling + RL")
    df = load_data()
    df, topic_model = preprocess_data(df)
    
    st.sidebar.header("User Controls")
    min_topic_size = st.sidebar.slider("Min Topic Size", 5, 50, 10)
    top_n_words = st.sidebar.slider("Top N Words per Topic", 5, 20, 10)
    investment_amount = st.sidebar.number_input("Investment Amount", min_value=1000, max_value=100000, value=10000, step=1000)
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=1)
    
    # Topic Distribution Bar Chart
    st.subheader("Topic Distribution")
    topic_counts = df['Topic'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(10,5))
    sns.barplot(x=topic_counts.index, y=topic_counts.values, ax=ax1)
    ax1.set_xlabel("Topics")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)
    
    # Word Clouds for Bullish & Bearish Topics
    st.subheader("Word Clouds")
    bullish_words = ' '.join(df[df['Topic'].isin([0, 1])]['Transcript'])
    bearish_words = ' '.join(df[df['Topic'].isin([2, 3])]['Transcript'])
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(WordCloud().generate(bullish_words))
    axes[0].set_title("Bullish Topics")
    axes[0].axis("off")
    axes[1].imshow(WordCloud().generate(bearish_words))
    axes[1].set_title("Bearish Topics")
    axes[1].axis("off")
    st.pyplot(fig2)
    
    # Reinforcement Learning Model Training
    st.subheader("Reinforcement Learning Model Training")
    env = PortfolioEnv(df)
    model = PPO("MlpPolicy", env, verbose=1,learning_rate=0.0003)
    
    # Track rewards during training
    rewards = []
    total_timesteps = 100
    for i in range(total_timesteps):
        model.learn(total_timesteps=1, reset_num_timesteps=False)
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    
    # Plot RL Training Rewards
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(range(total_timesteps), rewards, label="Cumulative Reward", color='blue')
    ax3.set_xlabel("Training Timesteps")
    ax3.set_ylabel("Cumulative Reward")
    ax3.set_title("RL Model Training Performance")
    ax3.legend()
    st.pyplot(fig3)
    
    # Optimal Portfolio Allocation
    st.subheader("Optimal Portfolio Allocation Before & After RL")
    allocation_before = np.random.dirichlet(np.ones(len(df['Sector'].unique())), size=1)[0]
    allocation_after = np.random.dirichlet(np.ones(len(df['Sector'].unique())), size=1)[0]
    
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].pie(allocation_before, labels=df['Sector'].unique(), autopct='%1.1f%%')
    axes[0].set_title("Before RL Optimization")
    axes[1].pie(allocation_after, labels=df['Sector'].unique(), autopct='%1.1f%%')
    axes[1].set_title("After RL Optimization")
    st.pyplot(fig3)
    
    # Portfolio Returns vs. Benchmark
    st.subheader("Cumulative Portfolio Returns vs. Benchmark")
    dates = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
    portfolio_returns = np.cumsum(np.random.randn(len(df)))
    benchmark_returns = np.cumsum(np.random.randn(len(df)))
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(dates, portfolio_returns, label="RL Portfolio", color='blue')
    ax4.plot(dates, benchmark_returns, label="S&P 500", color='red')
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Cumulative Returns")
    ax4.legend()
    st.pyplot(fig4)

if __name__ == "__main__":
    main()