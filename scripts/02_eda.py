# scripts/02_eda.py
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_loader import load_raw_data
from src.config import config
import os

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

df = load_raw_data()

# Calculate returns and volatility
df['Return'] = df['Close'].pct_change()
df['Volatility_20'] = df['Return'].rolling(20).std() * (252**0.5)  # annualized

fig, axes = plt.subplots(4, 1, figsize=(14, 20))

# 1. Price + Volume
axes[0].plot(df.index, df['Close'], label='Close Price', color='#1f77b4', linewidth=1.2)
axes[0].set_title(f"{config.TICKER} – Price & Volume", fontsize=16, fontweight='bold')
axes[0].set_ylabel("Price ($)")
axes[0].legend()
ax2 = axes[0].twinx()
ax2.bar(df.index, df['Volume'], alpha=0.3, color='gray')
ax2.set_ylabel("Volume")

# 2. Candlestick-style simple plot
axes[1].plot(df.index, df['High'], color='green', alpha=0.6, label='High/Low')
axes[1].plot(df.index, df['Low'], color='red', alpha=0.6)
axes[1].fill_between(df.index, df['Low'], df['High'], alpha=0.2, color='gray')
axes[1].plot(df.index, df['Close'], color='black', linewidth=1.5, label='Close')
axes[1].set_title("Daily High / Low Range", fontsize=14)
axes[1].legend()

# 3 Daily Returns
axes[2].plot(df.index, df['Return'] * 100, color='#ff7f0e')
axes[2].set_title("Daily Returns (%)", fontsize=14)
axes[2].set_ylabel("Return (%)")
axes[2].axhline(0, color='black', linewidth=0.8)

# 4 Rolling Volatility
axes[3].plot(df.index, df['Volatility_20'], color='#d62728', linewidth=2)
axes[3].set_title("20-Day Annualized Volatility", fontsize=14)
axes[3].set_ylabel("Volatility")

plt.tight_layout()

# Save plots
os.makedirs(config.FIGURES_DIR, exist_ok=True)
plt.savefig(config.FIGURES_DIR / "eda_overview.png", dpi=300, bbox_inches='tight')
print(f"Plots saved → {config.FIGURES_DIR / 'eda_overview.png'}")
plt.show()