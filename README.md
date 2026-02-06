# Zero-Cost Investment Hunter Bot 🏹📈

An automated AI-powered investment assistant that hunts for high-quality market opportunities, filters for Trade Republic availability (High Cap/Liquidity), and notifies you via Telegram.

## 🚀 Features (Phase 3 Active)

-   **Zero-Cost Hunter:** Automated AI assistant that hunts for high-quality market opportunities.
-   **News Hunter (Fast Lane):** Scrapes RSS/X/Yahoo with parallel fetching (< 1h latency).
-   **AI Brain (Multi-Model):** Gemini 2.5 Flash + DeepSeek R1 for deep analysis.
-   **ML Ensemble (Hybrid):** Combines **Gradient Boosting** (Regime/Technical) with **LSTM** (Time-Series) for robust predictions.
-   **Tax-Loss Harvesting:** Advisor module optimized for **Italian Tax Law** (Crypto, ETFs, Stocks).
-   **Sector Rotation:** Multi-timeframe momentum analysis to identify leading sectors.
-   **Supabase Memory:** Logs all predictions, signals, and portfolio state.
-   **Telegram Bot:** 20+ commands for portfolio management and alerts.
-   **Zero-Cost:** Runs entirely on GitHub Actions/Vercel + Free Tier APIs.

---

## 🧠 AI Intelligence Levels

| Level | Module | Description |
|-------|--------|-------------|
| L1 | `hunter.py` | News collection (RSS/Yahoo/Social) - Parallel |
| L2 | `brain.py` | Sentiment analysis (Gemini 2.5 Flash) |
| L3 | `pattern_recognizer.py` | Chart pattern detection |
| L4 | `ml_predictor.py` | **Ensemble ML** (XGBoost + LSTM) |
| L5 | `market_data.py` | **Sector Rotation** & Macro Analysis |
| L6 | `rebalancer.py` | Portfolio strategy (DeepSeek R1 + Consensus) |
| L7 | `advisor.py` | **Tax-Loss Harvesting** (Italian Law) |
| L8 | `strategy_manager.py` | Governance Rules (LONG_TERM, SWING) |
| L9 | `economist.py` | Macro Regime (VIX, DXY, Yields) |
| L10 | `portfolio_backtest.py` | Performance Attribution & Risk Metrics |

---

## 📊 Key Telegram Commands

| Command | Description |
|---------|-------------|
| `/hunt` | 🏹 Manual news hunt |
| `/analyze TICKER` | 🔬 Deep dive on single ticker |
| `/rebalance` | ⚖️ Portfolio rebalancing with AI |
| `/harvest` | 💰 Check Tax-Loss Harvesting opps |
| `/sectors` | 🔄 View Sector Rotation/Momentum |
| `/portfolio` | 📊 Live portfolio value |
| `/benchmark` | 📈 Compare vs S&P500/BTC |
| `/macro` | 🏛 FED/VIX/DXY context |
| `/help` | ❓ All commands |

---

## 🛠️ Architecture

1.  **`main.py`**: Orchestrates the pipeline (Hunt -> Analyze -> Predict -> Signal).
2.  **`hunter.py`**: Parallel news fetching (Fast Lane).
3.  **`brain.py`**: LLM integration (Gemini/DeepSeek) for sentiment & reasoning.
4.  **`ml_predictor.py`**: **Hybrid Ensemble** (pure Python XGBoost + LSTM) for price prediction.
5.  **`market_data.py`**: Technical analysis & **Sector Rotation** logic.
6.  **`advisor.py`**: **Tax-Loss Harvesting** and strategic advice.
7.  **`rebalancer.py`**: Portfolio optimization engine.
8.  **`db_handler.py`**: Supabase persistence layer.
9.  **`telegram_bot.py`**: User interface and notification system.

---

## 📦 Detailed Setup Guide

Follow these steps carefully to get the bot running.

### 1. Prerequisites & Account Setup

Before processing, ensure you have the following accounts:

1.  **Google Cloud (Gemini API):**
    -   Go to [Google AI Studio](https://aistudio.google.com/).
    -   Click "Get API key" -> "Create API key in new project".
    -   Copy the key (starts with `AIza...`).

2.  **Supabase (Database):**
    -   Go to [Supabase](https://supabase.com/) and create a free account.
    -   Click "New Project" and give it a name.
    -   Once the project is created, go to **Project Settings (Code icon) -> API**.
    -   Copy the `Project URL`.
    -   Copy the `service_role` key (Warning: this key has admin privileges, which is what we need for a backend script like this).

3.  **Telegram (Notifications):**
    -   Open Telegram and search for **@BotFather**.
    -   Send `/newbot`, give it a name, and get the **HTTP API Token**.
    -   Search for **@userinfobot** and click Start. Copy your **Id** (this is your `CHAT_ID`).

### 2. Installation (Local)

1.  **Get the Code:**
    -   **Option A (New Repo):** Create a repository on GitHub, then clone it (replace `YOUR_USERNAME` and `REPO_NAME`):
        ```bash
        git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
        cd REPO_NAME
        ```
    -   **Option B (Existing Code):** If you already have the files, just open a terminal in the project folder.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    # MacOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration (.env)

1.  **Create the file:**
    Duplicate the `.env.example` file and rename it to `.env`:
    ```bash
    cp .env.example .env
    ```

2.  **Edit the file:**
    Open `.env` in your text editor and fill in the values you got in Step 1.

    ```env
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_KEY=eyJhb... (your service_role key)
    GEMINI_API_KEY=AIza...
    TELEGRAM_BOT_TOKEN=123...:ABC...
    TELEGRAM_CHAT_ID=123456
    ```

### 4. Database Setup (Schema)

1.  Go to your Supabase project dashboard.
2.  Click on the **SQL Editor** icon in the left sidebar.
3.  Click **New query**.
4.  Copy the entire content of the `schema.sql` file in this repository.
5.  Paste it into the SQL Editor and click **Run**.
6.  (Optional) Go to the **Table Editor** to verify that a table named `predictions` has been created.

### 5. Running the Bot

To test the bot manually:

```bash
python main.py
```

**What to expect:**
-   **Terminal:** You will see logs indicating "Fetching news...", "Analyzing...", "Saving to DB...", and potentially "Sent Telegram alert...".
-   **No News is Good News:** If the API finds no *actionable* high-confidence news for major assets, it might silently finish without sending a Telegram message. This is normal behavior to reduce spam.
-   **Telegram:** If a high-confidence signal is found, you will receive a message instantly.

---

## ☁️ Deployment (GitHub Actions)

To make this run automatically every few hours for free:

1.  **Push your code to GitHub:**
    -   Create a **new empty repository** on [GitHub](https://github.com/new).
    -   Run these commands in your terminal:
        ```bash
        # Initialize git (if you haven't already)
        git init
        
        # Add all files
        git add .
        
        # Commit changes
        git commit -m "Initial automated hunter setup"
        
        # Renaissance the branch to main
        git branch -M main
        
        # Link to your new repo (Replace URL with YOUR repository URL)
        git remote add origin https://github.com/bruciato87/zerocosthunter.git
        
        # Push the code
        git push -u origin main
        ```
2.  Go to your Repository's **Settings** tab.
3.  On the left, assume **Secrets and variables** -> **Actions**.
4.  Click **New repository secret** and add your 5 secrets exactly as named in the `.env` file:
    -   `SUPABASE_URL`
    -   `SUPABASE_KEY`
    -   `GEMINI_API_KEY`
    -   `TELEGRAM_BOT_TOKEN`
    -   `TELEGRAM_CHAT_ID`
5.  Go to the **Actions** tab in GitHub to see your workflow running (it runs on push and on schedule).

---

## ❓ Troubleshooting

-   **`Error: Module not found`**: Make sure you activated your virtual environment and ran `pip install -r requirements.txt`.
-   **Telegram messages not sending**: Verify your `TELEGRAM_CHAT_ID`. It must be an integer (e.g., `123456789`). Also, ensure you have clicked "Start" on your new bot in Telegram.
-   **Supabase connection error**: Ensure you used the `service_role` key, not the `anon` key, if you have Row Level Security (RLS) enabled (or disable RLS for testing).
-   **GitHub "Password authentication is not supported"**:
    -   GitHub does not accept your account password in the terminal.
    -   You must use a **Personal Access Token (PAT)**.
    -   **Quick Link:** [Generate Token (Classic)](https://github.com/settings/tokens/new)
    -   **Manual Path:** Settings -> Scroll to the very bottom left -> **Developer settings** -> Personal access tokens -> Tokens (classic).
    -   Generate a new token.
    -   **CRITICAL:** You MUST check the box for **`repo`** AND the box for **`workflow`**.
    -   When the terminal asks for "Password", paste this long token instead.
-   **Old Token Cached (Mac)**:
    -   If Git doesn't ask for a password and fails with 403/Authentication failed, your Mac remembered the old wrong token.
    -   Run this command to clear it:
        ```bash
        printf "protocol=https\nhost=github.com\n" | git credential-osxkeychain erase
        ```
    -   Then try `git push` again.

## 🛡️ Disclaimer

This bot provides information based on AI analysis of public news feeds. It is **NOT** financial advice. Automating trades based on this bot is risky. Always do your own research (DYOR).
