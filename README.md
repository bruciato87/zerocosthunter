# Zero-Cost Investment Hunter Bot 🏹📈

An automated AI-powered investment assistant that hunts for high-quality market opportunities, filters for Trade Republic availability (High Cap/Liquidity), and notifies you via Telegram.

## 🚀 Features

-   **News Hunter:** Scrapes RSS feeds from CoinTelegraph, CNBC, Yahoo Finance, and MarketWatch.
-   **AI Brain (Gemini 1.5 Flash):** Analyzes news sentiment, strictly filters for major assets (S&P 500, BTC, ETH), and predicts price direction.
-   **Supabase Memory:** Logs all predictions to avoid duplicate alerts and track performance.
-   **Telegram Bot:** Sends real-time BUY/SELL signals directly to your phone.
-   **Zero-Cost:** Runs entirely on GitHub Actions (Free Tier) + Gemini Free Tier + Supabase Free Tier.

---

## 🛠️ Architecture

1.  **`main.py`**: Orchestrates the entire pipeline.
2.  **`hunter.py`**: Fetches market news.
3.  **`brain.py`**: Sends news to Google Gemini for specific analysis.
4.  **`db_handler.py`**: Manages Supabase connection.
5.  **`telegram_bot.py`**: Handles user notifications.

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
        git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
        
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

## 🛡️ Disclaimer

This bot provides information based on AI analysis of public news feeds. It is **NOT** financial advice. Automating trades based on this bot is risky. Always do your own research (DYOR).
