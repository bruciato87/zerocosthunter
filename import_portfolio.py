import logging
import sys
import os
from dotenv import load_dotenv
from db_handler import DBHandler
from brain import Brain

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Importer")

def import_portfolio():
    load_dotenv()
    
    # 1. Read Input File
    file_path = "portfolio_input.txt"
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found!")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if len(content) < 50:
        logger.error("File seems empty or too short. Please paste your portfolio text.")
        return

    # 2. Parse with AI
    logger.info("Asking Gemini to interpret your portfolio...")
    try:
        brain = Brain()
        holdings = brain.parse_portfolio_data(content)
    except Exception as e:
        logger.error(f"AI Parsing failed: {e}")
        return

    if not holdings:
        logger.warning("No holdings found. Try copy-pasting differently.")
        return

    # 3. Confirm and Save
    print("\n--- FOUND HOLDINGS ---")
    for item in holdings:
        print(f"🔹 {item['ticker']}: {item['quantity']} units @ ${item['avg_price']} (Sector: {item['sector']})")
    
    print("\n----------------------")
    confirm = input("Do you want to save these to Supabase? (y/n): ").strip().lower()
    
    if confirm == 'y':
        db = DBHandler()
        for item in holdings:
            db.add_to_portfolio(
                ticker=item['ticker'],
                amount=item['quantity'],
                price=item['avg_price'],
                sector=item['sector']
            )
        logger.info("✅ Portfolio successfully imported!")
    else:
        logger.info("❌ Import cancelled.")

if __name__ == "__main__":
    import_portfolio()
