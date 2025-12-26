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
    
    # 1. Find Image File
    valid_extensions = [".jpg", ".jpeg", ".png"]
    image_path = None
    
    for ext in valid_extensions:
        temp_path = f"portfolio{ext}"
        if os.path.exists(temp_path):
            image_path = temp_path
            break
            
    if not image_path:
        logger.error("No 'portfolio.jpg', 'portfolio.jpeg', or 'portfolio.png' found in current directory.")
        logger.info("Please take a screenshot of your Trade Republic portfolio and save it here.")
        return

    # 2. Parse with AI (Vision)
    logger.info(f"Asking Gemini to look at {image_path}...")
    try:
        brain = Brain()
        holdings = brain.parse_portfolio_from_image(image_path)
    except Exception as e:
        logger.error(f"AI Parsing failed: {e}")
        return

    if not holdings:
        logger.warning("No holdings found in the image. Try a clearer screenshot.")
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
