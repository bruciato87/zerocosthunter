from db_handler import DBHandler

def main():
    print("Updating user settings...")
    try:
        db = DBHandler()
        # Force update to 0.50
        success = db.update_settings(min_confidence=0.50)
        
        if success:
            print("✅ Settings updated successfully!")
            # Verify
            new_settings = db.get_settings()
            print(f"Current Settings: {new_settings}")
        else:
            print("❌ Failed to update settings.")
            
    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    main()
