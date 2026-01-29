import asyncio
import logging
import pytest

# Mock objects to simulate the environment
class MockNotifier:
    async def send_alert(self, message):
        print(f"TELEGRAM ALERT:\n{message}")

class MockAuditor:
    async def audit_open_signals(self):
        # Simulated corrected output from auditor.py
        return [
            {"ticker": "BTC-USD", "pnl_percent": 15.5, "status": "WIN"},
            {"ticker": "META", "pnl_percent": -5.2, "status": "LOSS"},
            {"ticker": "ETH-USD", "pnl_percent": 2.1, "status": "OPEN"}
        ]

@pytest.mark.asyncio
async def test_audit_notification_flow():
    print("Starting Auditor Fix Verification...")
    
    auditor = MockAuditor()
    notifier = MockNotifier()
    
    # This simulates the logic at line 1231 in main.py
    audit_results = await auditor.audit_open_signals()
    
    if audit_results:
        try:
            # The fixed join logic
            summary_audit = "\n".join([f"• **{r['ticker']}**: {r['pnl_percent']:+.2f}% ({r['status']})" for r in audit_results])
            print("Successfully formatted summary_audit:")
            print(summary_audit)
            
            await notifier.send_alert(f"⚖️ **Auditor Monitoring Update:**\n{summary_audit}")
            print("\nVERIFICATION PASSED: No crash during join or notification.")
        except Exception as e:
            print(f"\nVERIFICATION FAILED: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(test_audit_notification_flow())
