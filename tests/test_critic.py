import unittest
from unittest.mock import MagicMock, patch
import logging
from critic import Critic, CriticVerdict

class TestCritic(unittest.TestCase):
    def setUp(self):
        self.critic = Critic()
        # Mock the brain internal generation to avoid API calls
        self.critic.model = "mock-model"
        
    @patch('brain.Brain')
    def test_critique_approve_strong_signal(self, MockBrain):
        """Test that the Critic approves a fundamentally strong signal."""
        # Setup Mock Brain
        mock_brain_instance = MockBrain.return_value
        mock_brain_instance._generate_with_fallback.return_value = """
        {
            "verdict": "APPROVE",
            "score": 95,
            "concerns": [],
            "reasoning": "Solid fundamentals and technicals match."
        }
        """
        
        signal = {
            "ticker": "BTC-USD",
            "direction": "BUY",
            "confidence": 0.90,
            "reasoning": "Strong ETF inflows and breakout above resistance."
        }
        context = "Bitcoin price $100k. Market Sentiment Extreme Greed. No neg news."
        
        result = self.critic.critique_signal(signal, context)
        
        self.assertEqual(result.verdict, "APPROVE")
        self.assertEqual(result.score, 95)
        self.assertIn("Solid", result.reasoning)

    @patch('brain.Brain')
    def test_critique_reject_risky_signal(self, MockBrain):
        """Test that the Critic REJECTS a signal when context is dangerous."""
        # Setup Mock Brain
        mock_brain_instance = MockBrain.return_value
        # Simulate critic saying REJECT due to Macro Headwinds
        mock_brain_instance._generate_with_fallback.return_value = """
        {
            "verdict": "REJECT",
            "score": 20,
            "concerns": ["Macro Headwinds", "Fed Meeting Risk"],
            "reasoning": "Buying now is suicide ahead of FOMC rate hike layout."
        }
        """
        
        signal = {
            "ticker": "TSLA",
            "direction": "BUY",
            "confidence": 0.75,
            "reasoning": "New model launch rumored."
        }
        context = "Market crashing. VIX at 35. Fed speaks tomorrow."
        
        result = self.critic.critique_signal(signal, context)
        
        self.assertEqual(result.verdict, "REJECT")
        self.assertEqual(result.score, 20)
        self.assertIn("suicide", result.reasoning)
        self.assertIn("Macro Headwinds", result.concerns)

    def test_hold_signal_pass_through(self):
        """Test that HOLD signals skip the expensive critique."""
        signal = {"ticker": "AAPL", "direction": "HOLD", "confidence": 0.5}
        result = self.critic.critique_signal(signal, "Context")
        
        self.assertEqual(result.verdict, "APPROVE")
        self.assertEqual(result.score, 100)
        self.assertIn("Signal is HOLD", result.reasoning)

if __name__ == '__main__':
    unittest.main()
