
import logging
from brain import Brain

# Mock logger
logging.basicConfig(level=logging.INFO)

# Subclass to mock requests without real API calls
class MockBrain(Brain):
    def _call_deepseek_mock(self, mock_response_data):
        # We simulate the LOGIC inside _call_deepseek here, 
        # or simplified since we can't easily mock 'requests' inside the class without patching.
        # Let's extract the key logic block we just wrote.
        
        choice = mock_response_data['choices'][0]['message']
        reasoning = choice.get('reasoning_content', '')
        if reasoning:
            print(f"✅ LOGGED REASONING: {reasoning[:50]}...")
            
        content = choice.get('content', '')
        
        # JSON Repair Logic Re-implementation for Test
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            json_match_raw = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match_raw:
                content = json_match_raw.group(1)
                
        return content

def test_parser():
    b = MockBrain()
    
    # Case 1: Perfect R1 Response
    print("Testing Case 1: Perfect R1 Response...")
    data1 = {
        "choices": [{
            "message": {
                "reasoning_content": "Thinking about taxes... fee is 1 euro...",
                "content": '{"strategy": "hold"}'
            }
        }]
    }
    res1 = b._call_deepseek_mock(data1)
    assert res1 == '{"strategy": "hold"}'
    print("✅ Case 1 Passed")

    # Case 2: Chatty R1 Response (Markdown)
    print("\nTesting Case 2: Chatty Response...")
    data2 = {
        "choices": [{
            "message": {
                "reasoning_content": "Calculating...",
                "content": "Here is the JSON you requested:\n```json\n{\"strategy\": \"sell\"}\n```\nHope it helps!"
            }
        }]
    }
    res2 = b._call_deepseek_mock(data2)
    assert res2 == '{"strategy": "sell"}'
    print("✅ Case 2 Passed")

    # Case 3: Messy Response (No markdown, just text + json)
    print("\nTesting Case 3: Messy Response...")
    data3 = {
        "choices": [{
            "message": {
                "reasoning_content": "Hmm...",
                "content": "Sure, {\"strategy\": \"buy\"} is the best move."
            }
        }]
    }
    res3 = b._call_deepseek_mock(data3)
    assert res3 == '{"strategy": "buy"}'
    print("✅ Case 3 Passed")

if __name__ == "__main__":
    test_parser()
