import os
from dotenv import load_dotenv
import dashscope

# Load environment variables
load_dotenv()

# Set the API key
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')

print(f"API Key loaded: {bool(dashscope.api_key)}")

try:
    from dashscope import Generation
    
    response = Generation.call(
        model='qwen-turbo',
        prompt='Hello, how are you?',
        max_tokens=100
    )
    
    print(f"Response: {response}")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
