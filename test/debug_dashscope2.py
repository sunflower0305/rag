import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Set dashscope API key explicitly
import dashscope
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')

# Now import langchain_dashscope
from langchain_dashscope import ChatDashScope

print(f"API Key loaded: {bool(os.environ.get('DASHSCOPE_API_KEY'))}")

try:
    print("Trying initialization...")
    llm = ChatDashScope(model="qwen-turbo")
    print("ChatDashScope initialized successfully!")
    print(f"Client object: {llm.client}")
    
    # Try a simple invoke
    response = llm.invoke("Hello, how are you?")
    print(f"Response: {response}")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
