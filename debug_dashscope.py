import os
from dotenv import load_dotenv
from langchain_dashscope import ChatDashScope

# Load environment variables
load_dotenv()

print(f"API Key loaded: {bool(os.environ.get('DASHSCOPE_API_KEY'))}")
print(f"API Key starts with: {os.environ.get('DASHSCOPE_API_KEY', '')[:10]}...")

try:
    # Try different initialization methods
    print("Trying initialization with model parameter...")
    llm = ChatDashScope(model="qwen-turbo")
    print("ChatDashScope initialized successfully!")
    print(f"Client object: {llm.client}")
    
    # Try a simple invoke
    response = llm.invoke("Hello, how are you?")
    print(f"Response: {response}")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
