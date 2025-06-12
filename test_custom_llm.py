from custom_dashscope_llm import CustomDashScopeLLM

# Test the custom LLM
try:
    llm = CustomDashScopeLLM(model_name="qwen-turbo", temperature=0.1)
    
    response = llm.invoke("Hello, how are you?")
    print(f"Response: {response}")
    
    # Test with a more complex prompt
    response2 = llm.invoke("What is RAG in AI?")
    print(f"Response 2: {response2}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
