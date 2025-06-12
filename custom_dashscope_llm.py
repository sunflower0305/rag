import os
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import dashscope
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CustomDashScopeLLM(LLM):
    """Custom DashScope LLM wrapper that works with LangChain"""
    
    model_name: str = "qwen-turbo"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set the API key
        dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not dashscope.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")
    
    @property
    def _llm_type(self) -> str:
        return "custom_dashscope"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the DashScope API"""
        try:
            from dashscope import Generation
            
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stop,
                **kwargs
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                raise Exception(f"DashScope API error: {response.message}")
                
        except Exception as e:
            raise Exception(f"Error calling DashScope API: {str(e)}")
