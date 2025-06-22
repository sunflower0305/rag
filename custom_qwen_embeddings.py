from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI
import os

class CustomQwenEmbeddings(Embeddings):
    """自定义千问嵌入类，使用OpenAI兼容接口"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-v4"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=1024,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"嵌入文档时出错: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=1024,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"嵌入查询时出错: {e}")
            raise