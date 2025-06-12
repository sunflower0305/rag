import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Test data
raw_documents = [
    "RAG (Retrieval-Augmented Generation) 是一种结合了检索和生成的AI技术。",
    "它通过检索相关文档来增强语言模型的生成能力。",
    "RAG系统通常包含一个向量数据库用于存储和检索文档。",
    "Transformer是一种基于注意力机制的神经网络架构。",
    "注意力机制允许模型关注输入序列的不同部分。"
]

print("=== 测试文档分割 ===")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splits = text_splitter.create_documents(raw_documents)
print(f"原始文档数量: {len(raw_documents)}")
print(f"分割后文档块数量: {len(splits)}")

print("\n=== 测试向量化 ===")
embeddings = DashScopeEmbeddings(model="text-embedding-v1")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("向量存储创建成功")

print("\n=== 测试检索 ===")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
question = "什么是RAG？"
docs = retriever.invoke(question)
print(f"检索到 {len(docs)} 个相关文档:")
for i, doc in enumerate(docs):
    print(f"文档 {i+1}: {doc.page_content}")

print("\n=== 测试完成 ===")
