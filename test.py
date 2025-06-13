import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

# Load environment variables from .env file
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_dashscope import DashScopeEmbeddings
from custom_dashscope_llm import CustomDashScopeLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. 设置你的DashScope API Key ---
# 确保你已经设置了DASHSCOPE_API_KEY环境变量
# 例如：os.environ["DASHSCOPE_API_KEY"] = "YOUR_DASHSCOPE_API_KEY"
if not os.environ.get("DASHSCOPE_API_KEY"):
    print("请设置DASHSCOPE_API_KEY环境变量。")
    exit()

# --- 2. 准备知识库文档 ---
# 这里我们使用一个简单的字符串作为文档内容
# 在实际应用中，你可以从文件、网页、数据库等加载文档
raw_documents = [
    "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的机器。",
    "机器学习（ML）是人工智能的一个子集，它使系统能够从数据中学习，而无需明确编程。",
    "深度学习（DL）是机器学习的一个子集，它使用多层神经网络来学习数据的复杂模式。",
    "自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
    "计算机视觉是人工智能的一个领域，使计算机能够从图像和视频中“看”和理解内容。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。",
    "DashScope是阿里云提供的大模型服务平台，提供了多种AI模型，包括文本生成、嵌入和多模态模型。",
    "ChromaDB是一个开源的嵌入数据库，可以轻松地存储和检索嵌入。",
    "RAG（Retrieval Augmented Generation）是一种结合了检索和生成能力的AI模型，它首先从外部知识库中检索相关信息，然后利用这些信息来生成更准确和上下文相关的回答。",
    "通义千问是阿里云开发的一系列大型语言模型，支持多种语言和任务。"
    "张鹏飞是张玲玲的父亲，张玲玲是石磊的女儿。"
]

# 将字符串列表转换为LangChain的Document对象
documents = [{"page_content": doc} for doc in raw_documents]

# --- 3. 文档分割 (Text Splitting) ---
# 将长文档分割成更小的块，以便更好地进行检索
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # 每个块的最大字符数
    chunk_overlap=20  # 块之间的重叠字符数
)
splits = text_splitter.create_documents(raw_documents)
print(f"原始文档数量: {len(raw_documents)}")
print(f"分割后文档块数量: {len(splits)}")

# --- 4. 创建嵌入模型 (Embedding Model) ---
# 使用DashScope的文本嵌入模型
embeddings_model = DashScopeEmbeddings(model="text-embedding-v2")

# --- 5. 构建向量存储 (Vector Store) ---
# 使用Chroma作为向量数据库，从分割后的文档块创建向量存储
# 第一次运行时会生成嵌入并存储，后续可以直接加载
print("正在创建或加载向量存储...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings_model,
    persist_directory="./chroma_db" # 将向量存储持久化到本地文件
)
print("向量存储准备就绪。")

# 创建检索器
retriever = vectorstore.as_retriever()

# --- 6. 定义RAG的提示模板 (Prompt Template) ---
# 这个模板告诉LLM如何使用检索到的上下文来回答问题
prompt_template = PromptTemplate.from_template(
    """你是一个乐于助人的AI助手。请根据提供的上下文信息回答问题。如果上下文中没有足够的信息，请说明你无法回答。

上下文信息:
{context}

问题: {question}

回答:"""
)

# --- 7. 初始化语言模型 (LLM) ---
# 使用自定义的DashScope LLM包装器
llm = CustomDashScopeLLM(
    model_name="qwen-turbo", 
    temperature=0.1
)

# --- 8. 构建RAG链 (RAG Chain) ---
# 这是一个LangChain表达式语言（LCEL）链，定义了RAG的流程

def format_docs(docs):
    """将检索到的文档格式化为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} # 检索上下文，并传入原始问题
    | prompt_template                                       # 将上下文和问题格式化到提示模板
    | llm                                                   # 调用LLM生成答案
    | StrOutputParser()                                     # 将LLM的输出解析为字符串
)

# --- 9. 运行RAG查询 ---
print("\n--- 开始RAG查询 ---")

# 示例查询1：答案在知识库中
question1 = "什么是RAG？"
print(f"\n问题: {question1}")
response1 = rag_chain.invoke(question1)
print(f"回答: {response1}")

# 示例查询2：答案在知识库中
question2 = "通义千问是谁开发的？"
print(f"\n问题: {question2}")
response2 = rag_chain.invoke(question2)
print(f"回答: {response2}")

# 示例查询3：答案不在知识库中
question3 = "地球离太阳有多远？"
print(f"\n问题: {question3}")
response3 = rag_chain.invoke(question3)
print(f"回答: {response3}")

# 示例查询4：更复杂的查询，需要结合多个信息
question4 = "LangChain和ChromaDB在构建AI应用中有什么作用？"
print(f"\n问题: {question4}")
response4 = rag_chain.invoke(question4)
print(f"回答: {response4}")

# 示例查询5：
question4 = "ChromaDB有什么作用？"
print(f"\n问题: {question4}")
response4 = rag_chain.invoke(question4)
print(f"回答: {response4}")

# 示例查询6：
question4 = "张军和石晶是什么关系？"
print(f"\n问题: {question4}")
response4 = rag_chain.invoke(question4)
print(f"回答: {response4}")

# 清理ChromaDB（可选，如果你想重新开始）
# vectorstore.delete_collection()
# print("\nChromaDB collection deleted.")
