from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from typing import Literal, Union, AbstractSet
from langchain_dashscope.embeddings import DashScopeEmbeddings
from custom_dashscope_llm import CustomDashScopeLLM
# from langchain_dashscope.chat_models import ChatDashScope
from dotenv import load_dotenv
from openai import OpenAI
import os
import time
import argparse


# 加载.env文件中的环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量获取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)

def create_paper_qa(pdf_path, api_key):
    """
    创建一个基于阿里云千问的论文问答系统
    使用千问embedding模型进行向量化，使用千问大语言模型进行问答
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量获取API密钥
    try:
        print("===== 开始执行千问RAG流程 =====")
        start_time = time.time()
        
        # 设置阿里云千问API密钥
        os.environ["DASHSCOPE_API_KEY"] = api_key
        
        # 1. 加载PDF文档
        print("1. 开始加载PDF文档...")
        loader = PyPDFLoader("attention_is_all_you_need.pdf")
        documents = loader.load()
        print(f"✓ PDF加载完成，共 {len(documents)} 个文档页面")
        print(f"  耗时: {time.time() - start_time:.2f}秒")
        
        # 2. 文档分块
        print("\n2. 开始文档分块...")
        chunk_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ 文档分块完成，共 {len(chunks)} 个文本块")
        print(f"  耗时: {time.time() - chunk_start:.2f}秒")
        
        # 3. 创建向量存储 - 使用千问的嵌入模型
        print("\n3. 开始加载千问嵌入模型...")
        embed_start = time.time()
        
        # 设置dashscope API key
        import dashscope
        dashscope.api_key = api_key
        
        # 使用千问text-embedding-v2模型进行embedding
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4"
        )
        print(f"✓ 千问嵌入模型加载完成")
        print(f"  耗时: {time.time() - embed_start:.2f}秒")
        
        print("\n4. 开始生成文档嵌入...")
        vector_start = time.time()
        
        # 由于DashScope embedding API有批量大小限制(最大25)，我们需要分批处理
        batch_size = 20  # 设置为20以确保不超过限制
        all_texts = [doc.page_content for doc in chunks]
        all_metadatas = [doc.metadata for doc in chunks]
        
        # 分批处理文档
        vector_store = None
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            batch_metadatas = all_metadatas[i:i+batch_size]
            
            print(f"  处理批次 {i//batch_size + 1}/{(len(all_texts) + batch_size - 1)//batch_size}...")
            
            if vector_store is None:
                # 创建第一个向量存储
                vector_store = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metadatas)
            else:
                # 添加到现有向量存储
                batch_vector_store = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metadatas)
                vector_store.merge_from(batch_vector_store)
        print(f"✓ 向量存储创建完成")
        print(f"  耗时: {time.time() - vector_start:.2f}秒")
        
        # 4. 创建检索器
        print("\n5. 创建检索器...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print(f"✓ 检索器创建完成")
        
        # 5. 创建LLM - 使用千问大模型
        print("\n6. 初始化千问大语言模型...")
        llm_start = time.time()
        llm = CustomDashScopeLLM(
            model_name="qwen-plus",
            temperature=0.1
        )
        print(f"✓ 千问大语言模型初始化完成")
        print(f"  耗时: {time.time() - llm_start:.2f}秒")
        
        # 6. 创建RAG链
        print("\n7. 创建RAG链...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        print(f"✓ RAG链创建完成")
        
        # 7. 查询论文主旨
        print("\n8. 开始查询论文主旨...")
        query_start = time.time()
        queries = [
            "这篇论文的主要研究内容是什么？",
            "这篇论文的主要贡献和创新点是什么？",
            "这篇论文的研究方法是什么？",
            "这篇论文的实验结果如何？"
        ]
        
        print("\n===== 论文主旨分析 =====")
        for i, query in enumerate(queries, 1):
            print(f"\n问题 {i}: {query}")
            response = qa_chain.invoke({"query": query})["result"]
            print(f"回答:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            print(f"查询耗时: {time.time() - query_start:.2f}秒")
            query_start = time.time()
        
        # 保存向量存储以便后续使用
        print("\n9. 保存向量存储...")
        save_start = time.time()
        vector_store.save_local("qianwen_faiss_index")
        print(f"✓ 向量存储已保存到 'qianwen_faiss_index'")
        print(f"  耗时: {time.time() - save_start:.2f}秒")
        
        print("\n===== 千问RAG流程执行完成 =====")
        print(f"总耗时: {time.time() - start_time:.2f}秒")
        
        return qa_chain
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():

    api_key = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量获取API密钥
    # parser = argparse.ArgumentParser(description='使用阿里云千问模型分析论文主旨')
    # parser.add_argument('--pdf', type=str, required=True, help='attention_is_all_you_need.pdf')
    # parser.add_argument('--api_key', type=str, required=True, help='阿里云)
    # parser.add_argument('--query', type=str, help='自定义问题(可选)')
    #
    # args = parser.parse_args()
    pdf = 'txt.pdf'
    qa_chain = create_paper_qa(pdf, api_key)
    query = '请给出文章的摘要'
    # 如果用户提供了自定义问题，则额外回答
    if qa_chain and query:
        print("\n===== 自定义问题 =====")
        print(f"问题: {query}")
        query_start = time.time()
        response = qa_chain.invoke({"query": query})["result"]
        print(f"回答:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print(f"查询耗时: {time.time() - query_start:.2f}秒")

if __name__ == "__main__":
    main()