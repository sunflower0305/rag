from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from custom_qwen_embeddings import CustomQwenEmbeddings
import os
import time
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载.env文件中的环境变量
load_dotenv()

@dataclass
class Config:
    """配置类"""
    CACHE_DIR: str = "pdf_embeddings_cache"
    FAISS_INDEX_DIR: str = "qianwen_faiss_index"
    CHROMA_DB_DIR: str = "chroma_db"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    BATCH_SIZE: int = 4
    MAX_RETRIES: int = 3
    RETRIEVER_K: int = 5
    EMBEDDING_MODEL: str = "text-embedding-v4"
    LLM_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.1
    BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    USE_CHROMA: bool = True

config = Config()
def get_file_hash(file_path: str) -> str:
    """计算文件的MD5哈希值"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):  # 增大缓冲区
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_cache_dir(pdf_hash: str) -> str:
    """获取缓存目录路径"""
    cache_dir = Path(config.CACHE_DIR) / pdf_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)

def save_embeddings_cache(pdf_hash: str, vector_store, chunks_metadata: List[dict]) -> None:
    """保存嵌入缓存"""
    try:
        cache_dir = get_cache_dir(pdf_hash)
        
        if config.USE_CHROMA:
            # ChromaDB会自动持久化，无需额外保存
            logger.info(f"ChromaDB已自动持久化: {config.CHROMA_DB_DIR}")
        else:
            # 使用FAISS自带的保存方法
            vector_store.save_local(cache_dir)
        
        # 单独保存元数据
        metadata_path = Path(cache_dir) / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks_metadata': chunks_metadata,
                'timestamp': time.time(),
                'version': '1.0',
                'vector_store_type': 'chroma' if config.USE_CHROMA else 'faiss'
            }, f)
        
        logger.info(f"嵌入缓存已保存: {cache_dir}")
    except Exception as e:
        logger.error(f"保存缓存失败: {e}")
        raise

def load_embeddings_cache(pdf_hash: str, embeddings):
    """加载嵌入缓存"""
    cache_dir = Path(config.CACHE_DIR) / pdf_hash
    if not cache_dir.exists():
        return None, None
        
    try:
        # 加载元数据
        metadata_path = cache_dir / "metadata.pkl"
        if not metadata_path.exists():
            logger.warning("缓存目录存在但元数据文件缺失")
            return None, None
            
        with open(metadata_path, 'rb') as f:
            metadata_data = pickle.load(f)
        
        vector_store_type = metadata_data.get('vector_store_type', 'faiss')
        
        if config.USE_CHROMA and vector_store_type == 'chroma':
            # 加载ChromaDB
            collection_name = f"pdf_{pdf_hash}"
            vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=config.CHROMA_DB_DIR,
                embedding_function=embeddings
            )
            
            # 检查集合是否存在且有数据
            try:
                collection = vector_store._collection
                count = collection.count()
                if count > 0:
                    logger.info(f"找到ChromaDB缓存嵌入，包含{count}个文档")
                    return vector_store, metadata_data['chunks_metadata']
                else:
                    logger.warning("ChromaDB集合存在但为空")
                    return None, None
            except Exception as e:
                logger.warning(f"ChromaDB集合检查失败: {e}")
                return None, None
        elif not config.USE_CHROMA and vector_store_type == 'faiss':
            # 加载FAISS向量存储
            vector_store = FAISS.load_local(
                str(cache_dir), 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("找到FAISS缓存嵌入，跳过重复处理")
            return vector_store, metadata_data['chunks_metadata']
        else:
            logger.info(f"向量存储类型不匹配(缓存:{vector_store_type}, 当前:{('chroma' if config.USE_CHROMA else 'faiss')})，将重新生成")
            return None, None
            
    except Exception as e:
        logger.warning(f"缓存加载失败: {e}，将重新生成嵌入")
        return None, None


def validate_inputs(pdf_path: str, api_key: str) -> None:
    """验证输入参数"""
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    
    if not api_key:
        raise ValueError("API密钥不能为空")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("文件必须是PDF格式")

def create_paper_qa(pdf_path: str, api_key: str) -> Optional[RetrievalQA]:
    """
    创建一个基于阿里云千问的论文问答系统
    使用OpenAI兼容的方式调用千问模型
    """
    try:
        validate_inputs(pdf_path, api_key)
        logger.info("开始执行千问RAG流程")
        start_time = time.time()
        
        # 设置环境变量供OpenAI兼容接口使用
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = config.BASE_URL
        
        # 计算PDF文件哈希
        logger.info("计算PDF文件哈希...")
        pdf_hash = get_file_hash(pdf_path)
        logger.info(f"PDF哈希: {pdf_hash}")
        
        # 创建嵌入对象（用于加载缓存）
        embeddings = CustomQwenEmbeddings(
            api_key=api_key,
            model=config.EMBEDDING_MODEL
        )
        
        # 尝试加载缓存的嵌入
        vector_store, cached_metadata = load_embeddings_cache(pdf_hash, embeddings)
        
        if vector_store is not None:
            logger.info("使用缓存的嵌入，跳过文档处理和向量化步骤")
        else:
            # 1. 加载PDF文档
            logger.info("开始加载PDF文档...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("PDF文档为空或无法读取")
                
            logger.info(f"PDF加载完成，共 {len(documents)} 个文档页面")
            logger.info(f"耗时: {time.time() - start_time:.2f}秒")
            
            # 2. 文档分块
            logger.info("开始文档分块...")
            chunk_start = time.time()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("文档分块失败，未生成任何文本块")
                
            logger.info(f"文档分块完成，共 {len(chunks)} 个文本块")
            logger.info(f"耗时: {time.time() - chunk_start:.2f}秒")
            
            # 3. 生成文档嵌入
            logger.info("开始生成文档嵌入...")
            vector_start = time.time()
            
            all_texts = [doc.page_content for doc in chunks]
            all_metadatas = [doc.metadata for doc in chunks]
            
            if not all_texts:
                raise ValueError("没有文本内容可以生成嵌入")
            
            # 分批处理文档并添加重试机制
            vector_store = _create_vector_store_with_batches(
                all_texts, all_metadatas, embeddings, config.BATCH_SIZE, config.MAX_RETRIES, pdf_hash
            )
            
            logger.info("向量存储创建完成")
            logger.info(f"耗时: {time.time() - vector_start:.2f}秒")
            
            # 保存嵌入缓存
            save_embeddings_cache(pdf_hash, vector_store, all_metadatas)
        
        # 4. 创建检索器
        logger.info("创建检索器...")
        retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
        logger.info("检索器创建完成")
        
        # 5. 创建LLM - 使用OpenAI兼容方式调用千问大模型
        logger.info("初始化千问大语言模型...")
        llm_start = time.time()
        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            openai_api_key=api_key,
            openai_api_base=config.BASE_URL
        )
        logger.info("千问大语言模型初始化完成")
        logger.info(f"耗时: {time.time() - llm_start:.2f}秒")
        
        # 6. 创建RAG链
        logger.info("创建RAG链...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        logger.info("RAG链创建完成")
        
        # 7. 预置查询论文主旨
        _execute_default_queries(qa_chain)
        
        # 保存向量存储以便后续使用
        if not config.USE_CHROMA:
            logger.info("保存FAISS向量存储...")
            save_start = time.time()
            vector_store.save_local(config.FAISS_INDEX_DIR)
            logger.info(f"FAISS向量存储已保存到 '{config.FAISS_INDEX_DIR}'")
            logger.info(f"耗时: {time.time() - save_start:.2f}秒")
        else:
            logger.info("ChromaDB已自动持久化，无需手动保存")
        
        logger.info("千问RAG流程执行完成")
        logger.info(f"总耗时: {time.time() - start_time:.2f}秒")
        
        return qa_chain
        
    except Exception as e:
        logger.error(f"创建论文问答系统失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def _create_vector_store_with_batches(
    all_texts: List[str], 
    all_metadatas: List[dict], 
    embeddings, 
    batch_size: int, 
    max_retries: int,
    pdf_hash: str = None
):
    """分批创建向量存储"""
    if config.USE_CHROMA:
        return _create_chroma_vector_store(all_texts, all_metadatas, embeddings, pdf_hash)
    else:
        return _create_faiss_vector_store(all_texts, all_metadatas, embeddings, batch_size, max_retries)

def _create_chroma_vector_store(all_texts: List[str], all_metadatas: List[dict], embeddings, pdf_hash: str):
    """创建ChromaDB向量存储"""
    collection_name = f"pdf_{pdf_hash}" if pdf_hash else "default_collection"
    
    # 确保持久化目录存在
    Path(config.CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
    
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    
    # 批量添加文档以提高性能
    total_batches = (len(all_texts) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    
    for i in range(0, len(all_texts), config.BATCH_SIZE):
        batch_texts = all_texts[i:i+config.BATCH_SIZE]
        batch_metadatas = all_metadatas[i:i+config.BATCH_SIZE]
        batch_num = i // config.BATCH_SIZE + 1
        
        logger.info(f"处理ChromaDB批次 {batch_num}/{total_batches}...")
        
        # 为每个文档生成唯一ID
        batch_ids = [f"{pdf_hash}_{i+j}" for j in range(len(batch_texts))]
        
        try:
            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        except Exception as e:
            logger.warning(f"ChromaDB批次{batch_num}添加失败: {e}")
            # ChromaDB通常不需要重试，直接抛出异常
            raise
    
    logger.info(f"ChromaDB向量存储创建完成，集合名称: {collection_name}")
    return vector_store

def _create_faiss_vector_store(
    all_texts: List[str], 
    all_metadatas: List[dict], 
    embeddings, 
    batch_size: int, 
    max_retries: int
) -> FAISS:
    """创建FAISS向量存储"""
    vector_store = None
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        batch_metadatas = all_metadatas[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"处理FAISS批次 {batch_num}/{total_batches}...")
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metadatas)
                else:
                    batch_vector_store = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metadatas)
                    vector_store.merge_from(batch_vector_store)
                break
            except Exception as e:
                logger.warning(f"第{attempt + 1}次尝试失败: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
    
    return vector_store

def _execute_default_queries(qa_chain: RetrievalQA) -> None:
    """执行默认查询"""
    logger.info("开始查询论文主旨...")
    queries = [
        "这篇论文的主要研究内容是什么？",
    ]
    
    print("\n===== 论文主旨分析 =====")
    for i, query in enumerate(queries, 1):
        print(f"\n问题 {i}: {query}")
        query_start = time.time()
        try:
            response = qa_chain.invoke({"query": query})["result"]
            print(f"回答:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            print(f"查询耗时: {time.time() - query_start:.2f}秒")
        except Exception as e:
            logger.error(f"查询失败: {e}")

def ask_question(qa_chain: RetrievalQA, question: str) -> Optional[str]:
    """询问问题"""
    if not qa_chain or not question:
        return None
        
    try:
        query_start = time.time()
        response = qa_chain.invoke({"query": question})["result"]
        logger.info(f"问题回答完成，耗时: {time.time() - query_start:.2f}秒")
        return response
    except Exception as e:
        logger.error(f"问题回答失败: {e}")
        return None

def main():
    """主函数"""
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")
            
        pdf_file = 'attention_is_all_you_need.pdf'
        qa_chain = create_paper_qa(pdf_file, api_key)
        
        if not qa_chain:
            logger.error("创建问答系统失败")
            return
            
        # 自定义问题
        custom_question = '请给出文章的摘要'
        if custom_question:
            print("\n===== 自定义问题 =====")
            print(f"问题: {custom_question}")
            
            response = ask_question(qa_chain, custom_question)
            if response:
                print(f"回答:")
                print("-" * 50)
                print(response)
                print("-" * 50)
            else:
                print("问题回答失败")
                
    except Exception as e:
        logger.error(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()