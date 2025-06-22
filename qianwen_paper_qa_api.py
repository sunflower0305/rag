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
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass
import shutil

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
    VECTOR_STORE_TYPE: str = "chroma"  # "chroma" or "faiss"

config = Config()

class QianwenPaperQAAPI:
    """千问论文问答API封装类"""
    
    def __init__(self, api_key: str = None):
        """初始化API"""
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API密钥不能为空，请设置DASHSCOPE_API_KEY环境变量或传入api_key参数")
        
        self.qa_chain: Optional[RetrievalQA] = None
        self.current_document: Optional[str] = None
        self.document_info: Dict[str, Any] = {}
        self.vector_store: Optional[Union[FAISS, Chroma]] = None
        self.embeddings = None
        
        # 设置环境变量供OpenAI兼容接口使用
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_BASE_URL"] = config.BASE_URL
    
    def get_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_cache_dir(self, pdf_hash: str) -> str:
        """获取缓存目录路径"""
        cache_dir = Path(config.CACHE_DIR) / pdf_hash
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)
    
    def save_embeddings_cache(self, pdf_hash: str, vector_store, chunks_metadata: List[dict]) -> None:
        """保存嵌入缓存"""
        try:
            cache_dir = self.get_cache_dir(pdf_hash)
            
            # 根据向量存储类型保存
            if isinstance(vector_store, FAISS):
                vector_store.save_local(cache_dir)
            elif isinstance(vector_store, Chroma):
                # ChromaDB 会自动持久化，无需手动保存
                pass
            
            metadata_path = Path(cache_dir) / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'chunks_metadata': chunks_metadata,
                    'timestamp': time.time(),
                    'version': '1.0',
                    'vector_store_type': config.VECTOR_STORE_TYPE
                }, f)
            
            logger.info(f"嵌入缓存已保存: {cache_dir}")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
            raise
    
    def load_embeddings_cache(self, pdf_hash: str, embeddings) -> Tuple[Optional[Union[FAISS, Chroma]], Optional[List[dict]]]:
        """加载嵌入缓存"""
        cache_dir = Path(config.CACHE_DIR) / pdf_hash
        if not cache_dir.exists():
            return None, None
            
        try:
            metadata_path = cache_dir / "metadata.pkl"
            if not metadata_path.exists():
                logger.warning("缓存目录存在但元数据文件缺失")
                return None, None
                
            with open(metadata_path, 'rb') as f:
                metadata_data = pickle.load(f)
            
            vector_store_type = metadata_data.get('vector_store_type', 'faiss')
            
            if vector_store_type == 'faiss':
                vector_store = FAISS.load_local(
                    str(cache_dir), 
                    embeddings=embeddings, 
                    allow_dangerous_deserialization=True
                )
            elif vector_store_type == 'chroma':
                chroma_db_path = Path(config.CHROMA_DB_DIR) / pdf_hash
                if chroma_db_path.exists():
                    collection_name = f"doc_{pdf_hash[:8]}"
                    vector_store = Chroma(
                        persist_directory=str(chroma_db_path),
                        embedding_function=embeddings,
                        collection_name=collection_name
                    )
                    # 验证集合是否有数据
                    try:
                        count = vector_store._collection.count()
                        if count == 0:
                            logger.warning("ChromaDB集合为空，需要重新生成")
                            return None, None
                    except Exception as e:
                        logger.warning(f"检查ChromaDB集合失败: {e}")
                        return None, None
                else:
                    logger.warning("ChromaDB目录不存在")
                    return None, None
            else:
                logger.warning(f"未知的向量存储类型: {vector_store_type}")
                return None, None
            
            logger.info(f"找到缓存嵌入({vector_store_type})，跳过重复处理")
            return vector_store, metadata_data['chunks_metadata']
                
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}，将重新生成嵌入")
            return None, None
    
    def validate_pdf_file(self, file_path: str) -> None:
        """验证PDF文件"""
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("文件必须是PDF格式")
    
    def create_vector_store_with_batches(
        self, 
        all_texts: List[str], 
        all_metadatas: List[dict], 
        embeddings, 
        batch_size: int, 
        max_retries: int,
        pdf_hash: str = None
    ) -> Union[FAISS, Chroma]:
        """分批创建向量存储"""
        if config.VECTOR_STORE_TYPE == "chroma":
            return self._create_chroma_vector_store(all_texts, all_metadatas, embeddings, pdf_hash)
        else:
            return self._create_faiss_vector_store(all_texts, all_metadatas, embeddings, batch_size, max_retries)
    
    def _create_chroma_vector_store(
        self,
        all_texts: List[str],
        all_metadatas: List[dict],
        embeddings,
        pdf_hash: str
    ) -> Chroma:
        """创建ChromaDB向量存储"""
        chroma_db_path = Path(config.CHROMA_DB_DIR) / pdf_hash if pdf_hash else Path(config.CHROMA_DB_DIR) / "default"
        chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        # 为每个文档创建唯一的collection名称
        collection_name = f"doc_{pdf_hash[:8]}" if pdf_hash else "default_collection"
        
        # 创建空的ChromaDB实例
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=str(chroma_db_path),
            collection_name=collection_name
        )
        
        # 分批添加文档以避免API批次大小限制
        batch_size = config.BATCH_SIZE  # 使用配置的批次大小
        total_batches = (len(all_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            batch_metadatas = all_metadatas[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"处理ChromaDB批次 {batch_num}/{total_batches}...")
            
            # 生成批次IDs
            batch_ids = [f"{pdf_hash}_{i+j}" for j in range(len(batch_texts))]
            
            # 添加文档到ChromaDB
            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        logger.info(f"ChromaDB向量存储创建完成，集合: {collection_name}")
        return vector_store
    
    def _create_faiss_vector_store(
        self,
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
            
            logger.info(f"处理批次 {batch_num}/{total_batches}...")
            
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
                    time.sleep(2 ** attempt)
        
        return vector_store
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        处理PDF文档并创建问答系统
        返回处理结果和文档信息
        """
        try:
            self.validate_pdf_file(pdf_path)
            logger.info("开始处理PDF文档")
            start_time = time.time()
            
            # 计算PDF文件哈希
            pdf_hash = self.get_file_hash(pdf_path)
            logger.info(f"PDF哈希: {pdf_hash}")
            
            # 创建嵌入对象
            self.embeddings = CustomQwenEmbeddings(
                api_key=self.api_key,
                model=config.EMBEDDING_MODEL
            )
            
            # 尝试加载缓存的嵌入
            vector_store, cached_metadata = self.load_embeddings_cache(pdf_hash, self.embeddings)
            
            chunks_count = 0
            pages_count = 0
            
            if vector_store is not None:
                logger.info("使用缓存的嵌入")
                chunks_count = len(cached_metadata) if cached_metadata else 0
            else:
                # 加载PDF文档
                logger.info("加载PDF文档...")
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                if not documents:
                    raise ValueError("PDF文档为空或无法读取")
                
                pages_count = len(documents)
                logger.info(f"PDF加载完成，共 {pages_count} 个文档页面")
                
                # 文档分块
                logger.info("文档分块...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP
                )
                chunks = text_splitter.split_documents(documents)
                
                if not chunks:
                    raise ValueError("文档分块失败")
                
                chunks_count = len(chunks)
                logger.info(f"文档分块完成，共 {chunks_count} 个文本块")
                
                # 生成文档嵌入
                logger.info("生成文档嵌入...")
                all_texts = [doc.page_content for doc in chunks]
                all_metadatas = [doc.metadata for doc in chunks]
                
                vector_store = self.create_vector_store_with_batches(
                    all_texts, all_metadatas, self.embeddings, config.BATCH_SIZE, config.MAX_RETRIES, pdf_hash
                )
                
                # 保存嵌入缓存
                self.save_embeddings_cache(pdf_hash, vector_store, all_metadatas)
            
            # 存储向量存储引用
            self.vector_store = vector_store
            
            # 创建检索器
            retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
            
            # 创建LLM
            llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                openai_api_key=self.api_key,
                openai_api_base=config.BASE_URL
            )
            
            # 创建RAG链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )
            
            # 更新文档信息
            self.current_document = pdf_path
            self.document_info = {
                'file_name': os.path.basename(pdf_path),
                'file_path': pdf_path,
                'file_hash': pdf_hash,
                'pages_count': pages_count,
                'chunks_count': chunks_count,
                'processing_time': time.time() - start_time,
                'timestamp': time.time(),
                'vector_store_type': config.VECTOR_STORE_TYPE
            }
            
            logger.info(f"文档处理完成，总耗时: {time.time() - start_time:.2f}秒")
            
            return {
                'success': True,
                'message': '文档处理成功',
                'document_info': self.document_info
            }
            
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            return {
                'success': False,
                'message': f'文档处理失败: {str(e)}',
                'document_info': None
            }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        询问问题
        返回回答结果
        """
        if not self.qa_chain:
            return {
                'success': False,
                'message': '请先上传并处理PDF文档',
                'answer': None,
                'processing_time': 0
            }
        
        if not question or not question.strip():
            return {
                'success': False,
                'message': '问题不能为空',
                'answer': None,
                'processing_time': 0
            }
        
        try:
            start_time = time.time()
            logger.info(f"处理问题: {question}")
            
            response = self.qa_chain.invoke({"query": question.strip()})["result"]
            processing_time = time.time() - start_time
            
            logger.info(f"问题回答完成，耗时: {processing_time:.2f}秒")
            
            return {
                'success': True,
                'message': '回答生成成功',
                'answer': response,
                'processing_time': processing_time,
                'question': question.strip()
            }
            
        except Exception as e:
            logger.error(f"问题回答失败: {e}")
            return {
                'success': False,
                'message': f'问题回答失败: {str(e)}',
                'answer': None,
                'processing_time': 0
            }
    
    def get_document_summary(self) -> Dict[str, Any]:
        """获取文档摘要"""
        if not self.qa_chain:
            return {
                'success': False,
                'message': '请先上传并处理PDF文档',
                'summary': None
            }
        
        summary_question = "请简要总结这篇文档的主要内容、核心观点和关键信息。"
        return self.ask_question(summary_question)
    
    def get_document_info(self) -> Dict[str, Any]:
        """获取当前文档信息"""
        return {
            'has_document': self.qa_chain is not None,
            'document_info': self.document_info if self.qa_chain else None
        }
    
    def add_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        向现有向量存储中添加新文档（仅支持ChromaDB）
        """
        if config.VECTOR_STORE_TYPE != "chroma":
            return {
                'success': False,
                'message': '动态更新仅支持ChromaDB，请在配置中设置VECTOR_STORE_TYPE="chroma"',
                'document_info': None
            }
        
        if not self.vector_store or not isinstance(self.vector_store, Chroma):
            return {
                'success': False,
                'message': '请先初始化一个文档再添加新文档',
                'document_info': None
            }
            
        try:
            self.validate_pdf_file(pdf_path)
            logger.info(f"开始添加新文档: {pdf_path}")
            start_time = time.time()
            
            # 加载PDF文档
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("PDF文档为空或无法读取")
            
            pages_count = len(documents)
            logger.info(f"PDF加载完成，共 {pages_count} 个文档页面")
            
            # 文档分块
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("文档分块失败")
            
            chunks_count = len(chunks)
            logger.info(f"文档分块完成，共 {chunks_count} 个文本块")
            
            # 添加到现有向量存储
            all_texts = [doc.page_content for doc in chunks]
            all_metadatas = [{
                **doc.metadata,
                'source_file': os.path.basename(pdf_path),
                'added_at': time.time()
            } for doc in chunks]
            
            # 分批添加文档以避免 API 限制
            batch_size = config.BATCH_SIZE
            total_batches = (len(all_texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i+batch_size]
                batch_metadatas = all_metadatas[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"添加批次 {batch_num}/{total_batches}...")
                
                # 生成批次IDs
                batch_ids = [f"add_{int(time.time())}_{i+j}" for j in range(len(batch_texts))]
                
                self.vector_store.add_texts(
                    texts=batch_texts, 
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"文档已添加到向量存储")
            
            # 更新文档信息
            processing_time = time.time() - start_time
            added_doc_info = {
                'file_name': os.path.basename(pdf_path),
                'file_path': pdf_path,
                'pages_count': pages_count,
                'chunks_count': chunks_count,
                'processing_time': processing_time,
                'timestamp': time.time()
            }
            
            # 如果有多个文档，将其存储在列表中
            if 'documents' not in self.document_info:
                self.document_info['documents'] = [self.document_info.copy()] if self.document_info else []
                
            self.document_info['documents'].append(added_doc_info)
            
            logger.info(f"文档添加完成，总耗时: {processing_time:.2f}秒")
            
            return {
                'success': True,
                'message': '文档添加成功',
                'document_info': added_doc_info
            }
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return {
                'success': False,
                'message': f'添加文档失败: {str(e)}',
                'document_info': None
            }
    
    def delete_document_by_source(self, source_file: str) -> Dict[str, Any]:
        """
        根据源文件名删除文档（仅支持ChromaDB）
        """
        if config.VECTOR_STORE_TYPE != "chroma":
            return {
                'success': False,
                'message': '文档删除仅支持ChromaDB',
            }
        
        if not self.vector_store or not isinstance(self.vector_store, Chroma):
            return {
                'success': False,
                'message': '未找到向量存储',
            }
            
        try:
            # 获取所有文档
            collection = self.vector_store._collection
            results = collection.get()
            
            # 找到需要删除的文档ID
            ids_to_delete = []
            for i, metadata in enumerate(results['metadatas']):
                if metadata and metadata.get('source_file') == source_file:
                    ids_to_delete.append(results['ids'][i])
            
            if not ids_to_delete:
                return {
                    'success': False,
                    'message': f'未找到源文件为 {source_file} 的文档',
                }
            
            # 删除文档
            collection.delete(ids=ids_to_delete)
            
            logger.info(f"已删除 {len(ids_to_delete)} 个来自 {source_file} 的文档块")
            
            return {
                'success': True,
                'message': f'成功删除 {len(ids_to_delete)} 个文档块',
                'deleted_count': len(ids_to_delete)
            }
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return {
                'success': False,
                'message': f'删除文档失败: {str(e)}',
            }
    
    def list_documents(self) -> Dict[str, Any]:
        """
        列出向量存储中的所有文档
        """
        if not self.vector_store:
            return {
                'success': False,
                'message': '未找到向量存储',
                'documents': []
            }
            
        try:
            if isinstance(self.vector_store, Chroma):
                collection = self.vector_store._collection
                results = collection.get()
                
                # 统计每个源文件的文档数量
                source_files = {}
                for metadata in results['metadatas']:
                    if metadata:
                        # 获取源文件名，优先使用source_file，其次使用source
                        source_file = metadata.get('source_file') or metadata.get('source')
                        if source_file:
                            # 提取文件名（去除路径）
                            if '/' in source_file:
                                source_file = source_file.split('/')[-1]
                            elif '\\' in source_file:
                                source_file = source_file.split('\\')[-1]
                            
                            if source_file not in source_files:
                                source_files[source_file] = {
                                    'chunks_count': 0,
                                    'source_file': source_file,
                                    'added_at': metadata.get('added_at')
                                }
                            source_files[source_file]['chunks_count'] += 1
                
                documents_list = list(source_files.values())
                
            elif isinstance(self.vector_store, FAISS):
                # FAISS不支持直接查询元数据，返回当前文档信息
                documents_list = [self.document_info] if self.document_info else []
            else:
                documents_list = []
            
            return {
                'success': True,
                'message': f'共找到 {len(documents_list)} 个文档',
                'documents': documents_list
            }
            
        except Exception as e:
            logger.error(f"列出文档失败: {e}")
            return {
                'success': False,
                'message': f'列出文档失败: {str(e)}',
                'documents': []
            }
    
    def reset(self) -> None:
        """重置API状态"""
        self.qa_chain = None
        self.current_document = None
        self.document_info = {}
        self.vector_store = None
        self.embeddings = None
        logger.info("API状态已重置")