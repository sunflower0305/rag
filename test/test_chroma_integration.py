#!/usr/bin/env python3
"""
测试ChromaDB集成的简单脚本
"""
import os
from dotenv import load_dotenv
from custom_qwen_embeddings import CustomQwenEmbeddings
from langchain_chroma import Chroma
import tempfile
import shutil

# 加载环境变量
load_dotenv()

def test_chroma_basic():
    """测试基本的ChromaDB功能"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        return False
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    
    try:
        print("=== ChromaDB基础功能测试 ===")
        
        # 创建嵌入对象
        embeddings = CustomQwenEmbeddings(
            api_key=api_key,
            model="text-embedding-v4"
        )
        
        # 创建ChromaDB实例
        print("1. 创建ChromaDB实例...")
        chroma_db = Chroma(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=embeddings
        )
        
        # 添加测试文档
        print("2. 添加测试文档...")
        test_texts = [
            "这是第一个测试文档，内容关于人工智能。",
            "这是第二个测试文档，讨论机器学习算法。", 
            "这是第三个测试文档，介绍深度学习技术。"
        ]
        
        test_metadatas = [
            {"source": "doc1", "type": "ai"},
            {"source": "doc2", "type": "ml"},
            {"source": "doc3", "type": "dl"}
        ]
        
        chroma_db.add_texts(
            texts=test_texts,
            metadatas=test_metadatas,
            ids=["doc1", "doc2", "doc3"]
        )
        
        # 测试相似性搜索
        print("3. 测试相似性搜索...")
        query = "人工智能和机器学习"
        results = chroma_db.similarity_search(query, k=2)
        
        print(f"查询: {query}")
        print(f"找到 {len(results)} 个相关文档:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content[:50]}...")
            print(f"     元数据: {doc.metadata}")
        
        # 测试文档计数
        print("4. 测试文档计数...")
        collection = chroma_db._collection
        count = collection.count()
        print(f"集合中的文档数量: {count}")
        
        # 测试持久化后重新加载
        print("5. 测试持久化和重新加载...")
        del chroma_db  # 删除当前实例
        
        # 重新创建实例，应该能加载之前的数据
        chroma_db_reload = Chroma(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=embeddings
        )
        
        collection_reload = chroma_db_reload._collection
        count_reload = collection_reload.count()
        print(f"重新加载后的文档数量: {count_reload}")
        
        if count == count_reload:
            print("✅ ChromaDB持久化测试通过")
        else:
            print("❌ ChromaDB持久化测试失败")
            return False
            
        print("✅ ChromaDB基础功能测试全部通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    success = test_chroma_basic()
    if success:
        print("\n🎉 ChromaDB集成测试成功!")
    else:
        print("\n💥 ChromaDB集成测试失败!")
        exit(1)