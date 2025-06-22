import gradio as gr
import tempfile
import os
import time
from typing import List, Tuple, Optional
from qianwen_paper_qa_api import QianwenPaperQAAPI, config
from chat_history_db import ChatHistoryDB
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioRAGApp:
    """Gradio RAG应用类"""
    
    def __init__(self):
        self.api = None
        self.chat_history = []
        self.documents = {}  # 存储多个文档 {doc_id: document_info}
        self.current_doc_id = None
        self.vector_store_type = "chroma"  # 默认使用ChromaDB
        self.db = ChatHistoryDB()  # 初始化数据库
        self.current_session_id = None  # 当前会话ID
        self._session_option_map = {}  # 存储选项到session_id的映射
    
    def initialize_api(self, api_key: str, vector_store_type: str) -> Tuple[str, bool]:
        """初始化API"""
        if not api_key or not api_key.strip():
            return "❌ 请输入有效的API密钥", False
        
        try:
            # 设置向量存储类型
            config.VECTOR_STORE_TYPE = vector_store_type
            self.vector_store_type = vector_store_type
            
            self.api = QianwenPaperQAAPI(api_key=api_key.strip())
            return f"✅ API初始化成功（使用{vector_store_type.upper()}），可以上传PDF文档了", True
        except Exception as e:
            return f"❌ API初始化失败: {str(e)}", False
    
    def upload_documents(self, files) -> Tuple[str, str, bool, str]:
        """上传并处理多个PDF文档"""
        if not self.api:
            return "❌ 请先输入API密钥并初始化", "", False, ""
        
        if not files:
            return "❌ 请选择PDF文件", "", False, ""
        
        processed_docs = []
        failed_docs = []
        
        # 判断是首次上传还是追加上传
        is_first_upload = not self.documents  # 如果没有现有文档，则为首次上传
        
        try:
            for i, file in enumerate(files):
                try:
                    # 获取原始文件名
                    original_filename = os.path.basename(file)
                    
                    # 处理单个文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        with open(file, 'rb') as f:
                            tmp_file.write(f.read())
                        tmp_file_path = tmp_file.name
                    
                    # 智能选择处理方式
                    if i == 0 and is_first_upload:
                        # 第一个文档且是首次上传：初始化向量存储
                        result = self.api.process_document(tmp_file_path, original_filename)
                    elif self.vector_store_type == "chroma":
                        # ChromaDB模式：添加到现有向量存储
                        result = self.api.add_document(tmp_file_path, original_filename)
                    elif self.vector_store_type == "faiss" and not is_first_upload:
                        # FAISS模式且不是首次上传：重置并重新初始化
                        failed_docs.append(f"{original_filename}: FAISS模式不支持追加文档，将重置文档库")
                        result = self.api.process_document(tmp_file_path, original_filename)
                        # 清空现有文档记录，因为FAISS会重置
                        self.documents = {}
                    else:
                        # FAISS模式批量上传多个文档：跳过后续文档
                        failed_docs.append(f"{original_filename}: FAISS模式不支持批量上传多个文档，请使用ChromaDB模式")
                        continue
                    
                    # 清理临时文件
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                    
                    if result['success']:
                        doc_id = f"doc_{len(self.documents)}"
                        self.documents[doc_id] = result['document_info']
                        processed_docs.append(result['document_info']['file_name'])
                        if not self.current_doc_id:  # 设置第一个为当前文档
                            self.current_doc_id = doc_id
                    else:
                        failed_docs.append(f"{os.path.basename(file)}: {result['message']}")
                        
                except Exception as e:
                    failed_docs.append(f"{os.path.basename(file)}: {str(e)}")
            
            # 生成结果信息
            if processed_docs:
                if is_first_upload:
                    self.chat_history = []  # 重置聊天历史
                    # 创建新的会话
                    doc_info = self.documents[list(self.documents.keys())[0]] if self.documents else None
                    self.current_session_id = self.db.create_session(
                        document_info=doc_info,
                        vector_store_type=self.vector_store_type
                    )
                    
                info_text = f"""
📄 **文档处理结果**
✅ 成功处理: {len(processed_docs)}个文档
{chr(10).join([f"- {name}" for name in processed_docs])}
"""
                if failed_docs:
                    info_text += f"\n❌ 失败: {len(failed_docs)}个文档\n{chr(10).join([f"- {name}" for name in failed_docs])}"
                
                # 获取文档列表
                doc_list = self.get_document_list()
                
                return f"✅ 成功处理 {len(processed_docs)} 个文档", info_text, True, doc_list
            else:
                return f"❌ 所有文档处理失败", "", False, ""
                
        except Exception as e:
            # 清理临时文件
            try:
                if 'tmp_file_path' in locals():
                    os.unlink(tmp_file_path)
            except:
                pass
            return f"❌ 批量处理失败: {str(e)}", "", False, ""
    
    def get_document_summary(self) -> Tuple[List, str]:
        """获取文档摘要"""
        if not self.api:
            return self.chat_history, "❌ 请先初始化API"
        
        if not self.documents:
            return self.chat_history, "❌ 请先上传PDF文档"
        
        try:
            start_time = time.time()
            result = self.api.get_document_summary()
            processing_time = time.time() - start_time
            
            if result['success']:
                # 保存到数据库
                if self.current_session_id:
                    self.db.add_message(
                        session_id=self.current_session_id,
                        role="user",
                        content="请总结这篇文档的主要内容"
                    )
                    self.db.add_message(
                        session_id=self.current_session_id,
                        role="assistant",
                        content=result['answer'],
                        processing_time=processing_time
                    )
                
                # 添加到聊天历史
                self.chat_history.append({
                    "role": "user",
                    "content": "请总结这篇文档的主要内容"
                })
                self.chat_history.append({
                    "role": "assistant",
                    "content": result['answer']
                })
                return self.chat_history, ""
            else:
                return self.chat_history, f"❌ {result['message']}"
                
        except Exception as e:
            return self.chat_history, f"❌ 获取摘要失败: {str(e)}"
    
    def ask_question(self, question: str, history: List) -> Tuple[List, str]:
        """询问问题"""
        if not self.api:
            return history, "❌ 请先初始化API"
        
        if not self.documents:
            return history, "❌ 请先上传PDF文档"
        
        if not question or not question.strip():
            return history, "❌ 请输入问题"
        
        try:
            start_time = time.time()
            result = self.api.ask_question(question.strip())
            processing_time = time.time() - start_time
            
            if result['success']:
                # 保存到数据库
                if self.current_session_id:
                    # 保存用户问题
                    self.db.add_message(
                        session_id=self.current_session_id,
                        role="user",
                        content=question.strip()
                    )
                    # 保存助手回答
                    self.db.add_message(
                        session_id=self.current_session_id,
                        role="assistant",
                        content=result['answer'],
                        processing_time=processing_time
                    )
                
                # 更新聊天历史
                self.chat_history.append({
                    "role": "user",
                    "content": question.strip()
                })
                self.chat_history.append({
                    "role": "assistant",
                    "content": result['answer']
                })
                return self.chat_history, ""
            else:
                return history, f"❌ {result['message']}"
                
        except Exception as e:
            return history, f"❌ 询问失败: {str(e)}"
    
    def get_document_list(self) -> str:
        """获取文档列表"""
        if not self.api:
            return "未初始化API"
            
        try:
            result = self.api.list_documents()
            if result['success'] and result['documents']:
                doc_list = "📁 **当前文档列表:**\n\n"
                for i, doc in enumerate(result['documents'], 1):
                    source_file = doc.get('source_file', 'Unknown')
                    chunks_count = doc.get('chunks_count', 0)
                    doc_list += f"{i}. **{source_file}** ({chunks_count} 个文本块)\n"
                return doc_list
            else:
                return "📁 **当前文档列表:** 无文档"
        except Exception as e:
            return f"获取文档列表失败: {str(e)}"
    
    
    def delete_document(self, filename: str) -> Tuple[str, str]:
        """删除文档（仅ChromaDB支持）"""
        if not self.api:
            return "❌ 请先初始化API", ""
        
        if self.vector_store_type != "chroma":
            return "❌ 文档删除仅支持ChromaDB", ""
        
        if not filename or not filename.strip():
            return "❌ 请输入文件名", ""
        
        try:
            result = self.api.delete_document_by_source(filename.strip())
            
            if result['success']:
                # 更新文档列表
                doc_list = self.get_document_list()
                return f"✅ 成功删除文档: {filename} ({result['deleted_count']} 个文本块)", doc_list
            else:
                return f"❌ 删除文档失败: {result['message']}", ""
                
        except Exception as e:
            return f"❌ 删除文档失败: {str(e)}", ""
    
    def clear_chat(self) -> Tuple[List, str]:
        """清空聊天记录"""
        self.chat_history = []
        return [], ""
    
    def get_session_history(self, session_id: str) -> Tuple[List, str]:
        """加载指定会话的历史记录"""
        if not session_id:
            return [], "❌ 请选择一个会话"
        
        try:
            messages = self.db.get_session_messages(session_id)
            
            # 转换为Gradio格式
            history = []
            for msg in messages:
                history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            self.chat_history = history
            self.current_session_id = session_id
            
            return history, f"✅ 已加载会话历史（{len(messages)}条消息）"
            
        except Exception as e:
            return [], f"❌ 加载会话历史失败: {str(e)}"
    
    def get_sessions_display_text(self) -> str:
        """获取会话列表的显示文本"""
        try:
            sessions = self.db.get_recent_sessions(limit=20)
            if not sessions:
                return "📜 **历史会话：** 暂无历史记录"
            
            session_list = "📜 **最近会话：**\n\n"
            for i, session in enumerate(sessions):
                msg_count = session['message_count']
                updated_time = session['updated_at'][:16]  # 截取到分钟
                doc_name = "未知文档"
                if session['document_info']:
                    doc_info = session['document_info']
                    if isinstance(doc_info, str):
                        import json
                        try:
                            doc_info = json.loads(doc_info)
                        except:
                            pass
                    if isinstance(doc_info, dict) and 'file_name' in doc_info:
                        doc_name = doc_info['file_name']
                
                # 使用HTML样式来使其可点击
                session_list += f"""<div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 8px 0; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); cursor: pointer;" onclick="selectSession('{session['session_id']}', {i})">
🔸 **{session['session_name']}**  
📄 文档: {doc_name}  
💬 消息: {msg_count}条 | ⏰ {updated_time}
</div>

"""
            
            return session_list
            
        except Exception as e:
            logger.error(f"获取会话列表失败: {e}")
            return f"❌ 获取会话列表失败: {str(e)}"
    
    def get_sessions_for_radio(self) -> List[str]:
        """获取会话选项列表（用于Radio组件）"""
        try:
            sessions = self.db.get_recent_sessions(limit=20)
            if not sessions:
                return []
            
            session_options = []
            for session in sessions:
                msg_count = session['message_count']
                updated_time = session['updated_at'][:16]
                doc_name = "未知文档"
                if session['document_info']:
                    doc_info = session['document_info']
                    if isinstance(doc_info, str):
                        import json
                        try:
                            doc_info = json.loads(doc_info)
                        except:
                            pass
                    if isinstance(doc_info, dict) and 'file_name' in doc_info:
                        doc_name = doc_info['file_name']
                
                # 创建选项显示文本，同时在内部保存session_id的映射
                option_text = f"🔸 {session['session_name']} | 📄 {doc_name} | 💬 {msg_count}条 | ⏰ {updated_time}"
                session_options.append(option_text)
                
                # 保存选项到session_id的映射
                if not hasattr(self, '_session_option_map'):
                    self._session_option_map = {}
                self._session_option_map[option_text] = session['session_id']
            
            return session_options
            
        except Exception as e:
            logger.error(f"获取会话选项失败: {e}")
            return []
    
    def get_session_details(self, session_id: str) -> str:
        """获取会话详情显示"""
        if not session_id:
            return "请从上方列表中选择一个会话"
        
        try:
            sessions = self.db.get_recent_sessions(limit=100)
            session = next((s for s in sessions if s['session_id'] == session_id), None)
            
            if not session:
                return "❌ 会话不存在"
            
            doc_name = "未知文档"
            if session['document_info']:
                doc_info = session['document_info']
                if isinstance(doc_info, str):
                    import json
                    try:
                        doc_info = json.loads(doc_info)
                    except:
                        pass
                if isinstance(doc_info, dict) and 'file_name' in doc_info:
                    doc_name = doc_info['file_name']
            
            details = f"""
### 📋 会话详情

**🔸 会话名称：** {session['session_name']}  
**📄 关联文档：** {doc_name}  
**💬 消息数量：** {session['message_count']}条  
**🗂️ 存储模式：** {session['vector_store_type'].upper()}  
**⏰ 创建时间：** {session['created_at'][:19]}  
**🔄 更新时间：** {session['updated_at'][:19]}  
**🆔 会话ID：** `{session['session_id']}`

---
**💡 操作提示：**
- 点击 "📖 加载此会话" 继续对话
- 点击 "🗑️ 删除此会话" 永久删除
            """
            
            return details
            
        except Exception as e:
            return f"❌ 获取会话详情失败: {str(e)}"
    
    def delete_session_by_id(self, session_id: str) -> str:
        """删除指定会话"""
        if not session_id or not session_id.strip():
            return "❌ 请输入会话ID"
        
        try:
            success = self.db.delete_session(session_id.strip())
            if success:
                if self.current_session_id == session_id.strip():
                    self.current_session_id = None
                    self.chat_history = []
                return f"✅ 已删除会话: {session_id.strip()}"
            else:
                return f"❌ 删除失败，会话不存在: {session_id.strip()}"
                
        except Exception as e:
            return f"❌ 删除会话失败: {str(e)}"
    
    def search_chat_history(self, query: str) -> str:
        """搜索聊天历史"""
        if not query or not query.strip():
            return "❌ 请输入搜索关键词"
        
        try:
            messages = self.db.search_messages(query.strip(), limit=20)
            if not messages:
                return f"🔍 **搜索结果：** 未找到包含 '{query.strip()}' 的消息"
            
            result = f"🔍 **搜索结果：** 找到 {len(messages)} 条相关消息\n\n"
            
            for msg in messages:
                timestamp = msg['timestamp'][:16]
                role_name = "👤 用户" if msg['role'] == 'user' else "🤖 助手"
                content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                
                result += f"📍 **{msg['session_name']}** ({timestamp})\n"
                result += f"{role_name}: {content_preview}\n\n"
            
            return result
            
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"
    
    def create_interface(self):
        """创建Gradio界面"""
        
        # 自定义CSS样式
        css = """
        .gradio-container {
            max-width: 1400px !important;
        }
        .chat-container {
            height: 500px !important;
        }
        .document-info {
            background-color: var(--background-fill-secondary);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid var(--color-accent);
            color: var(--body-text-color);
        }
        .dark .document-info {
            background-color: rgba(255, 255, 255, 0.05);
            border-left-color: #4a9eff;
            color: #ffffff;
        }
        /* 确保Markdown内容在暗色模式下可见 */
        .dark .markdown {
            color: #ffffff !important;
        }
        .dark .markdown strong {
            color: #ffffff !important;
        }
        .dark .markdown h1, .dark .markdown h2, .dark .markdown h3 {
            color: #ffffff !important;
        }
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        """
        
        with gr.Blocks(css=css, title="千问RAG问答系统", theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🤖 千问RAG智能问答系统
            
            基于阿里云千问大模型的PDF文档问答系统，支持ChromaDB动态文档管理和FAISS传统模式。
            
            **📋 使用步骤：**
            1. 🔑 输入DashScope API密钥并选择向量存储类型
            2. 📄 上传PDF文档（支持批量上传）
            3. 💬 开始智能问答
            4. 📜 在右侧历史记录中查看和管理对话
            
            **🎆 功能特色：**
            - ✅ **统一文档管理**：一个界面处理所有文档操作
            - ✅ **ChromaDB动态管理**：随时添加/删除文档
            - ✅ **多文档批量处理**：同时上传多个PDF
            - ✅ **实时文档列表**：查看所有已加载文档
            - ✅ **SQLite历史记录**：自动保存所有问答历史，支持搜索和回顾
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # API密钥输入区域
                    gr.Markdown("### 🔑 API设置")
                    api_key_input = gr.Textbox(
                        label="DashScope API密钥",
                        type="password",
                        placeholder="请输入阿里云DashScope API密钥",
                        info="获取API密钥: https://dashscope.console.aliyun.com/"
                    )
                    
                    # 向量存储类型选择
                    vector_store_choice = gr.Radio(
                        choices=[("ChromaDB (推荐，支持动态更新)", "chroma"), ("FAISS (传统选项)", "faiss")],
                        value="chroma",
                        label="向量存储类型",
                        info="ChromaDB支持动态添加/删除文档"
                    )
                    
                    init_btn = gr.Button("初始化API", variant="primary")
                    api_status = gr.Textbox(label="API状态", interactive=False)
                    
                    # 文档管理区域
                    gr.Markdown("### 📄 文档管理")
                    
                    # 统一的文档上传区域
                    with gr.Group(visible=False) as document_management_group:
                        # 智能文档上传（支持单个或批量）
                        file_upload = gr.File(
                            label="📁 上传PDF文档（支持单个或多个文件）",
                            file_types=[".pdf"],
                            file_count="multiple"
                        )
                        
                        gr.Markdown("""
                        **📋 上传说明：**
                        - **首次上传**：可以选择一个或多个PDF文件
                        - **追加文档**：可以继续上传新文档（ChromaDB模式支持，FAISS模式会重置）
                        - **文件限制**：仅支持PDF格式，建议单文件不超过100MB
                        """)
                        
                        # 文档删除（ChromaDB）
                        with gr.Row(visible=False) as delete_row:
                            delete_filename = gr.Textbox(
                                label="要删除的文件名",
                                placeholder="例如: document.pdf",
                                scale=3
                            )
                            delete_btn = gr.Button("🗑️ 删除文档", variant="stop", scale=1)
                        
                        # 统一的状态显示
                        upload_status = gr.Textbox(label="操作状态", interactive=False)
                        
                        # 文档信息显示
                        document_info = gr.Markdown(
                            label="文档信息",
                            visible=False,
                            elem_classes=["document-info"]
                        )
                    
                    # 文档列表显示
                    document_list = gr.Markdown(
                        label="文档列表",
                        visible=False,
                        elem_classes=["document-info"]
                    )
                    
                    # 获取文档摘要按钮
                    summary_btn = gr.Button(
                        "📋 获取文档摘要", 
                        visible=False,
                        variant="secondary"
                    )
                
                with gr.Column(scale=2):
                    # 问答区域
                    gr.Markdown("### 💬 智能问答")
                    
                    chatbot = gr.Chatbot(
                        label="问答历史",
                        height=400,
                        elem_classes=["chat-container"],
                        type="messages"
                    )
                    
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="",
                            placeholder="请输入您的问题...",
                            scale=4
                        )
                        ask_btn = gr.Button("发送", variant="primary", scale=1)
                    
                    question_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                        visible=False
                    )
                    
                    with gr.Row():
                        clear_btn = gr.Button("清空对话", variant="secondary")
                
                with gr.Column(scale=1):
                    # 历史记录管理区域
                    gr.Markdown("### 📜 历史记录管理")
                    
                    # 初始化会话选项
                    initial_sessions_options = self.get_sessions_for_radio()
                    
                    # 会话列表（使用Radio显示）
                    sessions_radio = gr.Radio(
                        choices=initial_sessions_options,
                        value=None,
                        label="📋 历史会话列表（点击选择会话）",
                        info="选择会话将自动加载对话"
                    )
                    
                    # 选中的会话ID（隐藏组件，用于传递数据）
                    selected_session_id = gr.Textbox(
                        value="",
                        visible=False
                    )
                    
                    # 初始会话详情显示
                    initial_details_text = "请从上方列表中选择一个会话"
                    if initial_sessions_options:
                        # 从数据库获取统计信息
                        sessions_data = self.db.get_recent_sessions(limit=20)
                        if sessions_data:
                            initial_details_text = f"""
### 📊 历史记录统计

📋 **总会话数：** {len(sessions_data)}个  
🕒 **最新会话：** {sessions_data[0]['session_name']}  
📄 **最新文档：** {sessions_data[0].get('document_info', {}).get('file_name', '未知文档') if sessions_data[0].get('document_info') else '未知文档'}  

---
💡 **操作提示：** 点击上方列表中的会话选项将自动加载对话
                            """
                    
                    # 会话详情显示
                    session_details = gr.Markdown(
                        value=initial_details_text,
                        elem_classes=["document-info"],
                        label="会话详情"
                    )
                    
                    # 选中会话的操作区域（只在选中会话时显示）
                    with gr.Row(visible=False) as session_action_row:
                        gr.Markdown("✅ **会话已自动加载**")
                        delete_session_btn = gr.Button("🗑️ 删除此会话", variant="stop", size="sm")
                    
                    # 搜索功能
                    search_input = gr.Textbox(
                        placeholder="搜索聊天记录...",
                        show_label=False
                    )
                    
                    search_results = gr.Markdown(
                        value="",
                        visible=False,
                        elem_classes=["document-info"]
                    )
                    
                    # 操作状态
                    history_status = gr.Textbox(
                        label="操作状态",
                        interactive=False
                    )
            
            # 示例问题和使用说明
            gr.Markdown("""
            ### 💡 示例问题
            **📖 单文档分析：**
            - 这篇文档的主要研究内容是什么？
            - 文档中提到了哪些关键技术或方法？
            - 有什么重要的结论或发现？
            
            **🗂️ 多文档对比（ChromaDB模式）：**
            - 这些文档有什么共同点或区别？
            - 请比较不同文档中的观点
            - 总结所有文档的核心内容
            
            ### 📝 使用说明
            - **ChromaDB模式**：支持多次上传、动态添加/删除文档，推荐使用
            - **FAISS模式**：高性能检索，但只支持单次批量上传
            - **上传方式**：可以一次选择多个PDF文件，也可以分多次上传
            - **文档限制**：仅支持PDF格式，建议单文件不超过100MB
            - **历史管理**：自动加载历史会话，点击选择会话即可加载对话；输入关键词按回车搜索
            """)
            
            # 事件绑定
            
            # API初始化
            def update_file_visibility(api_key, vector_store_type):
                status, visible = self.initialize_api(api_key, vector_store_type)
                is_chroma = vector_store_type == "chroma"
                return (
                    status, 
                    gr.update(visible=visible),  # document_management_group
                    gr.update(visible=visible and is_chroma)   # delete_row
                )
            
            init_btn.click(
                fn=update_file_visibility,
                inputs=[api_key_input, vector_store_choice],
                outputs=[api_status, document_management_group, delete_row],
                show_progress=True
            )
            
            # 文档上传（智能处理）
            def update_upload_status(files):
                status, info, visible, doc_list = self.upload_documents(files)
                # 如果上传成功，自动刷新历史记录
                if visible:
                    options = self.get_sessions_for_radio()
                    # 更新会话统计
                    sessions_data = self.db.get_recent_sessions(limit=20)
                    if sessions_data:
                        updated_details = f"""
### 📊 历史记录统计

📋 **总会话数：** {len(sessions_data)}个  
🕒 **最新会话：** {sessions_data[0]['session_name']}  
📄 **最新文档：** {sessions_data[0].get('document_info', {}).get('file_name', '未知文档') if sessions_data[0].get('document_info') else '未知文档'}  

---
💡 **操作提示：** 点击上方列表中的会话选项将自动加载对话
                        """
                    else:
                        updated_details = "请从上方列表中选择一个会话"
                    
                    return (status, info, gr.update(visible=visible), gr.update(value=doc_list, visible=visible), 
                           gr.update(choices=options), updated_details, gr.update(visible=False))
                else:
                    return status, info, gr.update(visible=visible), gr.update(value=doc_list, visible=visible), gr.update(), gr.update(), gr.update()
            
            file_upload.upload(
                fn=update_upload_status,
                inputs=[file_upload],
                outputs=[upload_status, document_info, summary_btn, document_list, sessions_radio, session_details, session_action_row],
                show_progress=True
            )
            
            # 删除文档
            def handle_delete_document(filename):
                status, doc_list = self.delete_document(filename)
                return status, doc_list, ""  # 清空输入框
            
            delete_btn.click(
                fn=handle_delete_document,
                inputs=[delete_filename],
                outputs=[upload_status, document_list, delete_filename],
                show_progress=True
            )
            
            # 获取文档摘要
            summary_btn.click(
                fn=self.get_document_summary,
                inputs=[],
                outputs=[chatbot, question_status],
                show_progress=True
            )
            
            # 询问问题
            ask_btn.click(
                fn=self.ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_status],
                show_progress=True
            ).then(
                lambda: "",  # 清空输入框
                inputs=[],
                outputs=[question_input]
            )
            
            # 回车键发送问题
            question_input.submit(
                fn=self.ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_status],
                show_progress=True
            ).then(
                lambda: "",  # 清空输入框
                inputs=[],
                outputs=[question_input]
            )
            
            # 清空对话
            clear_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot, question_status]
            )
            
            # 历史记录管理事件绑定
            
            # 会话Radio选择事件（自动加载会话）
            def on_session_radio_change(selected_option):
                if selected_option and hasattr(self, '_session_option_map'):
                    try:
                        session_id = self._session_option_map.get(selected_option)
                        if session_id:
                            # 自动加载会话历史
                            history, load_status = self.get_session_history(session_id)
                            details = self.get_session_details(session_id)
                            
                            # 显示操作区域
                            return (
                                history,  # chatbot
                                details,  # session_details  
                                session_id,  # selected_session_id
                                gr.update(visible=True),  # session_action_row
                                f"✅ 已自动加载会话: {selected_option.split('|')[0].strip()}"  # history_status
                            )
                        else:
                            return [], "❌ 会话ID未找到", "", gr.update(visible=False), "❌ 会话ID未找到"
                    except Exception as e:
                        logger.error(f"选择会话失败: {e}")
                        return [], "❌ 选择会话失败", "", gr.update(visible=False), "❌ 选择会话失败"
                
                # 没有选择时隐藏操作区域
                return [], "请选择一个会话", "", gr.update(visible=False), ""
            
            sessions_radio.change(
                fn=on_session_radio_change,
                inputs=[sessions_radio],
                outputs=[chatbot, session_details, selected_session_id, session_action_row, history_status]
            )
            
            # 删除会话
            def handle_delete_session(session_id):
                if not session_id:
                    return "❌ 请先选择一个会话", gr.update(), "请从上方列表中选择一个会话", "", gr.update(visible=False)
                
                status = self.delete_session_by_id(session_id)
                # 刷新会话列表
                options = self.get_sessions_for_radio()
                return (
                    status,  # history_status
                    gr.update(choices=options, value=None),  # sessions_radio
                    "请从上方列表中选择一个会话",  # session_details
                    "",  # selected_session_id
                    gr.update(visible=False)  # session_action_row
                )
            
            delete_session_btn.click(
                fn=handle_delete_session,
                inputs=[selected_session_id],
                outputs=[history_status, sessions_radio, session_details, selected_session_id, session_action_row]
            )
            
            # 搜索历史（按回车键搜索）
            def handle_search(query):
                if query and query.strip():
                    result = self.search_chat_history(query)
                    return result, gr.update(visible=True)
                else:
                    return "", gr.update(visible=False)
            
            search_input.submit(
                fn=handle_search,
                inputs=[search_input],
                outputs=[search_results, search_results]
            )
        
        return app

def main():
    """主函数"""
    app_instance = GradioRAGApp()
    app = app_instance.create_interface()
    
    # 启动应用
    print("启动千问RAG问答系统...")
    print("注意: 如果localhost不可访问，将创建共享链接")
    
    try:
        # 首先尝试本地启动
        app.launch(
            share=False,
            inbrowser=False,
            debug=False,
            quiet=False
        )
    except Exception as e:
        print(f"本地启动失败: {e}")
        print("创建公共共享链接...")
        # 如果本地失败，使用共享链接
        app.launch(
            share=True,
            inbrowser=True,
            debug=False,
            quiet=False
        )

if __name__ == "__main__":
    main()