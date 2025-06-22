import gradio as gr
import tempfile
import os
import time
from typing import List, Tuple, Optional
from qianwen_paper_qa_api import QianwenPaperQAAPI, config
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
    
    def upload_documents(self, files, is_first_upload=True) -> Tuple[str, str, bool, str]:
        """上传并处理多个PDF文档"""
        if not self.api:
            return "❌ 请先输入API密钥并初始化", "", False, ""
        
        if not files:
            return "❌ 请选择PDF文件", "", False, ""
        
        processed_docs = []
        failed_docs = []
        
        try:
            for i, file in enumerate(files):
                try:
                    # 处理单个文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        with open(file, 'rb') as f:
                            tmp_file.write(f.read())
                        tmp_file_path = tmp_file.name
                    
                    # 对于第一个文档或单独上传，使用process_document
                    # 对于后续文档，使用add_document（仅ChromaDB支持）
                    if i == 0 and is_first_upload:
                        result = self.api.process_document(tmp_file_path)
                    elif self.vector_store_type == "chroma" and not is_first_upload:
                        result = self.api.add_document(tmp_file_path)
                    else:
                        result = self.api.process_document(tmp_file_path)
                    
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
            result = self.api.get_document_summary()
            
            if result['success']:
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
            result = self.api.ask_question(question.strip())
            
            if result['success']:
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
    
    def add_single_document(self, file) -> Tuple[str, str]:
        """添加单个文档（仅ChromaDB支持）"""
        if not self.api:
            return "❌ 请先初始化API", ""
        
        if self.vector_store_type != "chroma":
            return "❌ 动态添加文档仅支持ChromaDB", ""
        
        if not file:
            return "❌ 请选择PDF文件", ""
        
        try:
            # 处理文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                with open(file, 'rb') as f:
                    tmp_file.write(f.read())
                tmp_file_path = tmp_file.name
            
            # 添加文档
            result = self.api.add_document(tmp_file_path)
            
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            if result['success']:
                doc_info = result['document_info']
                # 更新文档列表
                doc_list = self.get_document_list()
                return f"✅ 成功添加文档: {doc_info['file_name']} ({doc_info['chunks_count']} 个文本块)", doc_list
            else:
                return f"❌ 添加文档失败: {result['message']}", ""
                
        except Exception as e:
            try:
                if 'tmp_file_path' in locals():
                    os.unlink(tmp_file_path)
            except:
                pass
            return f"❌ 添加文档失败: {str(e)}", ""
    
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
            
            **🎆 新功能：**
            - ✅ ChromaDB支持：动态添加/删除文档
            - ✅ 多文档管理：同时处理多个PDF
            - ✅ 实时文档列表：查看已加载文档
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
                    
                    # 文档上传区域
                    gr.Markdown("### 📄 文档上传")
                    
                    file_upload = gr.File(
                        label="上传PDF文档（初始化）",
                        file_types=[".pdf"],
                        file_count="multiple",
                        visible=False
                    )
                    
                    upload_status = gr.Textbox(label="上传状态", interactive=False)
                    
                    # 文档信息显示
                    document_info = gr.Markdown(
                        label="文档信息",
                        visible=False,
                        elem_classes=["document-info"]
                    )
                    
                    # 动态文档管理区域（仅ChromaDB）
                    with gr.Group(visible=False) as dynamic_docs_group:
                        gr.Markdown("### 🔄 动态文档管理 (ChromaDB)")
                        
                        # 添加文档
                        add_file_upload = gr.File(
                            label="添加新PDF文档",
                            file_types=[".pdf"],
                            file_count="single"
                        )
                        add_doc_status = gr.Textbox(label="添加状态", interactive=False)
                        
                        # 删除文档
                        with gr.Row():
                            delete_filename = gr.Textbox(
                                label="要删除的文件名",
                                placeholder="例如: document.pdf",
                                scale=3
                            )
                            delete_btn = gr.Button("🗑️ 删除文档", variant="stop", scale=1)
                        
                        delete_status = gr.Textbox(label="删除状态", interactive=False)
                    
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
                        height=500,
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
            
            # 示例问题
            gr.Markdown("""
            ### 💡 示例问题
            **单文档问题：**
            - 这篇文档的主要研究内容是什么？
            - 文档中提到了哪些关键技术或方法？
            - 有什么重要的结论或发现？
            
            **多文档问题（ChromaDB）：**
            - 这些文档有什么共同点或区别？
            - 请比较不同文档中的观点
            - 总结所有文档的核心内容
            """)
            
            # 事件绑定
            
            # API初始化
            def update_file_visibility(api_key, vector_store_type):
                status, visible = self.initialize_api(api_key, vector_store_type)
                dynamic_visible = visible and vector_store_type == "chroma"
                return status, gr.update(visible=visible), gr.update(visible=dynamic_visible)
            
            init_btn.click(
                fn=update_file_visibility,
                inputs=[api_key_input, vector_store_choice],
                outputs=[api_status, file_upload, dynamic_docs_group],
                show_progress=True
            )
            
            # 文档上传
            def update_upload_status(files):
                status, info, visible, doc_list = self.upload_documents(files, is_first_upload=True)
                return status, info, gr.update(visible=visible), gr.update(value=doc_list, visible=visible)
            
            file_upload.upload(
                fn=update_upload_status,
                inputs=[file_upload],
                outputs=[upload_status, document_info, summary_btn, document_list],
                show_progress=True
            )
            
            # 添加单个文档
            def handle_add_document(file):
                status, doc_list = self.add_single_document(file)
                return status, doc_list
            
            add_file_upload.upload(
                fn=handle_add_document,
                inputs=[add_file_upload],
                outputs=[add_doc_status, document_list],
                show_progress=True
            )
            
            # 删除文档
            def handle_delete_document(filename):
                status, doc_list = self.delete_document(filename)
                return status, doc_list, ""  # 清空输入框
            
            delete_btn.click(
                fn=handle_delete_document,
                inputs=[delete_filename],
                outputs=[delete_status, document_list, delete_filename],
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