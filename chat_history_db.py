import sqlite3
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ChatHistoryDB:
    """聊天历史数据库管理类"""
    
    def __init__(self, db_path: str = "sqlite/chat_history.db"):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """初始化数据库表结构"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建会话表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        session_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        document_info TEXT,  -- JSON格式存储文档信息
                        vector_store_type TEXT
                    )
                """)
                
                # 创建消息表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,  -- 'user' 或 'assistant'
                        content TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processing_time REAL,  -- 处理时间（秒）
                        document_sources TEXT,  -- JSON格式存储相关文档来源
                        FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
                    )
                """)
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON chat_messages(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON chat_sessions(updated_at)")
                
                conn.commit()
                logger.info("数据库初始化完成")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def create_session(self, session_name: str = None, document_info: Dict = None, vector_store_type: str = "chroma") -> str:
        """
        创建新的聊天会话
        
        Args:
            session_name: 会话名称
            document_info: 文档信息
            vector_store_type: 向量存储类型
            
        Returns:
            session_id: 新创建的会话ID
        """
        session_id = f"session_{int(time.time() * 1000)}"
        
        if not session_name:
            if document_info and 'file_name' in document_info:
                session_name = f"关于 {document_info['file_name']} 的对话"
            else:
                session_name = f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_sessions (session_id, session_name, document_info, vector_store_type)
                    VALUES (?, ?, ?, ?)
                """, (session_id, session_name, json.dumps(document_info) if document_info else None, vector_store_type))
                conn.commit()
                
                logger.info(f"创建新会话: {session_id} - {session_name}")
                return session_id
                
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            raise
    
    def add_message(self, session_id: str, role: str, content: str, processing_time: float = None, 
                   document_sources: List[str] = None) -> int:
        """
        添加消息到会话
        
        Args:
            session_id: 会话ID
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
            processing_time: 处理时间
            document_sources: 相关文档来源
            
        Returns:
            message_id: 新添加的消息ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 添加消息
                cursor.execute("""
                    INSERT INTO chat_messages (session_id, role, content, processing_time, document_sources)
                    VALUES (?, ?, ?, ?, ?)
                """, (session_id, role, content, processing_time, 
                     json.dumps(document_sources) if document_sources else None))
                
                message_id = cursor.lastrowid
                
                # 更新会话的最后更新时间
                cursor.execute("""
                    UPDATE chat_sessions 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
                logger.info(f"添加消息到会话 {session_id}: {role} - {len(content)} 字符")
                return message_id
                
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            raise
    
    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取会话的所有消息
        
        Args:
            session_id: 会话ID
            limit: 消息数量限制
            
        Returns:
            messages: 消息列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # 使返回结果可以像字典一样访问
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM chat_messages 
                    WHERE session_id = ? 
                    ORDER BY timestamp 
                    LIMIT ?
                """, (session_id, limit))
                
                messages = []
                for row in cursor.fetchall():
                    message = {
                        'id': row['id'],
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'],
                        'processing_time': row['processing_time'],
                        'document_sources': json.loads(row['document_sources']) if row['document_sources'] else None
                    }
                    messages.append(message)
                
                return messages
                
        except Exception as e:
            logger.error(f"获取会话消息失败: {e}")
            return []
    
    def get_recent_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取最近的会话列表
        
        Args:
            limit: 会话数量限制
            
        Returns:
            sessions: 会话列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.*, COUNT(m.id) as message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    session = {
                        'session_id': row['session_id'],
                        'session_name': row['session_name'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'document_info': json.loads(row['document_info']) if row['document_info'] else None,
                        'vector_store_type': row['vector_store_type'],
                        'message_count': row['message_count']
                    }
                    sessions.append(session)
                
                return sessions
                
        except Exception as e:
            logger.error(f"获取会话列表失败: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话及其所有消息
        
        Args:
            session_id: 会话ID
            
        Returns:
            success: 是否删除成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 删除消息
                cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                
                # 删除会话
                cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
                
                conn.commit()
                logger.info(f"删除会话: {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            return False
    
    def search_messages(self, query: str, session_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        搜索消息内容
        
        Args:
            query: 搜索关键词
            session_id: 指定会话ID（可选）
            limit: 结果数量限制
            
        Returns:
            messages: 匹配的消息列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute("""
                        SELECT m.*, s.session_name
                        FROM chat_messages m
                        JOIN chat_sessions s ON m.session_id = s.session_id
                        WHERE m.session_id = ? AND m.content LIKE ?
                        ORDER BY m.timestamp DESC
                        LIMIT ?
                    """, (session_id, f'%{query}%', limit))
                else:
                    cursor.execute("""
                        SELECT m.*, s.session_name
                        FROM chat_messages m
                        JOIN chat_sessions s ON m.session_id = s.session_id
                        WHERE m.content LIKE ?
                        ORDER BY m.timestamp DESC
                        LIMIT ?
                    """, (f'%{query}%', limit))
                
                messages = []
                for row in cursor.fetchall():
                    message = {
                        'id': row['id'],
                        'session_id': row['session_id'],
                        'session_name': row['session_name'],
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'],
                        'processing_time': row['processing_time'],
                        'document_sources': json.loads(row['document_sources']) if row['document_sources'] else None
                    }
                    messages.append(message)
                
                return messages
                
        except Exception as e:
            logger.error(f"搜索消息失败: {e}")
            return []
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            stats: 统计信息
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 总会话数
                cursor.execute("SELECT COUNT(*) FROM chat_sessions")
                total_sessions = cursor.fetchone()[0]
                
                # 总消息数
                cursor.execute("SELECT COUNT(*) FROM chat_messages")
                total_messages = cursor.fetchone()[0]
                
                # 最近7天的会话数
                cursor.execute("""
                    SELECT COUNT(*) FROM chat_sessions 
                    WHERE created_at >= datetime('now', '-7 days')
                """)
                recent_sessions = cursor.fetchone()[0]
                
                # 平均每个会话的消息数
                cursor.execute("""
                    SELECT AVG(message_count) FROM (
                        SELECT COUNT(*) as message_count 
                        FROM chat_messages 
                        GROUP BY session_id
                    )
                """)
                avg_messages_per_session = cursor.fetchone()[0] or 0
                
                return {
                    'total_sessions': total_sessions,
                    'total_messages': total_messages,
                    'recent_sessions_7days': recent_sessions,
                    'avg_messages_per_session': round(avg_messages_per_session, 2)
                }
                
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def export_session(self, session_id: str, format: str = 'json') -> Optional[str]:
        """
        导出会话数据
        
        Args:
            session_id: 会话ID
            format: 导出格式 ('json' 或 'txt')
            
        Returns:
            exported_data: 导出的数据字符串
        """
        try:
            # 获取会话信息
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
                session = cursor.fetchone()
                
                if not session:
                    return None
                
                messages = self.get_session_messages(session_id)
                
                if format == 'json':
                    export_data = {
                        'session_info': dict(session),
                        'messages': messages,
                        'exported_at': datetime.now().isoformat()
                    }
                    return json.dumps(export_data, ensure_ascii=False, indent=2)
                
                elif format == 'txt':
                    lines = [
                        f"会话名称: {session['session_name']}",
                        f"创建时间: {session['created_at']}",
                        f"文档信息: {session['document_info']}",
                        f"向量存储: {session['vector_store_type']}",
                        "-" * 50,
                        ""
                    ]
                    
                    for msg in messages:
                        timestamp = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                        role_name = "用户" if msg['role'] == 'user' else "助手"
                        lines.append(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {role_name}:")
                        lines.append(msg['content'])
                        if msg['processing_time']:
                            lines.append(f"(处理时间: {msg['processing_time']:.2f}秒)")
                        lines.append("")
                    
                    return "\n".join(lines)
                
        except Exception as e:
            logger.error(f"导出会话失败: {e}")
            return None