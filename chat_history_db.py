import sqlite3
import json
import time
import base64
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
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
        
        # 初始化加密密钥
        self._encryption_key = self._get_encryption_key()
        self._cipher = Fernet(self._encryption_key)
        
        self.init_database()
    
    def _get_encryption_key(self) -> bytes:
        """获取或生成加密密钥"""
        key_file = self.db_path.parent / "encryption.key"
        
        if key_file.exists():
            # 读取现有密钥
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # 生成新密钥
            password = os.environ.get('DB_ENCRYPTION_PASSWORD', 'default_password_change_in_production').encode()
            salt = os.urandom(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # 保存密钥和盐
            with open(key_file, 'wb') as f:
                f.write(key)
            
            with open(self.db_path.parent / "salt.key", 'wb') as f:
                f.write(salt)
            
            logger.info("生成新的数据库加密密钥")
            return key
    
    def _encrypt_api_key(self, api_key: str) -> str:
        """加密API密钥"""
        if not api_key:
            return ""
        encrypted = self._cipher.encrypt(api_key.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def _decrypt_api_key(self, encrypted_api_key: str) -> str:
        """解密API密钥"""
        if not encrypted_api_key:
            return ""
        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_api_key.encode())
            decrypted = self._cipher.decrypt(encrypted_data)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"解密API密钥失败: {e}")
            return ""
    
    def init_database(self):
        """初始化数据库表结构"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建用户表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        name TEXT,
                        email TEXT,
                        avatar_url TEXT,
                        dashscope_api_key TEXT,  -- 用户专属的DashScope API密钥（加密存储）
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 创建会话表（添加user_id外键）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id INTEGER,  -- 关联用户ID，NULL表示匿名用户
                        session_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        document_info TEXT,  -- JSON格式存储文档信息
                        vector_store_type TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
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
                
                # 检查是否需要迁移现有数据（添加user_id列）
                cursor.execute("PRAGMA table_info(chat_sessions)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'user_id' not in columns:
                    logger.info("检测到旧版数据库，正在迁移...")
                    cursor.execute("ALTER TABLE chat_sessions ADD COLUMN user_id INTEGER")
                    logger.info("数据库迁移完成")
                
                # 检查用户表是否需要添加API密钥列
                cursor.execute("PRAGMA table_info(users)")
                user_columns = [column[1] for column in cursor.fetchall()]
                if 'dashscope_api_key' not in user_columns:
                    logger.info("为用户表添加API密钥列...")
                    cursor.execute("ALTER TABLE users ADD COLUMN dashscope_api_key TEXT")
                    logger.info("用户表迁移完成")
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON chat_messages(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON chat_sessions(updated_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
                
                # 只有在user_id列存在时才创建相关索引
                cursor.execute("PRAGMA table_info(chat_sessions)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'user_id' in columns:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id)")
                
                conn.commit()
                logger.info("数据库初始化完成")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def upsert_user(self, user_id: int, username: str, name: str = None, email: str = None, avatar_url: str = None, dashscope_api_key: str = None) -> bool:
        """
        插入或更新用户信息
        
        Args:
            user_id: GitHub用户ID
            username: 用户名
            name: 显示名称
            email: 邮箱
            avatar_url: 头像URL
            dashscope_api_key: DashScope API密钥（可选，仅在更新时传入）
            
        Returns:
            success: 是否操作成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if dashscope_api_key is not None:
                    # 更新包含API密钥
                    encrypted_key = self._encrypt_api_key(dashscope_api_key)
                    cursor.execute("""
                        INSERT OR REPLACE INTO users (user_id, username, name, email, avatar_url, dashscope_api_key, last_login_at)
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (user_id, username, name, email, avatar_url, encrypted_key))
                    logger.info(f"更新用户信息和API密钥: {username} (ID: {user_id})")
                else:
                    # 只更新基本信息，保留现有API密钥
                    cursor.execute("""
                        INSERT OR IGNORE INTO users (user_id, username, name, email, avatar_url, last_login_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (user_id, username, name, email, avatar_url))
                    
                    # 更新现有用户的基本信息（不覆盖API密钥）
                    cursor.execute("""
                        UPDATE users 
                        SET name = ?, email = ?, avatar_url = ?, last_login_at = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                    """, (name, email, avatar_url, user_id))
                    logger.info(f"更新用户基本信息: {username} (ID: {user_id})")
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"更新用户信息失败: {e}")
            return False
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            user: 用户信息字典
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                if row:
                    user_info = dict(row)
                    # 解密API密钥
                    if user_info.get('dashscope_api_key'):
                        user_info['dashscope_api_key'] = self._decrypt_api_key(user_info['dashscope_api_key'])
                    return user_info
                return None
                
        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            return None
    
    def get_user_api_key(self, user_id: int) -> Optional[str]:
        """
        获取用户的DashScope API密钥
        
        Args:
            user_id: 用户ID
            
        Returns:
            api_key: 解密后的API密钥
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT dashscope_api_key FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                if row and row[0]:
                    return self._decrypt_api_key(row[0])
                return None
                
        except Exception as e:
            logger.error(f"获取用户API密钥失败: {e}")
            return None
    
    def update_user_api_key(self, user_id: int, api_key: str) -> bool:
        """
        更新用户的DashScope API密钥
        
        Args:
            user_id: 用户ID
            api_key: 新的API密钥
            
        Returns:
            success: 是否更新成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                encrypted_key = self._encrypt_api_key(api_key)
                cursor.execute("""
                    UPDATE users 
                    SET dashscope_api_key = ? 
                    WHERE user_id = ?
                """, (encrypted_key, user_id))
                
                conn.commit()
                logger.info(f"更新用户API密钥: 用户ID {user_id}")
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"更新用户API密钥失败: {e}")
            return False

    def create_session(self, session_name: str = None, document_info: Dict = None, vector_store_type: str = "chroma", user_id: int = None) -> str:
        """
        创建新的聊天会话
        
        Args:
            session_name: 会话名称
            document_info: 文档信息
            vector_store_type: 向量存储类型
            user_id: 用户ID（可选，NULL表示匿名用户）
            
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
                    INSERT INTO chat_sessions (session_id, user_id, session_name, document_info, vector_store_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (session_id, user_id, session_name, json.dumps(document_info) if document_info else None, vector_store_type))
                conn.commit()
                
                user_info = f" (用户ID: {user_id})" if user_id else ""
                logger.info(f"创建新会话: {session_id} - {session_name}{user_info}")
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
    
    def get_recent_sessions(self, limit: int = 20, user_id: int = None) -> List[Dict[str, Any]]:
        """
        获取最近的会话列表
        
        Args:
            limit: 会话数量限制
            user_id: 用户ID过滤（可选，None表示获取所有用户的会话）
            
        Returns:
            sessions: 会话列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if user_id is not None:
                    # 获取指定用户的会话
                    cursor.execute("""
                        SELECT s.*, u.username, u.name as user_name, COUNT(m.id) as message_count
                        FROM chat_sessions s
                        LEFT JOIN users u ON s.user_id = u.user_id
                        LEFT JOIN chat_messages m ON s.session_id = m.session_id
                        WHERE s.user_id = ?
                        GROUP BY s.session_id
                        ORDER BY s.updated_at DESC
                        LIMIT ?
                    """, (user_id, limit))
                else:
                    # 获取所有会话（包括匿名用户）
                    cursor.execute("""
                        SELECT s.*, u.username, u.name as user_name, COUNT(m.id) as message_count
                        FROM chat_sessions s
                        LEFT JOIN users u ON s.user_id = u.user_id
                        LEFT JOIN chat_messages m ON s.session_id = m.session_id
                        GROUP BY s.session_id
                        ORDER BY s.updated_at DESC
                        LIMIT ?
                    """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    session = {
                        'session_id': row['session_id'],
                        'user_id': row['user_id'],
                        'username': row['username'],
                        'user_name': row['user_name'],
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
    
    def search_messages(self, query: str, session_id: str = None, user_id: int = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        搜索消息内容
        
        Args:
            query: 搜索关键词
            session_id: 指定会话ID（可选）
            user_id: 指定用户ID（可选）
            limit: 结果数量限制
            
        Returns:
            messages: 匹配的消息列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if session_id:
                    # 搜索指定会话
                    cursor.execute("""
                        SELECT m.*, s.session_name, s.user_id
                        FROM chat_messages m
                        JOIN chat_sessions s ON m.session_id = s.session_id
                        WHERE m.session_id = ? AND m.content LIKE ?
                        ORDER BY m.timestamp DESC
                        LIMIT ?
                    """, (session_id, f'%{query}%', limit))
                elif user_id is not None:
                    # 搜索指定用户的所有会话
                    cursor.execute("""
                        SELECT m.*, s.session_name, s.user_id
                        FROM chat_messages m
                        JOIN chat_sessions s ON m.session_id = s.session_id
                        WHERE s.user_id = ? AND m.content LIKE ?
                        ORDER BY m.timestamp DESC
                        LIMIT ?
                    """, (user_id, f'%{query}%', limit))
                else:
                    # 搜索所有消息
                    cursor.execute("""
                        SELECT m.*, s.session_name, s.user_id
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
                        'user_id': row['user_id'],
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