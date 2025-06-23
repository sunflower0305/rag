"""
GitHub OAuth 认证模块
支持 GitHub OAuth 2.0 认证流程和用户会话管理
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from fastapi import APIRouter, Request, HTTPException, Depends, Response
from fastapi.responses import RedirectResponse, HTMLResponse
from authlib.integrations.httpx_client import AsyncOAuth2Client
from jose import jwt, JWTError
import httpx
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logger = logging.getLogger(__name__)

# GitHub OAuth 配置
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24 * 7  # 7天过期

# OAuth URLs
GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"

class GitHubAuth:
    """GitHub OAuth 认证管理器"""
    
    def __init__(self, redirect_uri: str = None):
        self.client_id = GITHUB_CLIENT_ID
        self.client_secret = GITHUB_CLIENT_SECRET
        # 默认使用独立的OAuth服务端口
        self.redirect_uri = redirect_uri or "http://localhost:8001/auth/callback"
        
        if not self.client_id or not self.client_secret:
            logger.warning("GitHub OAuth 配置未完成，请设置 GITHUB_CLIENT_ID 和 GITHUB_CLIENT_SECRET 环境变量")
    
    def is_configured(self) -> bool:
        """检查是否已配置 GitHub OAuth"""
        return bool(self.client_id and self.client_secret)
    
    def get_authorization_url(self, state: str = None) -> str:
        """获取 GitHub 授权 URL"""
        if not self.is_configured():
            raise ValueError("GitHub OAuth 未配置")
        
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "user:email",
            "state": state
        }
        
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{GITHUB_AUTHORIZE_URL}?{param_string}", state
    
    async def exchange_code_for_token(self, code: str) -> Optional[str]:
        """使用授权码交换访问令牌"""
        if not self.is_configured():
            raise ValueError("GitHub OAuth 未配置")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    GITHUB_TOKEN_URL,
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "code": code,
                        "redirect_uri": self.redirect_uri
                    },
                    headers={"Accept": "application/json"}
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    return token_data.get("access_token")
                else:
                    logger.error(f"获取访问令牌失败: {response.status_code} - {response.text}")
                    return None
                    
            except Exception as e:
                logger.error(f"交换访问令牌时发生错误: {e}")
                return None
    
    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    GITHUB_USER_URL,
                    headers={
                        "Authorization": f"token {access_token}",
                        "Accept": "application/json"
                    }
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"获取用户信息失败: {response.status_code} - {response.text}")
                    return None
                    
            except Exception as e:
                logger.error(f"获取用户信息时发生错误: {e}")
                return None
    
    def create_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """创建 JWT 令牌"""
        payload = {
            "user_id": user_data["id"],
            "username": user_data["login"],
            "name": user_data.get("name", user_data["login"]),
            "email": user_data.get("email"),
            "avatar_url": user_data.get("avatar_url"),
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证 JWT 令牌"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except JWTError as e:
            logger.error(f"JWT 令牌验证失败: {e}")
            return None

# 全局认证实例
github_auth = GitHubAuth()

# 创建路由器
auth_router = APIRouter(prefix="/auth", tags=["authentication"])

# 会话存储（简单内存存储，生产环境建议使用 Redis）
session_store = {}

def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """获取当前用户信息（从cookie中的JWT令牌）"""
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    return github_auth.verify_jwt_token(token)

@auth_router.get("/github")
async def github_login(request: Request):
    """GitHub 登录入口"""
    if not github_auth.is_configured():
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>GitHub OAuth 未配置</h1>
                    <p>请在 .env 文件中设置以下环境变量：</p>
                    <ul>
                        <li>GITHUB_CLIENT_ID</li>
                        <li>GITHUB_CLIENT_SECRET</li>
                    </ul>
                    <p><a href="/">返回首页</a></p>
                </body>
            </html>
            """,
            status_code=400
        )
    
    try:
        # 设置回调URL（动态获取域名）
        host = request.headers.get("host", "localhost:7860")
        scheme = "https" if "localhost" not in host else "http"
        github_auth.redirect_uri = f"{scheme}://{host}/auth/callback"
        
        # 生成授权URL
        auth_url, state = github_auth.get_authorization_url()
        
        # 存储state用于验证
        session_store[state] = {"timestamp": datetime.utcnow()}
        
        response = RedirectResponse(url=auth_url)
        response.set_cookie(key="oauth_state", value=state, max_age=600)  # 10分钟过期
        return response
        
    except Exception as e:
        logger.error(f"GitHub 登录失败: {e}")
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>登录失败</h1>
                    <p>错误: {str(e)}</p>
                    <p><a href="/">返回首页</a></p>
                </body>
            </html>
            """,
            status_code=500
        )

@auth_router.get("/callback")
async def github_callback(request: Request, code: str = None, state: str = None, error: str = None):
    """GitHub OAuth 回调处理"""
    if error:
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>授权失败</h1>
                    <p>错误: {error}</p>
                    <p><a href="/">返回首页</a></p>
                </body>
            </html>
            """,
            status_code=400
        )
    
    if not code or not state:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>授权失败</h1>
                    <p>缺少必要的授权参数</p>
                    <p><a href="/">返回首页</a></p>
                </body>
            </html>
            """,
            status_code=400
        )
    
    # 验证state
    stored_state = request.cookies.get("oauth_state")
    if not stored_state or stored_state != state or state not in session_store:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>授权失败</h1>
                    <p>状态验证失败，可能的CSRF攻击</p>
                    <p><a href="/">返回首页</a></p>
                </body>
            </html>
            """,
            status_code=400
        )
    
    try:
        # 交换访问令牌
        access_token = await github_auth.exchange_code_for_token(code)
        if not access_token:
            raise Exception("无法获取访问令牌")
        
        # 获取用户信息
        user_info = await github_auth.get_user_info(access_token)
        if not user_info:
            raise Exception("无法获取用户信息")
        
        # 创建JWT令牌
        jwt_token = github_auth.create_jwt_token(user_info)
        
        # 清理session存储
        session_store.pop(state, None)
        
        # 重定向到主应用并设置cookie
        response = HTMLResponse(
            content=f"""
            <html>
                <head>
                    <meta http-equiv="refresh" content="3;url=http://localhost:7860">
                    <style>
                        body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
                        .success {{ color: green; }}
                        .info {{ color: #666; margin: 20px 0; }}
                    </style>
                </head>
                <body>
                    <h1 class="success">🎉 GitHub 登录成功！</h1>
                    <p>欢迎 <strong>{user_info.get('name', user_info['login'])}</strong>！</p>
                    <p class="info">正在跳转到RAG应用...</p>
                    <p class="info">跳转后请点击"刷新登录状态"按钮来更新界面</p>
                    <p>如果没有自动跳转，<a href="http://localhost:7860">点击这里</a></p>
                    
                    <script>
                        // 3秒后自动跳转到主应用
                        setTimeout(function() {{
                            window.location.href = 'http://localhost:7860';
                        }}, 3000);
                    </script>
                </body>
            </html>
            """
        )
        
        # 设置JWT cookie（7天过期）
        # 为了支持跨端口，设置domain为localhost
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            max_age=JWT_EXPIRE_HOURS * 3600,
            httponly=True,
            secure=False,  # 本地开发设为False，生产环境设为True
            samesite="lax",
            domain="localhost"  # 允许跨端口共享cookie
        )
        
        # 清理oauth_state cookie
        response.delete_cookie("oauth_state")
        
        return response
        
    except Exception as e:
        logger.error(f"OAuth 回调处理失败: {e}")
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>登录失败</h1>
                    <p>错误: {str(e)}</p>
                    <p><a href="/">返回首页</a></p>
                </body>
            </html>
            """,
            status_code=500
        )

@auth_router.post("/logout")
@auth_router.get("/logout")
async def logout():
    """用户登出"""
    response = HTMLResponse(
        content="""
        <html>
            <head>
                <meta http-equiv="refresh" content="2;url=/">
            </head>
            <body>
                <h1>已成功登出</h1>
                <p>正在跳转到首页...</p>
                <p>如果没有自动跳转，<a href="/">点击这里</a></p>
            </body>
        </html>
        """
    )
    response.delete_cookie("access_token")
    return response

@auth_router.get("/user")
async def get_user(request: Request):
    """获取当前用户信息API"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="未授权")
    
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "name": user["name"],
        "email": user["email"],
        "avatar_url": user["avatar_url"]
    }

# 依赖注入：获取当前用户
def require_auth(request: Request) -> Dict[str, Any]:
    """要求用户认证的依赖"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="需要登录")
    return user

def optional_auth(request: Request) -> Optional[Dict[str, Any]]:
    """可选用户认证的依赖"""
    return get_current_user(request)