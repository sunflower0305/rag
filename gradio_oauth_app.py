"""
独立的OAuth处理应用
由于Gradio路由挂载的复杂性，我们创建一个独立的FastAPI应用来处理OAuth
"""

import uvicorn
from fastapi import FastAPI
from github_auth import auth_router
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_oauth_app():
    """创建OAuth处理应用"""
    app = FastAPI(
        title="GitHub OAuth Handler",
        description="GitHub OAuth 认证处理服务"
    )
    
    # 挂载OAuth路由
    app.include_router(auth_router)
    
    # 健康检查端点
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "service": "oauth-handler"}
    
    # 根路径重定向
    @app.get("/")
    async def root():
        return {"message": "GitHub OAuth Handler", "auth_endpoint": "/auth/github"}
    
    return app

if __name__ == "__main__":
    print("🔐 启动GitHub OAuth处理服务...")
    print("📍 OAuth端点: http://localhost:8001/auth/github")
    print("🔄 健康检查: http://localhost:8001/health")
    
    app = create_oauth_app()
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )