#!/usr/bin/env python
"""
启动带有GitHub OAuth功能的千问RAG系统
同时启动主应用和OAuth服务
"""

import subprocess
import sys
import time
import signal
import os
from multiprocessing import Process

def start_oauth_server():
    """启动OAuth服务器"""
    print("🔐 启动GitHub OAuth服务...")
    from gradio_oauth_app import create_oauth_app
    import uvicorn
    
    app = create_oauth_app()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )

def start_gradio_app():
    """启动Gradio主应用"""
    print("🚀 启动Gradio RAG应用...")
    time.sleep(2)  # 等待OAuth服务启动
    from gradio_app import main
    main()

def main():
    """主函数"""
    print("🎯 启动千问RAG系统（包含GitHub OAuth）")
    print("=" * 50)
    
    # 检查GitHub OAuth配置
    from github_auth import github_auth
    if not github_auth.is_configured():
        print("⚠️  GitHub OAuth 未配置，将以匿名模式运行")
        print("💡 如需启用GitHub登录，请配置 .env 文件")
        print("📋 参考 GITHUB_AUTH_SETUP.md 文档")
        print()
        
        # 直接启动主应用
        from gradio_app import main as gradio_main
        gradio_main()
        return
    
    print("✅ GitHub OAuth 已配置")
    print("🔗 主应用: http://localhost:7860")
    print("🔐 OAuth服务: http://localhost:8001")
    print()
    
    # 启动两个进程
    oauth_process = None
    gradio_process = None
    
    try:
        # 启动OAuth服务
        oauth_process = Process(target=start_oauth_server)
        oauth_process.start()
        
        # 启动Gradio应用
        gradio_process = Process(target=start_gradio_app)
        gradio_process.start()
        
        print("✅ 两个服务已启动")
        print("📝 使用 Ctrl+C 停止所有服务")
        
        # 等待进程
        oauth_process.join()
        gradio_process.join()
        
    except KeyboardInterrupt:
        print("\n🛑 收到停止信号，正在关闭服务...")
        
        if oauth_process and oauth_process.is_alive():
            oauth_process.terminate()
            oauth_process.join(timeout=5)
            
        if gradio_process and gradio_process.is_alive():
            gradio_process.terminate()
            gradio_process.join(timeout=5)
            
        print("✅ 所有服务已停止")
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()