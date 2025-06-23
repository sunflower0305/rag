# GitHub OAuth 登录功能设置指南

本文档说明如何在千问RAG系统中配置和使用GitHub OAuth登录功能。

## 功能特性

- ✅ **GitHub OAuth 2.0 认证**：安全的第三方登录
- ✅ **多用户支持**：每个用户拥有独立的会话和文档记录
- ✅ **数据隔离**：用户只能看到自己的聊天历史
- ✅ **无缝集成**：与现有功能完全兼容
- ✅ **可选配置**：不配置时系统以匿名模式运行

## 快速开始

### 1. 创建 GitHub OAuth App

1. 访问 [GitHub Developer Settings](https://github.com/settings/applications/new)
2. 填写应用信息：
   - **Application name**: `千问RAG系统` 或自定义名称
   - **Homepage URL**: `http://localhost:7860`（本地开发）
   - **Authorization callback URL**: `http://localhost:8001/auth/callback`
3. 点击 "Register application" 创建应用
4. 记录生成的 `Client ID` 和 `Client Secret`

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入配置：

```env
# 必需：阿里云DashScope API密钥
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# 可选：GitHub OAuth配置
GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here

# 可选：JWT密钥（建议生产环境设置）
JWT_SECRET_KEY=your_super_secret_jwt_key_at_least_32_characters_long
```

**重要**：GitHub OAuth App的回调URL需要设置为：`http://localhost:8001/auth/callback`

### 3. 安装依赖

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装新增的OAuth依赖
uv pip install -r requirements.txt
```

### 4. 启动应用

有两种启动方式：

#### 方式一：一键启动（推荐）
```bash
python start_with_oauth.py
```
这会同时启动主应用和OAuth服务。

#### 方式二：分别启动
```bash
# 终端1：启动OAuth服务
python gradio_oauth_app.py

# 终端2：启动主应用
python gradio_app.py
```

## 使用说明

### 登录流程

1. 打开应用后，在右上角可以看到 "🔗 GitHub 登录" 按钮
2. 点击按钮将跳转到GitHub授权页面
3. 授权后自动返回应用，显示用户信息
4. 登录后创建的会话将与您的GitHub账户关联

### 数据隔离

- **登录用户**：只能看到自己创建的会话和聊天记录
- **匿名用户**：可以正常使用所有功能，但会话不会持久化关联
- **会话隔离**：不同用户的数据完全隔离，保护隐私

### 登出

点击 "🚪 登出" 按钮即可登出，系统将：
- 清除当前会话状态
- 隐藏用户专属的历史记录
- 返回匿名模式

## 生产环境部署

### 1. 更新OAuth App配置

在GitHub OAuth App设置中：
- **Homepage URL**: `https://your-domain.com`
- **Authorization callback URL**: `https://your-domain.com:8001/auth/callback`

或者将OAuth服务配置到同一端口的不同路径。

### 2. 环境变量配置

```env
# 使用生产域名
GITHUB_CLIENT_ID=your_production_client_id
GITHUB_CLIENT_SECRET=your_production_client_secret

# 强JWT密钥
JWT_SECRET_KEY=your_super_strong_production_jwt_secret

# 其他生产配置
PORT=8080
APP_DOMAIN=your-domain.com
```

### 3. HTTPS配置

确保生产环境使用HTTPS，OAuth回调需要安全连接。

## 故障排除

### 常见问题

1. **"GitHub OAuth 未配置" 提示**
   - 检查 `.env` 文件中的 `GITHUB_CLIENT_ID` 和 `GITHUB_CLIENT_SECRET`
   - 确保环境变量正确加载

2. **登录后跳转失败**
   - 检查GitHub OAuth App的回调URL设置
   - 确保回调URL与实际运行地址匹配

3. **授权失败**
   - 检查Client ID和Secret是否正确
   - 查看应用日志获取详细错误信息

4. **会话丢失**
   - 检查JWT_SECRET_KEY是否改变
   - 确保cookie设置正确

### 调试方法

1. 查看应用日志：
   ```bash
   python gradio_app.py
   ```

2. 检查环境变量：
   ```python
   import os
   print("GitHub Client ID:", os.getenv("GITHUB_CLIENT_ID"))
   print("JWT Secret:", bool(os.getenv("JWT_SECRET_KEY")))
   ```

3. 测试OAuth配置：
   - 直接访问 `/auth/github` 路径
   - 检查是否正确跳转到GitHub

## 安全注意事项

1. **保护敏感信息**
   - 不要将 `.env` 文件提交到代码仓库
   - 定期更换API密钥和JWT密钥

2. **生产环境配置**
   - 使用HTTPS协议
   - 设置安全的JWT密钥（至少32位）
   - 监控OAuth应用使用情况

3. **用户隐私**
   - 系统仅获取GitHub用户基本信息
   - 用户数据完全隔离存储
   - 支持随时登出和数据清理

## API文档

### OAuth端点

- `GET /auth/github` - GitHub登录入口
- `GET /auth/callback` - OAuth回调处理
- `GET /auth/logout` - 用户登出
- `GET /auth/user` - 获取当前用户信息

### 数据库结构

新增用户表：
```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    name TEXT,
    email TEXT,
    avatar_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

会话表添加用户关联：
```sql
ALTER TABLE chat_sessions ADD COLUMN user_id INTEGER;
```

## 支持与反馈

如果在配置或使用过程中遇到问题，请：

1. 检查本文档的故障排除部分
2. 查看应用日志获取详细错误信息
3. 确认GitHub OAuth App配置正确
4. 验证环境变量设置

该功能完全可选，不影响系统的核心RAG功能。未配置GitHub OAuth时，系统将以匿名模式正常运行。