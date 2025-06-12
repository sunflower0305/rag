# PDF 项目

这是一个使用 OpenAI API 生成文本嵌入向量的项目。

## 环境设置

本项目使用 uv 进行依赖管理。

### 安装依赖

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
```

### 环境变量

复制 `.env.example` 文件为 `.env` 并填入你的 API 密钥：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

## 运行

```bash
python embedding.py
```