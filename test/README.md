# 测试和调试文件

这个目录包含所有用于测试和调试RAG系统的Python文件。

## 测试文件

### 功能测试
- `test_custom_llm.py` - 测试自定义LLM集成
- `test_retrieval_only.py` - 测试检索功能
- `test_chroma_integration.py` - 测试ChromaDB集成
- `test_gradio.py` - 测试Gradio界面

### 调试文件
- `debug_dashscope.py` - 调试DashScope API连接
- `debug_dashscope2.py` - DashScope API调试（版本2）
- `debug_native_dashscope.py` - 原生DashScope API调试

### 通用测试
- `test.py` - 通用测试脚本

## 运行测试

从项目根目录运行：

```bash
# 测试自定义LLM
python test/test_custom_llm.py

# 测试检索功能
python test/test_retrieval_only.py

# 测试ChromaDB
python test/test_chroma_integration.py

# 调试DashScope API
python test/debug_dashscope.py
```

## 注意事项

- 运行测试前确保已设置正确的API密钥
- 某些测试可能需要PDF文件存在于`pdf/`目录中
- 部分测试会创建临时文件，建议在测试环境中运行