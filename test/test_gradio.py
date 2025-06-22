import gradio as gr
import os

def test_api_key(api_key):
    if api_key:
        return f"✅ API密钥已输入: {api_key[:10]}..."
    return "❌ 请输入API密钥"

def simple_chat(message, history):
    if not message:
        return history, ""
    
    # 简单的回应
    response = f"收到您的消息: {message}"
    history.append([message, response])
    return history, ""

# 创建简单界面
with gr.Blocks(title="测试Gradio") as demo:
    gr.Markdown("# 🧪 Gradio测试界面")
    
    with gr.Row():
        with gr.Column():
            api_input = gr.Textbox(label="API密钥", type="password")
            test_btn = gr.Button("测试API")
            api_status = gr.Textbox(label="状态", interactive=False)
        
        with gr.Column():
            chatbot = gr.Chatbot(label="聊天测试")
            msg_input = gr.Textbox(label="消息", placeholder="输入测试消息...")
            send_btn = gr.Button("发送")
    
    # 事件绑定
    test_btn.click(test_api_key, inputs=[api_input], outputs=[api_status])
    send_btn.click(simple_chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])

if __name__ == "__main__":
    print("启动简单测试界面...")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=False,
            debug=True
        )
    except Exception as e:
        print(f"启动失败: {e}")
        # 尝试默认设置
        print("尝试默认设置...")
        demo.launch()