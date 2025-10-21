import dashscope
from dashscope import Generation

# 设置API Key
dashscope.api_key = "sk-9e05a08baf0142088b0de2f6dabbb730"

def call_with_message():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你能帮我做些什么？"}
    ]
    
    response = Generation.call(
        model="qwen-plus",
        messages=messages,
        result_format="message"
    )
    
    return response

if __name__ == "__main__":
    try:
        result = call_with_message()
        print(result.output.choices[0].message.content)
    except Exception as e:
        print(f"错误信息: {e}")
        print("请参考文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")