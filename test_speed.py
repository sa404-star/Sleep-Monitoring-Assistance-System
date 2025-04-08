import requests
import time

# ollama 的 API 地址
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

# 请求参数
payload = {
    "model": "deepseek-r1:8b",  # 替换为你的模型名称
    "prompt": "目标检测的具体含义是什么？",  # 替换为你的输入文本
    "stream": False,  # 设置为 False，一次性返回完整结果
    "max_tokens": 100  # 设置生成的最大 token 数量
}

# 打印 model 和 prompt 信息
print(f"使用的模型: {payload['model']}")
print(f"输入的问题: {payload['prompt']}")

# 记录开始时间
start_time = time.time()
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

# 发送请求
response = requests.post(OLLAMA_API_URL, json=payload)

# 记录结束时间
end_time = time.time()
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

# 解析响应
if response.status_code == 200:
    result = response.json()
    # print(result)
    generated_text = result.get("response", "")
    generated_tokens = result.get("eval_count", 0)  # 获取生成的 token 数量
    elapsed_time = end_time - start_time

    # 计算每秒生成的 token 数量
    tokens_per_second = generated_tokens / elapsed_time

    print(f"模型回答: {generated_text}")
    print(f"生成时间: {elapsed_time:.2f}秒")
    print(f"生成 token 数量: {generated_tokens}")
    print(f"每秒生成 token 数量: {tokens_per_second:.2f}")
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(f"错误信息: {response.text}")
