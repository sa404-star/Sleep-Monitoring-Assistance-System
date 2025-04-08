import time
from langchain_community.llms import Ollama
import statistics

def test_model_speed(model_name, prompt, num_tests=5):
    """测试Ollama模型的响应速度"""
    host = "127.0.0.1"
    port = "11434"
    llm = Ollama(
        base_url=f"http://{host}:{port}", 
        model=model_name
    )
    
    times = []
    token_counts = []
    
    print(f"开始测试模型 {model_name}...")
    
    for i in range(num_tests):
        start_time = time.time()
        response = llm.invoke(prompt)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        token_count = len(response.split())
        token_counts.append(token_count)
        
        print(f"测试 {i+1}: 耗时 {elapsed_time:.2f} 秒, 约 {token_count} 个token")
    
    avg_time = statistics.mean(times)
    avg_tokens = statistics.mean(token_counts)
    tokens_per_second = avg_tokens / avg_time
    
    print(f"\n结果摘要:")
    print(f"平均响应时间: {avg_time:.2f} 秒")
    print(f"平均token数: {avg_tokens:.1f}")
    print(f"生成速度: {tokens_per_second:.2f} tokens/秒")
    
    return {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tokens_per_second": tokens_per_second
    }

if __name__ == "__main__":
    # 可以测试多个模型进行比较
    models_to_test = ["deepseek-r1:8b"]  # 添加您想测试的其他模型
    test_prompt = "请详细解释人类的睡眠周期包含哪些阶段，每个阶段的特点是什么？"
    
    results = {}
    for model in models_to_test:
        results[model] = test_model_speed(model, test_prompt)
        print("\n" + "-"*50 + "\n")
    
    # 比较不同模型的结果
    if len(models_to_test) > 1:
        print("模型速度比较:")
        for model, result in results.items():
            print(f"{model}: {result['tokens_per_second']:.2f} tokens/秒")