# import requests

# host="127.0.0.1"
# port="8080"
# url = f"http://{host}:{port}/api/chat"
# model = "deepseek-r1:8b"
# headers = {"Content-Type": "application/json"}
# data = {
#         "model": model, #模型选择
#         "options": {
#             "temperature": 0.  #为0表示不让模型自由发挥，输出结果相对较固定，>0的话，输出的结果会比较放飞自我
#          },
#         "stream": False, #流式输出
#         "messages": [{
#             "role": "system",
#             "content":"你是谁？"
#         }] #对话列表
#     }
# response=requests.post(url,json=data,headers=headers,timeout=60)
# res=response.json()
# print(res)


from langchain_community.llms import Ollama
host="127.0.0.1"
port="11434" #默认的端口号为11434
llm=Ollama(base_url=f"http://{host}:{port}", model="deepseek-r1:8b",temperature=0)
res=llm.invoke("你是谁")
print(res)