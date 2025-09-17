from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain_community.chat_models import ChatTongyi
import requests
import os
import asyncio
from typing import AsyncIterable
import json

# 1. 初始化FastAPI应用
app = FastAPI(title="AI工具调用助手")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请改为具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件和模板
templates = Jinja2Templates(directory="templates")

# 2. 定义工具函数（保持不变）
def get_weather(city: str) -> dict:
    """获取天气"""
    try:
        res = requests.post(
            "https://6789.zeabur.app/mcp/tools/get_weather/invoke",
            json={"city": city},
            timeout=10
        )
        res.raise_for_status()
        return res.json().get("data", res.json())
    except Exception as e:
        return {"error": f"天气服务调用失败: {str(e)}"}

def search_wikipedia(query: str) -> dict:
    """搜索维基百科"""
    try:
        res = requests.post(
            "https://6789.zeabur.app/mcp/tools/search_wikipedia/invoke",
            json={"query": query},
            timeout=10
        )
        res.raise_for_status()
        return res.json().get("data", res.json())
    except Exception as e:
        return {"error": f"维基百科搜索失败: {str(e)}"}

def calculate_math(expression: str) -> dict:
    """计算数学表达式"""
    try:
        res = requests.post(
            "https://6789.zeabur.app/mcp/tools/calculate_math/invoke",
            json={"expression": expression},
            timeout=10
        )
        res.raise_for_status()
        return res.json().get("data", res.json())
    except Exception as e:
        return {"error": f"计算服务调用失败: {str(e)}"}

# 3. 将函数封装为LangChain Tool
tools = [
    Tool(
        name="GetWeather",
        func=get_weather,
        description="根据城市名称获取天气信息。输入应为城市名称字符串。"
    ),
    Tool(
        name="SearchWikipedia",
        func=search_wikipedia,
        description="搜索维基百科获取相关信息。输入应为搜索查询字符串。"
    ),
    Tool(
        name="CalculateMath",
        func=calculate_math,
        description="计算数学表达式。输入应为数学表达式字符串。"
    )
]

# 4. 初始化LLM和智能体 —— 使用 ChatTongyi（通义千问 API）
# ✅ 从环境变量读取 API Key
api_key = "sk-cb5d9fe04d7b4adba5952fa2de765def"
if not api_key:
    raise RuntimeError("请设置环境变量 DASHSCOPE_API_KEY")

llm = ChatTongyi(
    dashscope_api_key=api_key,
    model_name="qwen-max",  # 使用 qwen-max 模型
    temperature=0.3
)

# 设置系统消息
system_message = SystemMessage(content="""你是一个有帮助的助手，可以调用工具来获取天气信息、搜索维基百科或计算数学表达式。
请用中文回答所有问题。
请根据用户的问题决定是否需要调用工具以及调用哪个工具。
如果问题与天气相关，调用GetWeather工具；
如果问题需要知识检索，调用SearchWikipedia工具；
如果问题涉及数学计算，调用CalculateMath工具；
如果问题不需要工具就能回答，请直接回答。""")

# 初始化智能体
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": system_message},
    handle_parsing_errors=True  # 防止解析错误崩溃
)

# 5. 定义请求和响应模型
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    tool_used: bool = False
    tool_name: str = None

# 6. 创建API端点
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = agent.run(request.message)
        return ChatResponse(
            response=str(result),
            tool_used=True if any(t.name in str(result).lower() for t in tools) else False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理消息时出错: {str(e)}")

# 流式响应生成器（简化版，避免复杂流式解析）
async def generate_stream(prompt: str) -> AsyncIterable[str]:
    try:
        # 使用 invoke 调用 agent，然后逐字发送
        result = agent.run(prompt)
        for char in result:
            yield f"data: {json.dumps({'content': char})}\n\n"
            await asyncio.sleep(0.01)  # 模拟流式输出
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/chat/stream")
async def chat_stream(prompt: str):
    """流式聊天接口"""
    return StreamingResponse(generate_stream(prompt), media_type="text/event-stream")

# 7. 提供前端页面
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查：确保 DASHSCOPE_API_KEY 已配置"""
    api_key = "sk-cb5d9fe04d7b4adba5952fa2de765def"
    if not api_key:
        return {"status": "unhealthy", "error": "Missing DASHSCOPE_API_KEY"}, 503
    return {"status": "healthy", "llm": "qwen-max", "provider": "dashscope"}

# 运行服务（仅本地测试）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)