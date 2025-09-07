# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
from typing import Dict, Any
import json
import os
import uuid

from asr_module import transcribe_audio
from llm_module import generate_response, clear_conversation_history
from tts_module import text_to_speech
from tools_module import TOOL_REGISTRY

app = FastAPI()

# 创建临时音频目录
os.makedirs("temp_audio", exist_ok=True)
app.mount("/temp_audio", StaticFiles(directory="temp_audio"), name="temp_audio")

class VoiceQueryRequest(BaseModel):
    audio_data: bytes
    session_id: str = "default"
    audio_format: str = "wav"

class TextQueryRequest(BaseModel):
    text: str
    session_id: str = "default"

# 存储会话音频数据的简单缓存
audio_cache = {}

def execute_function_call(function_call: Dict) -> str:
    """执行函数调用并返回结果"""
    func_name = function_call.get('function')
    arguments = function_call.get('arguments', {})
    
    if func_name in TOOL_REGISTRY:
        try:
            result = TOOL_REGISTRY[func_name](**arguments)
            return result
        except Exception as e:
            return f"Error executing {func_name}: {str(e)}"
    else:
        return f"Unknown function: {func_name}"

async def handle_text_query(user_text: str, session_id: str) -> Dict[str, Any]:
    """处理文本查询的核心逻辑"""
    # 生成LLM响应
    llm_response = generate_response(user_text)
    
    final_response = ""
    
    if llm_response['is_function_call']:
        # 执行函数调用
        function_call = llm_response['function_call']
        function_result = execute_function_call(function_call)
        final_response = function_result
    else:
        # 普通文本响应
        final_response = llm_response['text_response']
    
    # 文本转语音
    audio_output = await text_to_speech(final_response)
    
    # 缓存音频数据
    audio_cache[session_id] = audio_output
    
    return {
        "text_response": final_response,
        "audio_data": audio_output,
        "is_function_call": llm_response['is_function_call'],
        "function_call": llm_response.get('function_call', None)
    }

@app.post("/api/text-query/")
async def text_query_endpoint(request: TextQueryRequest):
    try:
        response_data = await handle_text_query(request.text, request.session_id)
        
        # 生成唯一的音频文件名
        audio_filename = f"response_{uuid.uuid4().hex}.wav"
        audio_path = f"temp_audio/{audio_filename}"
        
        # 保存音频到文件
        with open(audio_path, "wb") as f:
            f.write(response_data["audio_data"])
        
        return {
            "text_response": response_data["text_response"],
            "audio_url": f"/temp_audio/{audio_filename}",
            "is_function_call": response_data["is_function_call"],
            "function_call": response_data.get("function_call")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice-query/")
async def voice_query_endpoint(request: VoiceQueryRequest):
    try:
        # 语音识别
        user_text = await transcribe_audio(request.audio_data, request.audio_format)
        
        # 处理文本查询
        response_data = await handle_text_query(user_text, request.session_id)
        
        # 生成唯一的音频文件名
        audio_filename = f"response_{uuid.uuid4().hex}.wav"
        audio_path = f"temp_audio/{audio_filename}"
        
        # 保存音频到文件
        with open(audio_path, "wb") as f:
            f.write(response_data["audio_data"])
        
        return {
            "transcribed_text": user_text,
            "text_response": response_data["text_response"],
            "audio_url": f"/temp_audio/{audio_filename}",
            "is_function_call": response_data["is_function_call"],
            "function_call": response_data.get("function_call")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset-conversation/")
async def reset_conversation(session_id: str = "default"):
    clear_conversation_history()
    if session_id in audio_cache:
        del audio_cache[session_id]
    return {"status": "conversation reset"}

# 清理临时文件的路由
@app.delete("/api/cleanup/{filename}")
async def cleanup_file(filename: str):
    try:
        audio_path = f"temp_audio/{filename}"
        if os.path.exists(audio_path):
            os.remove(audio_path)
            return {"status": "file deleted"}
        return {"status": "file not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))