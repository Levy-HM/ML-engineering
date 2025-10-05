# tts_module.py
import pyttsx3
import os
from fastapi import HTTPException

# 初始化 TTS 引擎（全局）
tts_engine = pyttsx3.init()

def synthesize_speech(text, filename="response.wav"):
    """
    将文本合成语音并保存到文件
    """
    try:
        # 配置参数
        tts_engine.setProperty('rate', 180)   # 语速
        tts_engine.setProperty('volume', 1.0) # 音量 0~1
        
        # 保存到文件
        tts_engine.save_to_file(text, filename)
        tts_engine.runAndWait()
        
        # 读取文件内容并返回字节数据
        with open(filename, 'rb') as f:
            audio_data = f.read()
        
        # 可选：删除临时文件（或者保留供后续使用）
        # os.remove(filename)
        
        return audio_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")

async def text_to_speech(text: str) -> bytes:
    """
    异步接口函数，用于FastAPI路由
    """
    # 使用唯一文件名避免冲突
    import uuid
    filename = f"temp_response_{uuid.uuid4().hex[:8]}.wav"
    
    try:
        audio_data = synthesize_speech(text, filename)
        return audio_data
    finally:
        # 清理临时文件
        if os.path.exists(filename):
            os.remove(filename)
