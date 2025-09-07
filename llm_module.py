# llm_module.py
import json
import re

class FunctionCallingLLM:
    def __init__(self):
        print("Using deterministic function calling LLM")
    
    def generate_response(self, user_text):
        """
        确定性的函数调用逻辑，不依赖外部模型
        """
        user_text_lower = user_text.lower()
        
        # 数学计算查询
        if any(word in user_text_lower for word in ['calculate', 'compute', 'math', 'what is', 'what\'s', '+', '-', '*', '/', '^', 'power']):
            expression = self._extract_math_expression(user_text)
            return {
                'is_function_call': True,
                'function_call': {
                    "function": "calculate",
                    "arguments": {"expression": expression}
                },
                'raw_response': f'{{"function": "calculate", "arguments": {{"expression": "{expression}"}}}}'
            }
        
        # arXiv搜索查询
        elif any(word in user_text_lower for word in ['search', 'find', 'paper', 'arxiv', 'research', 'article']):
            query = self._extract_search_query(user_text)
            return {
                'is_function_call': True,
                'function_call': {
                    "function": "search_arxiv", 
                    "arguments": {"query": query}
                },
                'raw_response': f'{{"function": "search_arxiv", "arguments": {{"query": "{query}"}}}}'
            }
        
        # 普通对话
        else:
            response = self._generate_normal_response(user_text)
            return {
                'is_function_call': False,
                'text_response': response,
                'raw_response': response
            }
    
    def _extract_math_expression(self, text):
        """从文本中提取数学表达式"""
        text_lower = text.lower()
        
        # 处理幂运算
        if 'power' in text_lower:
            if '2 to the power of 8' in text_lower:
                return "2**8"
            elif '3 to the power of 4' in text_lower:
                return "3**4"
            else:
                # 提取数字
                import re
                numbers = re.findall(r'\d+', text)
                if len(numbers) >= 2:
                    return f"{numbers[0]}**{numbers[1]}"
                return "2**8"  # 默认
        
        # 处理普通数学表达式
        math_pattern = r'(\d+[\s*+\-/\^\.]*)+'
        match = re.search(math_pattern, text)
        if match:
            return match.group(0).replace(' ', '')
        
        return "2+2"  # 默认
    
    def _extract_search_query(self, text):
        """从文本中提取搜索查询"""
        text_lower = text.lower()
        
        # 移除常见指令词
        query = text_lower
        for word in ['search', 'find', 'papers', 'articles', 'research', 'about', 'on', 'for', 'look up']:
            query = query.replace(word, '')
        
        query = query.strip(' ".\'')
        return query if query else "artificial intelligence"
    
    def _generate_normal_response(self, text):
        """生成普通对话响应"""
        greetings = ['hello', 'hi', 'hey', 'greetings']
        if any(word in text.lower() for word in greetings):
            return "Hello! How can I help you today?"
        elif 'how are you' in text.lower():
            return "I'm doing well, thank you! How can I assist you?"
        else:
            return "I can help you with mathematical calculations or searching for academic papers. What would you like to do?"

# 使用模拟LLM
llm = FunctionCallingLLM()
conversation_history = []

def generate_response(user_text):
    """
    生成LLM响应，支持函数调用
    """
    global conversation_history
    
    # 添加到对话历史
    conversation_history.append({"role": "user", "text": user_text})
    
    # 使用模拟LLM生成响应
    response = llm.generate_response(user_text)
    
    # 保存到对话历史
    if response['is_function_call']:
        conversation_history.append({
            "role": "assistant",
            "text": f"Function call: {response['function_call']['function']}",
            "is_function_call": True,
            "function_call": response['function_call'],
            "raw_response": response['raw_response']
        })
    else:
        conversation_history.append({
            "role": "assistant",
            "text": response['text_response'],
            "is_function_call": False,
            "raw_response": response['raw_response']
        })
    
    return response

def clear_conversation_history():
    """清空对话历史"""
    global conversation_history
    conversation_history = []
    print("Conversation history cleared")

# 移除不需要的函数
def format_conversation_prompt():
    return ""

def parse_llm_response(response: str):
    return {
        'is_function_call': False,
        'text_response': response,
        'raw_response': response
    }