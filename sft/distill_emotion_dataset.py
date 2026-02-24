import os
import json
import asyncio
from typing import List, Dict, Any
import aiohttp

# ==========================================
# 配置区域
# ==========================================
# 默认使用 OpenAI 兼容格式的 API (例如 DeepSeek 官方 API 或各类中转)
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-sp-070196a281124489bf71f482d10e6623")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
MODEL_NAME = os.environ.get("TEACHER_MODEL_NAME", "qwen3-coder-plus") # 使用 deepseek-chat 或 deepseek-reasoner

INPUT_FILE = "raw_dialogues.jsonl" # 待标注的原始语料
OUTPUT_FILE = "distilled_emotions.jsonl" # 生成的带有 8 维向量的伪标签数据集

MAX_CONCURRENT_REQUESTS = 5 # 控制并发量，避免被限流

# 8个固定维度
EMOTION_DIMS = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "平静"]

# ==========================================
# Prompt 设计
# ==========================================
SYSTEM_PROMPT = """你是一个极其敏锐的人类情感心理分析专家。
你需要阅读一段小说的剧本上下文，并重点分析【目标台词】说话人的即时情感状态。

【分析规则】
1. 你的分析必须基于前文语境（包含动作、神态、对话），而不能仅仅只看目标台词。
2. 请综合分析出该角色在说这句台词时，内心的情感混合分布。
3. 请严格输出一个包含 8 个情感维度的概率分布字典。这 8 个维度是：["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "平静"]。
4. 这 8 个数字必须是 0.0 到 1.0 之间的小数，且它们的总和必须**严格等于 1.0**。通常人类的情感是混合的，比如 0.6的悲伤 + 0.3的低落 + 0.1的平静。
5. 必须严格以 JSON 格式输出，不要包含任何其他分析过程或 Markdown 标记（如 ```json），只输出合法的 JSON 对象。

【输出格式示例】
{
    "高兴": 0.0,
    "愤怒": 0.1,
    "悲伤": 0.5,
    "恐惧": 0.1,
    "反感": 0.0,
    "低落": 0.3,
    "惊讶": 0.0,
    "平静": 0.0
}
"""

def build_user_prompt(context: str, current_line: str, prior_emotion: str = None) -> str:
    prompt = "【剧本片段】\n"
    prompt += f"上文语境（最多前200字）：\n{context}\n\n"
    if prior_emotion:
        prompt += f"前端基础系统初步识别的主情绪参考：[{prior_emotion}]\n\n"
    prompt += f"【目标台词（请重点分析这句）】\n{current_line}\n\n"
    prompt += "请给出这句目标台词对应的 8 维情感概率分布 JSON："
    return prompt

# ==========================================
# 异步请求逻辑
# ==========================================
async def fetch_emotion_vector(session: aiohttp.ClientSession, item: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    async with semaphore:
        context = item.get("context", "")
        current_line = item.get("current_line", "")
        prior_emotion = item.get("prior_emotion", "")
        
        user_prompt = build_user_prompt(context, current_line, prior_emotion)
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3, # 降低随机性，使输出更稳定
            # "response_format": {"type": "json_object"} # 如果模型支持该参数可开启
        }
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            async with session.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60) as response:
                if response.status != 200:
                    text = await response.text()
                    print(f"API 请求失败: {response.status} - {text}")
                    return None
                    
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                
                # 简单清理可能包含的 markdown 符号
                content = content.replace("```json", "").replace("```", "").strip()
                
                emotion_dict = json.loads(content)
                
                # 验证维度和归一化
                vector = []
                for dim in EMOTION_DIMS:
                    val = float(emotion_dict.get(dim, 0.0))
                    vector.append(val)
                
                # 简单归一化保障
                total = sum(vector)
                if total > 0:
                    vector = [round(v / total, 3) for v in vector]
                else:
                    vector = [0.0] * 7 + [1.0] # 默认全归为平静
                    
                # 返回组合好的数据
                result = item.copy()
                result["emotion_vector"] = vector
                result["raw_teacher_output"] = emotion_dict
                return result
                
        except Exception as e:
            print(f"处理数据时发生异常: {current_line[:20]}... Error: {str(e)}")
            return None

async def main():
    if not os.path.exists(INPUT_FILE):
        # 如果文件不存在，自动生成一些测试数据演示格式
        print(f"未找到输入文件 {INPUT_FILE}，正在生成演示测试数据...")
        generate_demo_data()
        
    print(f"读取原始数据从 {INPUT_FILE} ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
        
    print(f"共发现 {len(items)} 条数据待处理，启动异步并发...")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_emotion_vector(session, item, semaphore) for item in items]
        results = await asyncio.gather(*tasks)
        
    # 过滤掉失败的，并保存
    valid_results = [r for r in results if r is not None]
    
    print(f"处理完成！成功 {len(valid_results)}/{len(items)}。")
    print(f"写入到 {OUTPUT_FILE} ...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in valid_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print("数据集蒸馏脚本执行完毕。")

def generate_demo_data():
    demo_data = [
        {
            "id": 1,
            "context": "王老板把茶杯重重地摔在桌子上，茶水溅得到处都是：“你们就是这样办事的吗？！”\n李秘书吓得大气都不敢出，缩在角落里。",
            "current_line": "李秘书颤抖着说：“老板，我们真的尽力了……是对方突然毁约。”",
            "prior_emotion": "害怕"
        },
        {
            "id": 2,
            "context": "清晨的阳光洒在校园的花坛边，微风拂过。两人并肩走着，谁也没有刻意去打破这份宁静。",
            "current_line": "女孩突然停下脚步，转头看向他，嘴角勾起一抹笑意：“其实，那天我就猜到是你了。”",
            "prior_emotion": "高兴"
        }
    ]
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        for item in demo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
