"""
gen_instruction.py
Self-Instruct 指令生成脚本
从种子数据出发，调用 Qwen 模型批量生成 IC 设计领域指令，目标 50k 条。
"""

import asyncio
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# 加载根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

# ====== 配置 ======
SEED_FILE   = Path(__file__).parent.parent / "data/seeds/seeds_20.json"
OUTPUT_FILE = Path(__file__).parent.parent / "instruction/instructions_50k_raw.json"
TARGET      = 50000
CONCURRENCY = 10   # 同时并发请求数
BATCH_SIZE  = 20   # 每次生成指令数
SAVE_EVERY  = 5000 # 每累计多少条保存一次 checkpoint

async_client = AsyncOpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


async def generate_batch(all_instructions: list[str]) -> list[str]:
    """随机抽取种子，调用模型生成一批新指令。"""
    sample = random.sample(all_instructions, min(8, len(all_instructions)))
    prompt = f"""你是IC设计专家。参考以下示例指令的风格和类型，生成{BATCH_SIZE}条全新的、不重复的IC设计相关指令。

示例指令：
{chr(10).join(f'- {s}' for s in sample)}

要求：
1. 涵盖：规范解析、Verilog代码生成、时序分析、DFT、低功耗设计
2. 难度混合：基础概念 + 实战分析 + 代码生成
3. 只输出指令列表，每行一条，不要编号，不要解释

输出："""

    try:
        response = await async_client.chat.completions.create(
            model="qwen3.5-flash",
            max_tokens=1500,
            extra_body={"enable_thinking": False},
            messages=[{"role": "user", "content": prompt}],
        )
        lines = response.choices[0].message.content.strip().split("\n")
        return [l.strip("- ").strip() for l in lines if l.strip() and len(l.strip()) > 5]
    except Exception as e:
        print(f"请求失败: {e}")
        return []


async def main():
    # 读取种子数据
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        seed_data = json.load(f)

    # 种子只含 instruction 字段时提取文本，否则直接当字符串列表用
    if isinstance(seed_data[0], dict):
        all_instructions = [item["instruction"] for item in seed_data]
    else:
        all_instructions = seed_data

    print(f"初始种子：{len(all_instructions)} 条，目标：{TARGET} 条")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    round_num = 0
    while len(all_instructions) < TARGET:
        tasks = [generate_batch(all_instructions) for _ in range(CONCURRENCY)]
        results = await asyncio.gather(*tasks)

        new_count = sum(len(b) for b in results)
        for batch in results:
            all_instructions.extend(batch)

        round_num += 1
        print(f"第{round_num}轮 +{new_count} 条，当前共 {len(all_instructions)} 条")

        # 定期保存 checkpoint
        if len(all_instructions) % SAVE_EVERY < CONCURRENCY * BATCH_SIZE:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_instructions, f, ensure_ascii=False, indent=2)
            print("  checkpoint 已保存")

        await asyncio.sleep(0.3)

    # 截断至目标数量并最终保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_instructions[:TARGET], f, ensure_ascii=False, indent=2)
    print(f"完成！共 {TARGET} 条，保存至 {OUTPUT_FILE}")


asyncio.run(main())
