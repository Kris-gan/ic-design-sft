"""
gen_output.py
输出生成脚本
读取过滤后的指令，调用 Qwen 模型为每条指令生成专业回答，
支持断点续跑，每批完成后立即保存。
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# 加载根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

# ====== 路径配置 ======
INPUT_FILE  = Path(__file__).parent.parent / "filter/instructions_50k_filtered.json"
OUTPUT_FILE = Path(__file__).parent.parent / "output/dataset_50k_with_output.json"

# ====== 并发配置 ======
BATCH_SIZE  = 50   # 每批指令数，gather 一次
CONCURRENCY = 10   # 最大并发请求数

SYSTEM_PROMPT = """你是一位资深IC设计工程师，拥有10年以上数字芯片设计经验。
请用专业、准确、详细的方式回答IC设计相关问题。
- 代码类问题：给出可综合的完整代码，加关键注释
- 概念类问题：先给定义，再举例，最后说实际应用
- 分析类问题：给出分析思路 + 具体结论
回答语言：中文（代码保持英文）
注意：直接给出答案，不要说"你好"、不要自我介绍、不要重复问题。"""

async_client = AsyncOpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


async def generate_output(
    instruction: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> str | None:
    """为单条指令生成回答，失败时指数退避重试最多 3 次。"""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await async_client.chat.completions.create(
                    model="qwen3.5-122b-a10b",
                    max_tokens=1000,
                    extra_body={"enable_thinking": False},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": instruction},
                    ],
                )
                print(f"[{index + 1}/{total}] 完成: {instruction[:30]}...")
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[{index + 1}/{total}] 第{attempt + 1}次失败: {e}")
                await asyncio.sleep(2 ** attempt)  # 指数退避：1s / 2s / 4s
        return None


async def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        instructions = json.load(f)
    total = len(instructions)
    print(f"共 {total} 条指令待处理")

    # 断点续跑：跳过已生成的指令
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        done = {item["instruction"] for item in dataset}
        instructions = [inst for inst in instructions if inst not in done]
        print(f"已有 {len(dataset)} 条，剩余 {len(instructions)} 条")
    except FileNotFoundError:
        dataset = []
        print("从头开始生成")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    # 分批处理，每批完成后立即保存
    for batch_start in range(0, len(instructions), BATCH_SIZE):
        batch = instructions[batch_start: batch_start + BATCH_SIZE]

        tasks = [
            generate_output(inst, semaphore, len(dataset) + batch_start + i, total)
            for i, inst in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks)

        for inst, result in zip(batch, results):
            if result is not None:
                dataset.append({"instruction": inst, "input": "", "output": result})

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"--- 已保存 {len(dataset)} 条 ---")

    print(f"\n全部完成！共 {len(dataset)}/{total} 条，保存至 {OUTPUT_FILE}")


asyncio.run(main())
