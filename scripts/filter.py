"""
filter.py
数据质量过滤脚本
对原始生成的指令做两轮过滤：
  1. 规则过滤：长度、中文字符数、非法字符
  2. ROUGE-L 去重：相似度超过阈值的指令视为重复丢弃
"""

import json
import random
import re
from pathlib import Path

from rouge_chinese import Rouge

# ====== 路径配置 ======
INPUT_FILE  = Path(__file__).parent.parent / "instruction/instructions_50k_raw.json"
OUTPUT_FILE = Path(__file__).parent.parent / "filter/instructions_50k_filtered.json"

# ====== 过滤参数 ======
MIN_LEN         = 10     # 指令最短字符数
MAX_LEN         = 200    # 指令最长字符数
MIN_ZH_CHARS    = 3      # 最少中文字符数
ROUGE_THRESHOLD = 0.7    # 相似度超过此值视为重复
RECENT_WINDOW   = 100    # 去重时比对最近保留的 N 条
RANDOM_SAMPLE   = 100    # 去重时随机抽取的 N 条
PROGRESS_EVERY  = 500    # 进度打印间隔

rouge = Rouge()


def rule_filter(inst: str) -> bool:
    """规则过滤：长度、中文字符、非法字符。"""
    if not (MIN_LEN <= len(inst) <= MAX_LEN):
        return False
    if len(re.findall(r'[\u4e00-\u9fff]', inst)) < MIN_ZH_CHARS:
        return False
    if re.search(r'[<>|\\]{2,}', inst):
        return False
    return True


def rouge_l(a: str, b: str) -> float:
    """计算两条指令的 ROUGE-L F1（字符级分词）。"""
    try:
        a_ = " ".join(list(a))
        b_ = " ".join(list(b))
        return rouge.get_scores(a_, b_)[0]["rouge-l"]["f"]
    except Exception:
        return 0.0


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        instructions = json.load(f)
    print(f"过滤前：{len(instructions)} 条")

    # 第一轮：规则过滤
    after_rule = [inst for inst in instructions if rule_filter(inst)]
    print(f"规则过滤后：{len(after_rule)} 条")

    # 第二轮：ROUGE-L 去重
    kept = []
    total = len(after_rule)

    for i, inst in enumerate(after_rule):
        if i % PROGRESS_EVERY == 0:
            print(f"  ROUGE-L 去重进度：{i}/{total}  已保留：{len(kept)} 条")

        # 比对池：最近 N 条 + 随机抽 N 条，兼顾速度与覆盖率
        recent = kept[-RECENT_WINDOW:]
        random_pool = (
            random.sample(kept[:-RECENT_WINDOW], min(RANDOM_SAMPLE, len(kept) - RECENT_WINDOW))
            if len(kept) > RECENT_WINDOW else []
        )
        compare_pool = recent + random_pool

        if not any(rouge_l(inst, exist) > ROUGE_THRESHOLD for exist in compare_pool):
            kept.append(inst)

    print(f"去重后：{len(kept)} 条")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
