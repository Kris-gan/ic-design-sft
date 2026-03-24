"""
evaluate.py
模型评估脚本（在训练服务器上运行）
对比基座模型与 LoRA 微调后模型在 IC Design 测试集上的 ROUGE-L 分数。

运行环境：~/LlamaFactory/
运行方式：python scripts/evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from rouge_chinese import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer

# ====== 路径配置（按实际服务器路径修改） ======
MERGED_MODEL = "/home/ubuntu/models/qwen2.5-ic-merged"
BASE_MODEL   = "/home/ubuntu/models/Qwen/Qwen2.5-0.5B-Instruct"
TEST_FILE    = "data/ic_test_990.json"
RESULT_FILE  = "assets/eval_result.json"

# ====== 推理配置 ======
BATCH_SIZE     = 32    # 显存不足时调小（如改为 4）
MAX_NEW_TOKENS = 256   # 每条回答最大生成 token 数
MAX_INPUT_LEN  = 512   # 输入截断长度

rouge = Rouge()


def rouge_l_score(pred: str, ref: str) -> float:
    """计算单对预测/参考的 ROUGE-L F1（字符级分词，截断至512字）。"""
    try:
        p = " ".join(list(pred[:512]))
        r = " ".join(list(ref[:512]))
        return rouge.get_scores(p, r)[0]["rouge-l"]["f"]
    except Exception:
        return 0.0


def load_model(model_path: str):
    """加载 tokenizer 和模型，使用 bfloat16 + 自动设备映射。"""
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # 批量推理必须左填充
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def generate_batch(tokenizer, model, instructions: list[str]) -> list[str]:
    """批量推理，返回每条指令对应的生成文本。"""
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": inst}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for inst in instructions
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LEN,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # 只解码新生成的 token，跳过输入部分
    input_len = inputs.input_ids.shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True)
        for out in outputs
    ]


def evaluate(model_path: str, test_data: list[dict], label: str) -> tuple[list, float]:
    """对指定模型跑完整测试集，返回逐条结果和平均 ROUGE-L。"""
    tokenizer, model = load_model(model_path)
    scores = []
    results = []

    for i in range(0, len(test_data), BATCH_SIZE):
        batch        = test_data[i: i + BATCH_SIZE]
        instructions = [item["instruction"] for item in batch]
        preds        = generate_batch(tokenizer, model, instructions)

        for item, pred in zip(batch, preds):
            score = rouge_l_score(pred, item["output"])
            scores.append(score)
            results.append({
                "instruction": item["instruction"],
                "reference":   item["output"],
                "prediction":  pred,
                "rouge_l":     score,
            })

        done = min(i + BATCH_SIZE, len(test_data))
        if done % 50 == 0 or done == len(test_data):
            print(f"[{label}] {done}/{len(test_data)}  avg ROUGE-L: {np.mean(scores):.4f}")

    # 释放显存
    del model
    torch.cuda.empty_cache()
    return results, float(np.mean(scores))


def main():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试集：{len(test_data)} 条\n")

    ft_results,   ft_score   = evaluate(MERGED_MODEL, test_data, "微调后")
    base_results, base_score = evaluate(BASE_MODEL,   test_data, "基座")

    improvement = ft_score - base_score
    print("\n========== 评估结果 ==========")
    print(f"基座模型  ROUGE-L: {base_score:.4f}")
    print(f"微调模型  ROUGE-L: {ft_score:.4f}")
    print(f"提升幅度:          +{improvement:.4f}  ({improvement / base_score * 100:.1f}%)")

    Path(RESULT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_avg_rouge_l": base_score,
                "ft_avg_rouge_l":   ft_score,
                "improvement":      improvement,
                "base_results":     base_results,
                "ft_results":       ft_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n详细结果保存至 {RESULT_FILE}")


if __name__ == "__main__":
    main()
