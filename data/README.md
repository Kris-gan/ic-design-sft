# 数据集说明

## 数据来源

本项目数据通过 **Self-Instruct** 方法自动生成，领域为集成电路设计（IC Design）。  
生成模型：通义千问（Qwen）via DashScope API。

---

## 数据流转

```
seeds_20.json              ← 人工精选的20条种子问答
    ↓ gen_instruction.py
instructions_50k_raw.json  ← 自动生成的50k条原始指令
    ↓ filter.py
instructions_50k_filtered  ← 过滤后保留的高质量指令（~952条）
    ↓ gen_output.py
dataset_50k_with_output    ← 附带回答的完整训练数据
    ↓ evaluate.py
eval_result.json           ← ROUGE-L 评估结果
```

---

## 数据格式

每条数据为 JSON 格式，字段如下：

```json
{
  "instruction": "解释什么是建立时间（Setup Time）和保持时间（Hold Time）",
  "input": "",
  "output": "建立时间是指..."
}
```

|字段|类型|说明|
|---|---|---|
|`instruction`|string|指令/问题|
|`input`|string|补充上下文（可为空）|
|`output`|string|期望回答|

---

## 数据规模

|阶段|文件|数量|
|---|---|---|
|种子数据|`seeds_20.json`|20 条|
|原始生成|`instructions_50k_raw.json`|50,000 条|
|过滤后|`seeds_952_filtered.json`|952 条|
|带输出|`dataset_50k_with_output.json`|952 条|

> ⚠️ 完整数据集文件体积较大，已通过 `.gitignore` 排除，未上传至 GitHub。  
> 示例数据见 `sample/sample_data.json`（≤10 条）。

---

## 示例数据

见 [`sample/sample_data.json`](https://claude.ai/chat/sample/sample_data.json)

---

## LLaMA-Factory 数据注册

使用 `configs/lora_config.yaml` 训练前，需要将数据集注册到 LLaMA-Factory：

**第一步**：将完整训练数据复制到 LLaMA-Factory 数据目录：

```bash
cp output/dataset_50k_with_output.json ~/LlamaFactory/data/
```

**第二步**：在 `~/LlamaFactory/data/dataset_info.json` 中添加以下注册项：

```json
"ic_dataset": {
  "file_name": "dataset_50k_with_output.json"
}
```

**第三步**：启动训练：

```bash
llamafactory-cli train configs/lora_config.yaml
```