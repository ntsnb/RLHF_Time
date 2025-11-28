# RLHF_time项目微调训练器

这是一个完整的大语言模型监督微调(SFT)训练器，支持Qwen、Llama等主流模型的微调。

## 🚀 主要特性

### 训练策略
- **全量微调**: 训练所有模型参数
- **LoRA参数高效微调**: 减少显存占用，支持更大batch size
- **冻结层微调**: 冻结前N层，只训练后面的层

### 核心功能
- **自动混合精度训练**: 支持AMP加速训练
- **梯度累积**: 支持梯度累积以模拟更大batch size
- **学习率调度**: 支持线性、余弦、恒定调度策略
- **早停机制**: 基于评估指标自动停止训练
- **检查点保存**: 支持定期保存训练状态

### 评估和监控
- **实时评估**: 支持按步数或轮数评估
- **训练监控**: 详细的训练日志和进度显示
- **可视化**: 支持TensorBoard日志记录

### 损失函数
- **SFT损失**: 标准语言建模损失
- **成对损失**: 用于奖励模型训练
- **策略损失**: 用于RLHF/PPO训练
- **价值损失**: 价值网络训练
- **知识蒸馏损失**: 模型蒸馏

## 📦 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install transformers datasets
pip install modelscope

# 训练相关
pip install tqdm numpy pandas
pip install tensorboard

# LoRA支持 (可选)
pip install peft

# 其他
pip install scipy scikit-learn
```

## 🛠️ 快速开始

### 1. 基础使用

```python
from trainer import SFTrainer, TrainingConfig

# 创建配置
config = TrainingConfig(
    experiment_name="my_sft_experiment",
    model_name="Qwen/Qwen3-8B",  # 或本地模型路径
    model_type="Qwen",
    train_file="data/train.jsonl",
    eval_file="data/eval.jsonl",
    batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=3,
)

# 创建训练器
trainer = SFTrainer(config)

# 开始训练
results = trainer.train()

# 预测
predictions = trainer.predict(["你好，请介绍一下你自己。"])
```

### 2. LoRA微调

```python
config = TrainingConfig(
    experiment_name="my_lora_experiment",
    model_name="Qwen/Qwen3-8B",
    model_type="Qwen",
    
    # LoRA配置
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    
    # LoRA训练可以使用更大的batch size
    batch_size=8,
    learning_rate=1e-4,
)
```

### 3. 冻结层微调

```python
config = TrainingConfig(
    experiment_name="my_frozen_experiment",
    model_name="Qwen/Qwen3-8B",
    
    # 冻结前6层
    freeze_layers=list(range(6)),
    
    batch_size=4,
    learning_rate=5e-5,
)
```

## 📊 数据格式

### 支持的数据格式

1. **JSONL格式** (推荐)
```jsonl
{"prompt": "用户问题", "response": "模型回答"}
{"prompt": "另一个问题", "response": "另一个回答"}
```

2. **JSON格式**
```json
{
  "data": [
    {"prompt": "问题1", "response": "回答1"},
    {"prompt": "问题2", "response": "回答2"}
  ]
}
```

3. **CSV格式**
```csv
prompt,response
"问题1","回答1"
"问题2","回答2"
```

### 自定义模板

```python
config = TrainingConfig(
    prompt_template="请回答以下问题：{prompt}",
    response_template="回答：{response}",
)
```

## ⚙️ 配置参数

### 基础配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `experiment_name` | str | "sft_experiment" | 实验名称 |
| `model_name` | str | "Qwen/Qwen3-8B" | 模型名称或路径 |
| `model_type` | str | "Qwen" | 模型类型 (Qwen/Llama) |
| `device` | str | None | 设备类型 (cuda/cpu) |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | int | 8 | 批次大小 |
| `learning_rate` | float | 2e-5 | 学习率 |
| `num_train_epochs` | int | 3 | 训练轮数 |
| `max_seq_length` | int | 2048 | 最大序列长度 |
| `gradient_accumulation_steps` | int | 1 | 梯度累积步数 |
| `warmup_steps_ratio` | float | 0.1 | warmup步数占总训练步数的比例 |

### LoRA参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_lora` | bool | False | 是否使用LoRA |
| `lora_rank` | int | 8 | LoRA秩 |
| `lora_alpha` | int | 32 | LoRA缩放参数 |
| `lora_dropout` | float | 0.1 | LoRA dropout率 |
| `lora_target_modules` | List[str] | ["q_proj", "v_proj"] | 目标模块 |

### 优化器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `weight_decay` | float | 0.01 | 权重衰减 |
| `adam_beta1` | float | 0.9 | Adam参数β1 |
| `adam_beta2` | float | 0.999 | Adam参数β2 |
| `adam_epsilon` | float | 1e-8 | Adam参数ε |
| `max_grad_norm` | float | 1.0 | 最大梯度范数 |

## 🔧 高级用法

### 自定义损失函数

```python
from loss import create_loss

# 创建自定义损失的模型
loss_fn = create_loss("sft", label_smoothing=0.1)
# 或使用其他损失类型: "pairwise", "policy", "value", "kd"
```

### 回调函数

```python
def my_callback(trainer):
    print(f"训练步骤: {trainer.global_step}")
    # 自定义逻辑

trainer.callbacks.append(my_callback)
```

### 模型保存和加载

```python
# 保存模型
trainer.model.save_pretrained("./checkpoints/my_model")

# 加载训练器状态
trainer.load_from_checkpoint("./checkpoints/checkpoint-1000")
```

### 评估和生成

```python
# 评估
eval_result = trainer.evaluate()
print(f"评估结果: {eval_result}")

# 单轮对话
response = trainer.chat([
    {"role": "system", "content": "你是一个AI助手。"},
    {"role": "user", "content": "你好"}
])

# 批量生成
prompts = ["问题1", "问题2", "问题3"]
responses = trainer.predict(prompts, max_new_tokens=512)
```

## 📁 项目结构

```
RLHF_time/
├── trainer.py           # 主训练器
├── trainer_example.py   # 使用示例
├── loss.py              # 损失函数
├── dataset/
│   ├── sft_dataset.py   # 数据集处理
│   ├── prompt_maker.py  # 提示模板
│   └── ...
├── models/
│   ├── base_llm.py      # 基础模型类
│   ├── Qwen.py          # Qwen模型实现
│   └── Llama.py         # Llama模型实现
└── checkpoints/         # 模型检查点
```

## 🐛 常见问题

### 1. 显存不足
- 使用LoRA微调：`use_lora=True`
- 减小batch_size
- 启用梯度检查点：`use_gradient_checkpointing=True`
- 使用自动混合精度：`use_amp=True`

### 2. 训练速度慢
- 确保使用GPU
- 启用自动混合精度：`use_amp=True`
- 适当增大batch_size
- 使用多GPU（如果可用）

### 3. 模型不收敛
- 检查学习率是否合适
- 调整warmup_steps_ratio
- 检查数据质量
- 调整正则化参数

### 4. 内存泄露
- 定期调用：`torch.cuda.empty_cache()`
- 减小序列长度：`max_seq_length`
- 关闭梯度检查点

## 📚 参考资料

- [Qwen模型文档](https://huggingface.co/Qwen)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [PEFT库文档](https://huggingface.co/docs/peft)
- [Transformers库](https://huggingface.co/docs/transformers)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License