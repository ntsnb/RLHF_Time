"""
RLHF_time项目训练示例

展示如何使用SFTrainer进行模型微调，包括：
1. 基础监督微调
2. LoRA参数高效微调
3. 冻结部分层的微调
4. 评估和预测
"""

import torch
from pathlib import Path
from trainer import SFTrainer, TrainingConfig
from dataset.sft_dataset import SFTDataset, SFTConfig


def basic_sft_example():
    """
    基础监督微调示例
    """
    print("=== 基础监督微调示例 ===")
    
    # 1. 创建训练配置
    config = TrainingConfig(
        experiment_name="basic_sft_example",
        model_name="Qwen/Qwen3-8B",  # 可以是本地路径或模型名称
        model_type="Qwen",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="bfloat16",
        
        # 数据配置
        train_file="dataset_test/demo_prompts.jsonl",  # 确保文件存在
        eval_file="dataset_test/demo_prompts.jsonl",
        max_seq_length=1024,
        
        # 训练配置
        batch_size=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        warmup_steps_ratio=0.05,
        
        # 评估和保存配置
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        
        # 输出配置
        output_dir="./checkpoints/basic_sft",
        logging_dir="./logs/basic_sft",
    )
    
    # 2. 创建训练器
    trainer = SFTrainer(config)
    
    # 3. 开始训练
    try:
        results = trainer.train()
        print(f"训练完成! 结果: {results}")
        
        # 4. 测试预测
        test_prompts = [
            "请介绍一下机器学习的基本概念。",
            "什么是深度学习？",
            "如何进行模型评估？"
        ]
        
        predictions = trainer.predict(test_prompts)
        for i, (prompt, prediction) in enumerate(zip(test_prompts, predictions)):
            print(f"测试 {i+1}:")
            print(f"  输入: {prompt}")
            print(f"  输出: {prediction}")
            print()
        
        return trainer
        
    except Exception as e:
        print(f"训练失败: {e}")
        return None


def lora_sft_example():
    """
    LoRA参数高效微调示例
    """
    print("=== LoRA参数高效微调示例 ===")
    
    config = TrainingConfig(
        experiment_name="lora_sft_example",
        model_name="Qwen/Qwen3-8B",
        model_type="Qwen",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="bfloat16",
        
        # LoRA配置
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        
        # 训练配置
        batch_size=8,  # LoRA可以支持更大的batch size
        learning_rate=1e-4,  # LoRA通常使用更大的学习率
        num_train_epochs=3,
        gradient_accumulation_steps=2,  # 梯度累积以模拟更大的batch size
        
        # 数据配置
        train_file="dataset_test/demo_prompts.jsonl",
        eval_file="dataset_test/demo_prompts.jsonl",
        max_seq_length=2048,
        
        # 输出配置
        output_dir="./checkpoints/lora_sft",
        logging_dir="./logs/lora_sft",
    )
    
    trainer = SFTrainer(config)
    
    try:
        results = trainer.train()
        print(f"LoRA训练完成! 结果: {results}")
        
        # 测试对话
        messages = [
            {"role": "system", "content": "你是一个专业的AI助手。"},
            {"role": "user", "content": "请帮我写一个简单的Python函数来计算斐波那契数列。"}
        ]
        
        response = trainer.chat(messages)
        print(f"对话测试结果: {response}")
        
        return trainer
        
    except Exception as e:
        print(f"LoRA训练失败: {e}")
        return None


def frozen_layers_example():
    """
    冻结部分层的微调示例
    """
    print("=== 冻结部分层微调示例 ===")
    
    config = TrainingConfig(
        experiment_name="frozen_layers_example",
        model_name="Qwen/Qwen3-8B",
        model_type="Qwen",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="bfloat16",
        
        # 冻结前6层，只微调后面的层
        freeze_layers=list(range(6)),
        
        # 训练配置
        batch_size=4,
        learning_rate=5e-5,  # 冻结层时可以使用更大的学习率
        num_train_epochs=3,
        
        # 数据配置
        train_file="dataset_test/demo_prompts.jsonl",
        eval_file="dataset_test/demo_prompts.jsonl",
        max_seq_length=1024,
        
        # 输出配置
        output_dir="./checkpoints/frozen_sft",
        logging_dir="./logs/frozen_sft",
    )
    
    trainer = SFTrainer(config)
    
    try:
        results = trainer.train()
        print(f"冻结层训练完成! 结果: {results}")
        
        return trainer
        
    except Exception as e:
        print(f"冻结层训练失败: {e}")
        return None


def evaluation_example():
    """
    详细评估示例
    """
    print("=== 详细评估示例 ===")
    
    config = TrainingConfig(
        experiment_name="eval_example",
        model_name="Qwen/Qwen3-8B",
        model_type="Qwen",
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # 数据配置
        eval_file="dataset_test/demo_prompts.jsonl",
        max_seq_length=1024,
    )
    
    trainer = SFTrainer(config)
    
    # 创建评估数据集
    eval_dataset = SFTDataset(
        data_path=config.eval_file,
        config=SFTConfig(
            dataset_name="eval_dataset",
            max_seq_length=config.max_seq_length
        ),
        split="val"
    )
    
    # 执行评估
    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
    
    eval_result = trainer.evaluate(eval_dataloader)
    print(f"评估结果: {eval_result}")
    
    # 批量预测评估
    test_samples = [
        {"prompt": "解释一下什么是人工智能。", "expected_type": "知识问答"},
        {"prompt": "写一个快速排序算法。", "expected_type": "代码生成"},
        {"prompt": "帮我写一首关于春天的诗。", "expected_type": "创意写作"},
    ]
    
    for i, sample in enumerate(test_samples):
        prediction = trainer.predict(sample["prompt"])
        print(f"\n评估样本 {i+1}:")
        print(f"  提示: {sample['prompt']}")
        print(f"  期望类型: {sample['expected_type']}")
        print(f"  模型输出: {prediction}")
        print("-" * 50)


def compare_models_example():
    """
    模型对比示例
    """
    print("=== 模型对比示例 ===")
    
    # 测试不同配置的训练效果
    configs = [
        ("基础微调", TrainingConfig(
            experiment_name="compare_basic",
            batch_size=4,
            learning_rate=2e-5
        )),
        ("LoRA微调", TrainingConfig(
            experiment_name="compare_lora",
            use_lora=True,
            batch_size=8,
            learning_rate=1e-4
        )),
        ("冻结层微调", TrainingConfig(
            experiment_name="compare_frozen",
            freeze_layers=list(range(4)),
            batch_size=4,
            learning_rate=5e-5
        ))
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\n训练配置: {name}")
        config.train_file = "dataset_test/demo_prompts.jsonl"
        config.eval_file = "dataset_test/demo_prompts.jsonl"
        config.output_dir = f"./checkpoints/compare_{name}"
        
        trainer = SFTrainer(config)
        
        try:
            # 简化的训练（用于演示）
            train_dataloader, eval_dataloader = trainer._create_dataloaders()
            
            if eval_dataloader is not None:
                # 执行一次评估
                eval_result = trainer.evaluate(eval_dataloader)
                results[name] = eval_result
                print(f"  评估损失: {eval_result.get('eval_loss', 'N/A')}")
            else:
                print("  没有可用的评估数据")
                
        except Exception as e:
            print(f"  训练失败: {e}")
    
    print(f"\n=== 对比结果 ===")
    for name, result in results.items():
        print(f"{name}: {result}")


def main():
    """
    主函数 - 运行所有示例
    """
    print("RLHF_time项目微调训练示例")
    print("=" * 50)
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("警告: 没有检测到GPU，将使用CPU训练（速度较慢）")
    
    print()
    
    # 检查数据文件
    data_files = ["dataset_test/demo_prompts.jsonl", "dataset_test/answers_20251127_195800.jsonl"]
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✓ 数据文件存在: {file_path}")
        else:
            print(f"✗ 数据文件不存在: {file_path}")
    
    print()
    
    try:
        # 运行各种示例
        examples = [
            # ("基础微调", basic_sft_example),
            # ("LoRA微调", lora_sft_example),
            # ("冻结层微调", frozen_layers_example),
            ("评估示例", evaluation_example),
            # ("模型对比", compare_models_example),
        ]
        
        for name, example_func in examples:
            print(f"\n{'='*20} {name} {'='*20}")
            example_func()
            print()
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n示例运行出错: {e}")
    
    print("所有示例完成!")


if __name__ == "__main__":
    main()