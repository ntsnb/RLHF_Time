"""
RLHF训练器模块

提供完整的监督微调(SFT)训练流程，包括：
- 全量微调和参数高效微调(PEFT)
- 梯度累积和学习率调度
- 模型保存和加载
- 训练监控和日志记录
- 多轮评估和早停

支持Qwen、Llama等主流大语言模型的微调。
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
import numpy as np

from models.base_llm import BaseLLM
from dataset.sft_dataset import SFTDataset, SFTConfig, create_sft_dataloader
from loss import SFTLoss, PairWiseLoss, PolicyLoss, ValueLoss, KDLoss


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 基础配置
    experiment_name: str = "sft_experiment"
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    seed: int = 42
    
    # 模型配置
    model_name: str = "Qwen/Qwen3-8B"
    model_type: str = "Qwen"  # "Qwen" 或 "Llama"
    device: Optional[str] = None
    torch_dtype: str = "bfloat16"
    
    # 训练参数
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    max_train_steps: Optional[int] = None
    warmup_steps_ratio: float = 0.1  # warmup步数占总训练步数的比例
    max_seq_length: int = 2048
    
    # 优化器参数
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant"
    
    # 训练策略
    freeze_layers: List[int] = field(default_factory=list)  # 要冻结的层索引
    use_amp: bool = True  # 是否使用自动混合精度
    use_gradient_checkpointing: bool = False
    
    # PEFT配置 (可选)
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # 评估配置
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "no", "steps", "epoch"
    save_steps: int = 500
    save_strategy: str = "steps"  # "no", "steps", "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 早停配置
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # 数据配置
    train_file: str = "dataset_train.jsonl"
    eval_file: str = "dataset_eval.jsonl"
    prompt_template: Optional[str] = None
    response_template: Optional[str] = None
    
    # 日志配置
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # 推理配置（用于生成评估）
    generation_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1
    })


class SFTrainer:
    """
    监督微调训练器
    
    支持全量微调和参数高效微调，提供完整的训练、评估和保存功能。
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[BaseLLM] = None,
        train_dataset: Optional[SFTDataset] = None,
        eval_dataset: Optional[SFTDataset] = None,
        tokenizer=None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            model: 模型实例，如果为None则根据config自动创建
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            tokenizer: 分词器
            callbacks: 回调函数列表
        """
        self.config = config
        self.device = torch.device(
            config.device if config.device 
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf') if not config.greater_is_better else float('-inf')
        self.patience_counter = 0
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 初始化日志
        self._setup_logging()
        
        # 初始化模型
        if model is None:
            model = self._create_model()
        self.model = model.to(self.device)
        
        # 初始化数据集
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 初始化优化器和调度器
        self.optimizer, self.scheduler = self._setup_optimizer_and_scheduler()
        
        # 初始化损失函数
        self.criterion = SFTLoss()
        
        # 初始化AMP缩放器
        self.scaler = GradScaler() if config.use_amp else None
        
        # 回调函数
        self.callbacks = callbacks or []
        
        # 训练状态
        self.state = {
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,
            "is_local_master": True,  # 多GPU时的简化处理
        }
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"训练器初始化完成，设备: {self.device}")
        self.logger.info(f"模型类型: {config.model_type}, 模型名称: {config.model_name}")
        
        # 打印模型参数信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_logging(self):
        """设置日志记录"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            Path(self.config.logging_dir) / f"{self.config.experiment_name}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 清除已有处理器并添加新处理器
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _create_model(self) -> BaseLLM:
        """根据配置创建模型"""
        if self.config.model_type == "Qwen":
            from models.Qwen import Qwen3LLM
            model = Qwen3LLM(
                model_name=self.config.model_name,
                device=str(self.device),
                torch_dtype=self.config.torch_dtype,
            )
        elif self.config.model_type == "Llama":
            from models.Llama import Llama3LLM
            model = Llama3LLM(
                model_name=self.config.model_name,
                device=str(self.device),
                torch_dtype=self.config.torch_dtype,
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.config.model_type}")
        
        # 应用训练策略
        if self.config.freeze_layers:
            if self.config.model_type == "Qwen":
                model.freeze_layers(self.config.freeze_layers)
            elif self.config.model_type == "Llama":
                # Llama的层冻结策略需要调整参数名
                self.logger.warning("Llama模型的层冻结功能需要进一步实现")
        
        # 启用梯度检查点
        if self.config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing(model)
        
        # 应用LoRA（如果启用）
        if self.config.use_lora:
            self._apply_lora(model)
        
        return model
    
    def _enable_gradient_checkpointing(self, model: BaseLLM):
        """启用梯度检查点以节省显存"""
        try:
            if hasattr(model.model, 'gradient_checkpointing_enable'):
                model.model.gradient_checkpointing_enable()
                self.logger.info("已启用梯度检查点")
        except Exception as e:
            self.logger.warning(f"启用梯度检查点失败: {e}")
    
    def _apply_lora(self, model: BaseLLM):
        """应用LoRA参数高效微调"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            from peft.tuners.lora import LoraLayer
            
            # LoRA配置
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
            
            # 应用LoRA
            model.model = get_peft_model(model.model, lora_config)
            model.model.print_trainable_parameters()
            
            self.logger.info(f"已应用LoRA，目标模块: {self.config.lora_target_modules}")
            
        except ImportError:
            self.logger.error("请安装peft库以使用LoRA: pip install peft")
            raise
        except Exception as e:
            self.logger.error(f"应用LoRA失败: {e}")
            raise
    
    def _setup_optimizer_and_scheduler(self) -> tuple[Optimizer, _LRScheduler]:
        """设置优化器和学习率调度器"""
        # 过滤出需要梯度的参数
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        
        # 计算总训练步数
        if self.train_dataset is not None:
            num_samples = len(self.train_dataset)
            num_steps_per_epoch = num_samples // (self.config.batch_size * self.config.gradient_accumulation_steps)
            num_train_steps = self.config.num_train_epochs * num_steps_per_epoch
            if self.config.max_train_steps:
                num_train_steps = min(num_train_steps, self.config.max_train_steps)
        else:
            num_train_steps = self.config.max_train_steps or 1000
        
        # 计算warmup步数
        num_warmup_steps = int(num_train_steps * self.config.warmup_steps_ratio)
        
        # 创建调度器
        if self.config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps)
        elif self.config.lr_scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps,
            )
        else:  # constant
            from torch.optim.lr_scheduler import ConstantLR
            scheduler = ConstantLR(optimizer)
        
        self.logger.info(f"设置优化器和调度器，总训练步数: {num_train_steps}")
        
        return optimizer, scheduler
    
    def _create_dataloaders(self) -> tuple[DataLoader, Optional[DataLoader]]:
        """创建数据加载器"""
        # 数据集配置
        dataset_config = SFTConfig(
            dataset_name="sft_dataset",
            max_seq_length=self.config.max_seq_length,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            prompt_template=self.config.prompt_template,
            response_template=self.config.response_template,
            tokenizer_name=self.config.model_name,
        )
        
        # 创建训练数据集
        if self.train_dataset is None:
            if Path(self.config.train_file).exists():
                self.train_dataset = SFTDataset(
                    data_path=self.config.train_file,
                    config=dataset_config,
                    split="train",
                    tokenizer=self.model.tokenizer if hasattr(self.model, 'tokenizer') else None,
                )
        
        # 创建评估数据集
        if self.eval_dataset is None:
            if Path(self.config.eval_file).exists():
                self.eval_dataset = SFTDataset(
                    data_path=self.config.eval_file,
                    config=dataset_config,
                    split="val",
                    tokenizer=self.model.tokenizer if hasattr(self.model, 'tokenizer') else None,
                )
        
        # 创建DataLoader
        train_dataloader = create_sft_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        ) if self.train_dataset else None
        
        eval_dataloader = create_sft_dataloader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        ) if self.eval_dataset else None
        
        self.logger.info(f"创建数据加载器完成，训练样本: {len(self.train_dataset) if self.train_dataset else 0}")
        self.logger.info(f"评估样本: {len(self.eval_dataset) if self.eval_dataset else 0}")
        
        return train_dataloader, eval_dataloader
    
    def train(self) -> Dict[str, Any]:
        """
        执行完整的训练流程
        
        Returns:
            训练结果字典
        """
        self.logger.info("开始训练...")
        
        # 创建数据加载器
        train_dataloader, eval_dataloader = self._create_dataloaders()
        
        if train_dataloader is None:
            raise ValueError("训练数据集不可用，请检查配置")
        
        # 训练前的准备
        self.model.train()
        
        # 计算训练步数
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        if self.config.max_train_steps:
            total_train_steps = min(
                self.config.max_train_steps, 
                self.config.num_train_epochs * num_update_steps_per_epoch
            )
        else:
            total_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        
        num_train_epochs = self.config.max_train_steps // num_update_steps_per_epoch + 1
        
        self.logger.info(f"训练配置:")
        self.logger.info(f"  训练轮数: {num_train_epochs}")
        self.logger.info(f"  每轮步数: {num_update_steps_per_epoch}")
        self.logger.info(f"  总训练步数: {total_train_steps}")
        self.logger.info(f"  梯度累积步数: {self.config.gradient_accumulation_steps}")
        
        # 训练循环
        progress_bar = tqdm(
            total=total_train_steps,
            desc="训练进度",
            disable=not self.state["is_local_master"]
        )
        
        loss_history = []
        self.global_step = 0
        
        for epoch in range(num_train_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            epoch_result = self._train_epoch(train_dataloader, progress_bar)
            loss_history.append(epoch_result["avg_loss"])
            
            # 评估
            if eval_dataloader is not None and self._should_evaluate(epoch, num_train_epochs):
                eval_result = self.evaluate(eval_dataloader)
                self.logger.info(f"Epoch {epoch+1}/{num_train_epochs} - 评估结果: {eval_result}")
                
                # 检查是否需要保存模型
                if self._should_save(epoch, num_train_epochs):
                    self._save_checkpoint(eval_result)
                
                # 检查早停
                if self._check_early_stopping(eval_result):
                    self.logger.info(f"早停触发，在第{epoch+1}轮停止训练")
                    break
            
            # 保存训练历史
            self._save_training_history(loss_history)
        
        progress_bar.close()
        
        # 训练完成
        final_result = {
            "final_loss": loss_history[-1] if loss_history else float('inf'),
            "best_metric": self.best_metric,
            "total_train_steps": self.global_step,
            "num_train_epochs": self.current_epoch + 1,
            "loss_history": loss_history,
        }
        
        self.logger.info(f"训练完成! 最终损失: {final_result['final_loss']:.6f}")
        
        return final_result
    
    def _train_epoch(self, train_dataloader: DataLoader, progress_bar: tqdm) -> Dict[str, float]:
        """训练一个epoch"""
        total_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            # 梯度累积
            if step % self.config.gradient_accumulation_steps != 0:
                return_dict = self._compute_loss(batch)
                loss = return_dict["loss"] / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                # 执行优化步骤
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                self.global_step += 1
                
                # 日志记录
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{return_dict['loss'].item():.6f}",
                        "lr": f"{current_lr:.2e}",
                        "step": self.global_step
                    })
                
                progress_bar.update(1)
                
                # 评估
                if self.config.eval_strategy == "steps" and eval_dataloader is not None:
                    if self.global_step % self.config.eval_steps == 0:
                        eval_result = self.evaluate(eval_dataloader)
                        self.logger.info(f"Step {self.global_step} - 评估结果: {eval_result}")
                
                # 保存检查点
                if self.config.save_strategy == "steps":
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()
            
            # 累计损失
            total_loss += return_dict["loss"].item()
            num_batches += 1
            
            # 手动清理内存
            if self.global_step % 100 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            "avg_loss": avg_loss,
            "total_batches": num_batches,
        }
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        # 准备模型输入
        model_inputs = {k: v.to(self.device) for k, v in batch.items()}
        
        if self.config.use_amp:
            with autocast():
                loss = self.model.compute_loss(model_inputs)
                # 应用LoRA时需要额外的正则化
                if self.config.use_lora:
                    loss = self._add_lora_regularization(loss)
        else:
            loss = self.model.compute_loss(model_inputs)
            if self.config.use_lora:
                loss = self._add_lora_regularization(loss)
        
        return {"loss": loss}
    
    def _add_lora_regularization(self, loss: torch.Tensor) -> torch.Tensor:
        """为LoRA添加额外的正则化"""
        try:
            # 获取LoRA参数
            lora_params = []
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_params.extend([module.lora_A.weight, module.lora_B.weight])
            
            if lora_params:
                # 添加L2正则化
                l2_reg = sum(torch.norm(p, p=2) for p in lora_params)
                loss = loss + 0.01 * l2_reg  # 正则化系数
            
        except Exception as e:
            self.logger.warning(f"LoRA正则化失败: {e}")
        
        return loss
    
    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            eval_dataloader: 评估数据加载器
            
        Returns:
            评估结果字典
        """
        if eval_dataloader is None:
            eval_dataloader, _ = self._create_dataloaders()
        
        if eval_dataloader is None:
            return {"eval_loss": float('inf')}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="评估进度", disable=not self.state["is_local_master"]):
                model_inputs = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.config.use_amp:
                    with autocast():
                        loss = self.model.compute_loss(model_inputs)
                else:
                    loss = self.model.compute_loss(model_inputs)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        result = {"eval_loss": avg_loss}
        
        # 如果配置了生成评估，进行生成评估
        if hasattr(self, 'eval_generation') and callable(self.eval_generation):
            gen_metrics = self.eval_generation()
            result.update(gen_metrics)
        
        self.model.train()  # 切换回训练模式
        
        return result
    
    def _should_evaluate(self, epoch: int, num_epochs: int) -> bool:
        """判断是否应该进行评估"""
        if self.config.eval_strategy == "no":
            return False
        elif self.config.eval_strategy == "epoch":
            return epoch == num_epochs - 1
        elif self.config.eval_strategy == "steps":
            return True  # 在_train_epoch中处理
        return False
    
    def _should_save(self, epoch: int, num_epochs: int) -> bool:
        """判断是否应该保存模型"""
        if self.config.save_strategy == "no":
            return False
        elif self.config.save_strategy == "epoch":
            return epoch == num_epochs - 1
        elif self.config.save_strategy == "steps":
            return self.global_step % self.config.save_steps == 0
        return False
    
    def _check_early_stopping(self, eval_result: Dict[str, float]) -> bool:
        """检查是否触发早停"""
        metric_value = eval_result.get(self.config.metric_for_best_model, float('inf'))
        
        is_better = (
            metric_value < self.best_metric - self.config.early_stopping_threshold
            if not self.config.greater_is_better
            else metric_value > self.best_metric + self.config.early_stopping_threshold
        )
        
        if is_better:
            self.best_metric = metric_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, eval_result: Dict[str, float] = None):
        """保存检查点"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存模型权重
        model_path = checkpoint_dir / "pytorch_model.bin"
        torch.save(self.model.state_dict(), model_path)
        
        # 保存配置
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
        
        # 保存训练状态
        state_path = checkpoint_dir / "trainer_state.json"
        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,
        }
        if eval_result:
            state["eval_result"] = eval_result
        
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"已保存检查点到: {checkpoint_dir}")
    
    def _save_training_history(self, loss_history: List[float]):
        """保存训练历史"""
        history_path = Path(self.config.logging_dir) / f"{self.config.experiment_name}_history.json"
        history = {
            "loss_history": loss_history,
            "best_metric": self.best_metric,
            "config": self.config.__dict__,
        }
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def predict(
        self, 
        prompts: Union[str, List[str]], 
        max_new_tokens: int = 512,
        **generation_kwargs
    ) -> Union[str, List[str]]:
        """
        预测生成文本
        
        Args:
            prompts: 输入提示（单条或列表）
            max_new_tokens: 最大生成token数
            **generation_kwargs: 额外的生成参数
            
        Returns:
            生成结果（单条或列表）
        """
        self.model.eval()
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        results = []
        for prompt in prompts:
            with torch.no_grad():
                if self.config.use_amp:
                    with autocast():
                        result = self.model.generate(
                            prompt, 
                            max_new_tokens=max_new_tokens,
                            **generation_kwargs
                        )
                else:
                    result = self.model.generate(
                        prompt, 
                        max_new_tokens=max_new_tokens,
                        **generation_kwargs
                    )
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_new_tokens: int = 512,
        **generation_kwargs
    ) -> str:
        """
        多轮对话
        
        Args:
            messages: 对话历史
            max_new_tokens: 最大生成token数
            **generation_kwargs: 额外的生成参数
            
        Returns:
            模型回复
        """
        self.model.eval()
        
        with torch.no_grad():
            if self.config.use_amp:
                with autocast():
                    result = self.model.chat(
                        messages, 
                        max_new_tokens=max_new_tokens,
                        **generation_kwargs
                    )
            else:
                result = self.model.chat(
                    messages, 
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs
                )
        
        return result


# 使用示例
if __name__ == "__main__":
    # 1. 创建配置
    config = TrainingConfig(
        experiment_name="example_sft",
        model_name="/path/to/your/model",  # 替换为实际模型路径
        model_type="Qwen",
        train_file="dataset_test/demo_prompts.jsonl",  # 替换为实际数据路径
        eval_file="dataset_test/demo_prompts.jsonl",
        batch_size=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        eval_steps=100,
        save_steps=100,
        output_dir="./checkpoints/example",
    )
    
    # 2. 创建训练器
    trainer = SFTrainer(config)
    
    # 3. 开始训练
    try:
        results = trainer.train()
        print("训练完成!", results)
        
        # 4. 测试预测
        prompts = ["你好，请介绍一下你自己。"]
        predictions = trainer.predict(prompts)
        print("预测结果:", predictions)
        
    except Exception as e:
        print(f"训练失败: {e}")