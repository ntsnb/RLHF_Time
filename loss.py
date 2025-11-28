"""
损失函数模块

包含监督微调(SFT)、RLHF和各种相关损失函数的实现。
支持交叉熵损失、成对损失、策略损失、价值损失和知识蒸馏损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class SFTLoss(nn.Module):
    """
    监督微调(SFT)损失函数
    
    标准语言建模损失，对prompt部分使用-100标签忽略计算。
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        """
        初始化SFT损失
        
        Args:
            ignore_index: 忽略的标签索引（通常用于prompt部分）
            label_smoothing: 标签平滑系数
            reduction: 损失聚合方式
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        计算SFT损失
        
        Args:
            logits: 模型输出 logits [batch_size, seq_len, vocab_size]
            labels: 真实标签 [batch_size, seq_len]
            
        Returns:
            损失值
        """
        # 移动到相同设备
        logits = logits.contiguous()
        labels = labels.contiguous()
        
        # 获取有效序列长度（不是ignore_index的部分）
        valid_mask = (labels != self.ignore_index)
        
        # 如果没有有效标签，返回0
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        # 只对有效位置计算损失
        valid_loss = loss[valid_mask.view(-1)]
        
        # 根据reduction聚合损失
        if self.reduction == "mean":
            return valid_loss.mean()
        elif self.reduction == "sum":
            return valid_loss.sum()
        else:  # "none"
            return valid_loss


class PairWiseLoss(nn.Module):
    """
    成对比较损失（用于RLHF中的奖励模型）
    
    比较两个回答的得分，确保被选择回答的得分高于被拒绝回答的得分。
    """
    
    def __init__(
        self,
        margin: float = 0.0,
        loss_type: str = "sigmoid"  # "sigmoid", "hinge", "logistic"
    ):
        """
        初始化成对损失
        
        Args:
            margin: 间隔值
            loss_type: 损失类型
        """
        super().__init__()
        self.margin = margin
        self.loss_type =_loss_type
        
    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        计算成对损失
        
        Args:
            chosen_rewards: 被选择回答的奖励值
            rejected_rewards: 被拒绝回答的奖励值
            
        Returns:
            损失值
        """
        assert chosen_rewards.size() == rejected_rewards.size()
        
        if self.loss_type == "sigmoid":
            # DPO中使用的主要损失函数
            logits = chosen_rewards - rejected_rewards
            loss = -F.logsigmoid(logits).mean()
            
        elif self.loss_type == "hinge":
            # 间隔损失
            logits = chosen_rewards - rejected_rewards - self.margin
            loss = torch.clamp(-logits, min=0).mean()
            
        elif self.loss_type == "logistic":
            # 逻辑损失
            logits = chosen_rewards - rejected_rewards
            loss = F.softplus(-logits).mean()
            
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
        
        return loss


class PolicyLoss(nn.Module):
    """
    策略损失（用于RLHF/PPO）
    
    计算策略网络的损失，基于KL散度进行重要性采样权重校正。
    """
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        """
        初始化策略损失
        
        Args:
            clip_epsilon: PPO截断参数
            value_loss_coef: 价值损失权重系数
            entropy_coef: 熵正则化系数
        """
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
    def forward(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        计算策略损失
        
        Args:
            advantages: 优势函数值
            returns: 回报值
            values: 价值函数输出
            old_log_probs: 旧策略的对数概率
            new_log_probs: 新策略的对数概率
            
        Returns:
            包含各种损失项的字典
        """
        # 计算重要性采样比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO截断损失
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio, 
            1 - self.clip_epsilon, 
            1 + self.clip_epsilon
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # 价值函数损失
        value_loss = F.mse_loss(returns, values)
        
        # 熵损失（用于探索）
        entropy_loss = -(new_log_probs * torch.exp(new_log_probs)).mean()
        
        # 总损失
        total_loss = (
            policy_loss + 
            self.value_loss_coef * value_loss - 
            self.entropy_coef * entropy_loss
        )
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "ratio": ratio.mean()
        }


class ValueLoss(nn.Module):
    """
    价值损失
    
    用于计算价值网络的均方误差损失，支持裁剪版本。
    """
    
    def __init__(
        self,
        clip_param: float = 0.2,
        reduction: str = "mean"
    ):
        """
        初始化价值损失
        
        Args:
            clip_param: 裁剪参数
            reduction: 损失聚合方式
        """
        super().__init__()
        self.clip_param = clip_param
        self.reduction = reduction
        
    def forward(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        计算价值损失
        
        Args:
            values: 价值网络输出
            returns: 目标回报值
            old_values: 旧价值网络的输出（用于裁剪）
            
        Returns:
            价值损失
        """
        if old_values is not None:
            # 裁剪的价值损失
            value_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_param,
                self.clip_param
            )
            
            value_loss_1 = (values - returns) ** 2
            value_loss_2 = (value_clipped - returns) ** 2
            
            value_loss = 0.5 * torch.max(value_loss_1, value_loss_2)
        else:
            # 标准MSE损失
            value_loss = 0.5 * (values - returns) ** 2
        
        if self.reduction == "mean":
            return value_loss.mean()
        elif self.reduction == "sum":
            return value_loss.sum()
        else:
            return value_loss


class KDList(nn.Module):
    """
    知识蒸馏损失
    
    用于将大模型的知识蒸馏到小模型中。
    支持温度缩放的KL散度损失。
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,  # 蒸馏损失权重
        beta: float = 0.3,   # 原始任务损失权重
        reduction: str = "mean"
    ):
        """
        初始化知识蒸馏损失
        
        Args:
            temperature: 知识蒸馏温度
            alpha: 蒸馏损失权重
            beta: 原始任务损失权重
            reduction: 损失聚合方式
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_labels: torch.Tensor,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        计算知识蒸馏损失
        
        Args:
            student_logits: 学生模型的输出
            teacher_logits: 教师模型的输出
            student_labels: 学生模型的目标标签
            temperature: 温度参数（覆盖初始化值）
            
        Returns:
            包含各种损失项的字典
        """
        temp = temperature if temperature is not None else self.temperature
        
        # 教师模型的软目标分布
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        
        # 学生模型的软预测分布
        student_logits_temp = student_logits / temp
        student_probs = F.log_softmax(student_logits_temp, dim=-1)
        
        # KL散度损失（蒸馏损失）
        kl_loss = F.kl_div(
            student_probs, 
            teacher_probs, 
            reduction='none'
        ) * (temp ** 2)
        
        # 原始任务损失
        task_loss = F.cross_entropy(
            student_logits, 
            student_labels, 
            ignore_index=-100
        )
        
        # 总损失
        total_loss = self.alpha * kl_loss + self.beta * task_loss
        
        if self.reduction == "mean":
            kl_loss = kl_loss.mean()
            task_loss = task_loss.mean()
            total_loss = total_loss.mean()
        elif self.reduction == "sum":
            kl_loss = kl_loss.sum()
            task_loss = task_loss.sum()
            total_loss = total_loss.sum()
        
        return {
            "total_loss": total_loss,
            "kl_loss": kl_loss,
            "task_loss": task_loss,
            "alpha": self.alpha,
            "beta": self.beta,
            "temperature": temp
        }


# 损失函数工厂函数
def create_loss(loss_name: str, **kwargs) -> nn.Module:
    """
    创建指定类型的损失函数
    
    Args:
        loss_name: 损失函数名称
        **kwargs: 损失函数参数
        
    Returns:
        损失函数实例
    """
    loss_map = {
        "sft": SFTLoss,
        "pairwise": PairWiseLoss,
        "policy": PolicyLoss,
        "value": ValueLoss,
        "kd": KDLoss,
    }
    
    if loss_name.lower() not in loss_map:
        raise ValueError(f"不支持的损失函数: {loss_name}")
    
    return loss_map[loss_name.lower()](**kwargs)


# 辅助损失函数
class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(
            log_probs, targets, 
            ignore_index=self.ignore_index, 
            reduction='mean'
        )
        
        # 标签平滑
        smooth_loss = -log_probs.mean(dim=-1)
        smooth_loss = smooth_loss[targets != self.ignore_index].mean()
        
        return (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss


class FocalLoss(nn.Module):
    """
    焦点损失（用于处理类别不平衡）
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        # 计算焦点权重
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
