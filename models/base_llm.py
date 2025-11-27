# models/base_llm.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Sequence

import torch
import torch.nn as nn


class BaseLLM(nn.Module, ABC):
    """
    通用大语言模型基类：
    - 既能训练（有 compute_loss）
    - 也能推理 / 聊天（有 generate / chat）
    - 不关心底层是 Qwen / LLaMA / 还是其他，只规定接口
    """

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_by_prefixes(self, prefixes: Sequence[str]):
        """
        根据参数名前缀冻结参数。
        例如：
        - Llama: prefixes = ["model.layers.0.", "model.layers.1."]
        - Qwen:  prefixes = ["transformer.layers.0.", "transformer.layers.1."]

        注意：prefixes 是字符串列表，每个作为 name.startswith(prefix) 的匹配条件。
        """
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in prefixes):
                param.requires_grad = False

    def unfreeze_by_prefixes(self, prefixes: Sequence[str]):
        """跟 freeze_by_prefixes 相反：按前缀解冻。"""
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in prefixes):
                param.requires_grad = True

    def hidden_states_from_inputs(
            self,
            model_inputs: Dict[str, torch.Tensor],
            layer_indices: Optional[Sequence[int]] = None,
            last_token_only: bool = True,
        ) -> Dict[int, torch.Tensor]:
            """
            通用接口：给一批已经 tokenized 的输入，返回指定层的 hidden states。

            参数：
            - model_inputs: 一般来自 tokenizer(..., return_tensors="pt")
            - layer_indices:
                * None: 返回所有层（包括 embedding 层，index=0）
                * [1, 5, -1] 之类：返回指定层
                注意HF里 hidden_states[0] 是 embedding，1..L 是每一层的输出。
            - last_token_only:
                * True: 只取每个样本的最后一个 token 的向量，shape [B, D]
                * False: 返回整序列，shape [B, T, D]

            返回：
            - 一个 dict: {layer_index: hidden_state_tensor}
            """
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

            with torch.no_grad():
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
            # outputs.hidden_states: tuple(len=L+1)，0是embedding层，其后是每一层
            hidden_states = outputs.hidden_states  # tuple of [B, T, D]

            if layer_indices is None:
                layer_indices = list(range(len(hidden_states)))

            result = {}
            for idx in layer_indices:
                h = hidden_states[idx]  # [B, T, D]
                if last_token_only:
                    h = h[:, -1, :]      # [B, D]
                result[idx] = h
            return result

    # ========= 训练相关 =========
    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算一个 batch 的训练 loss。
        约定 batch 至少要包含:
        - input_ids: LongTensor [B, T]
        - attention_mask: LongTensor [B, T] (可选，但常见)
        - labels: LongTensor [B, T]

        由具体子类决定如何喂给底层模型。
        """
        pass

    # ========= 推理 / 生成相关 =========
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        **gen_kwargs: Any,
    ) -> str:
        """
        给定一个文本 prompt，生成一段回复文本。
        子类需要实现：
        - 如何调用底层模型的 generate
        - 如何 decode 输出 tokens
        """
        pass

    # ========= 多轮对话：可选，但很实用 =========
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        **gen_kwargs: Any,
    ) -> str:
        """
        messages 形式统一约定为：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是..."},
            {"role": "user", "content": "继续说"}
        ]

        默认实现：把 messages 转成一个大 prompt，再调 generate。
        像 Qwen 这种有自己 chat 模板的模型，会在子类里 override。
        """
        prompt = self._build_prompt_from_messages(messages)
        return self.generate(prompt, max_new_tokens=max_new_tokens, **gen_kwargs)

    def _build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        默认的、很朴素的多轮对话拼接方式。
        没有模型专用模板时可以用这个。
        比如：
        system: 你是一个助手
        user: 你好
        assistant: 你好，我是助手
        user: 请帮我写点代码
        assistant:
        """
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"{role}: {content}")
        parts.append("assistant: ")
        return "\n".join(parts)

    # ========= 冻结 / 解冻参数，方便微调后只做推理 =========
    def freeze(self):
        """冻结所有参数（不参与训练，用于纯推理阶段）。"""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """解冻所有参数（重新参与训练）。"""
        for p in self.parameters():
            p.requires_grad = True

    # ========= 保存 / 加载封装（本质还是调底层模型 & tokenizer 的方法） =========
    def save_pretrained(self, save_dir: str):
        """
        这里不实现，交给子类去决定如何保存（因为底层可能是HF/ModelScope/OpenAI）。
        对于本地HF/ModelScope模型，通常是 model.save_pretrained + tokenizer.save_pretrained。
        """
        raise NotImplementedError(
            "子类需要实现 save_pretrained，"
            "比如调用 self.model.save_pretrained 和 self.tokenizer.save_pretrained。"
        )
