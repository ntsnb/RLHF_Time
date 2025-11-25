# models/Llama.py
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_llm import BaseLLM


class Llama3LLM(BaseLLM):
    """
    基于 transformers 实现的 Llama 3 8B Instruct 封装：
    - 继承 BaseLLM：compute_loss + generate + chat
    - 底层用 AutoModelForCausalLM / AutoTokenizer
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",  # 显存允许的话优先 bf16/fp16
    ):
        super().__init__(device=device)

        # 1) tokenizer
        # llama 3 一般有自己的 chat_template
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,  # 有的 Llama tokenizer 不支持 fast，保险起见关掉
        )

        # 2) model
        if torch_dtype == "auto":
            dtype = "auto"
        else:
            dtype = getattr(torch, torch_dtype)  # "bfloat16" -> torch.bfloat16

        # 简单单卡加载：显存紧的话，可以之后再改成 device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
        ).to(self.device)

    # ========= 训练：算 loss =========
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch:
        - input_ids: [B, T]
        - attention_mask: [B, T] (可选)
        - labels: [B, T]
        """
        model_inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        return loss

    # ========= 推理：给 prompt 文本，生成一段回复 =========
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        **gen_kwargs: Any,
    ) -> str:
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        default_gen_kwargs = dict(
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        default_gen_kwargs.update(gen_kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **default_gen_kwargs,
            )[0]

        new_ids = output_ids[input_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return text

    # ========= 多轮对话：用 Llama3 自己的 chat_template =========
    def _build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Llama 3 的 HF tokenizer 也内置了 chat 模板，接口和 Qwen 类似：
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text

    # ========= 保存：微调后保存权重 =========
    def save_pretrained(self, save_dir: str):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
