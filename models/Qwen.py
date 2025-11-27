# models/qwen3_llm.py
from typing import Dict, Any, List, Optional, Sequence

import torch

from modelscope import AutoModelForCausalLM, AutoTokenizer  # 如果你用的是 transformers，就改成 transformers 里的
from .base_llm import BaseLLM


class Qwen3LLM(BaseLLM):
    """
    针对 Qwen3-8B 的实现：
    - 用 ModelScope 的 AutoModelForCausalLM / AutoTokenizer 加载 Qwen3
    - 训练时用 HF/ModelScope 内置的 loss
    - 推理时支持 prompt 文本生成
    - 聊天时使用 tokenizer.apply_chat_template 来构造 prompt
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: Optional[str] = None,
        torch_dtype: str = "auto",
    ):
        """
        model_name: 模型名称或本地路径（比如 "Qwen/Qwen3-8B" 或 "./qwen3-8b-finetuned"）
        device: "cuda" / "cuda:0" / "cpu"，不填则自动选择
        torch_dtype: "auto" / "float16" / "bfloat16" 等
        """
        super().__init__(device=device)

        # 1) 加载 tokenizer
        #   - Qwen 的 tokenizer 一般支持 apply_chat_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 2) 加载模型
        # 这里给出一个简单写法；显存紧张时你可以改为 device_map="auto" 等更高级配置
        if torch_dtype == "auto":
            dtype = "auto"
        else:
            # 字符串转成真正的 torch.dtype
            dtype = getattr(torch, torch_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,  # 简单情况：单卡，把参数都放到 self.device
        ).to(self.device)


    def freeze_layers(self, layer_indices: Sequence[int]):
        """
        冻结 Qwen 的若干 Transformer 层。

        Qwen 大多数情况下参数名前缀类似：
        - "transformer.layers.0."
        - "transformer.layers.1."
        所以我们用这个模式构造前缀。
        如有不一致，可以 print 一下：
        for name, _ in self.model.named_parameters(): print(name)
        """
        prefixes = [f"transformer.layers.{i}." for i in layer_indices]
        self.freeze_by_prefixes(prefixes)

    def unfreeze_layers(self, layer_indices: Sequence[int]):
        prefixes = [f"transformer.layers.{i}." for i in layer_indices]
        self.unfreeze_by_prefixes(prefixes)

    # ========= 2) 从 prompt 文本拿指定层 hidden states =========
    def get_hidden_states(
        self,
        prompt: str,
        layer_indices: Optional[Sequence[int]] = None,
        last_token_only: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        给一段文本 prompt，返回指定层的 hidden states。

        返回：
        - dict: {layer_index: hidden_tensor}
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        return self.hidden_states_from_inputs(
            inputs,
            layer_indices=layer_indices,
            last_token_only=last_token_only,
        )

    # ========= 训练：计算 loss =========
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        约定：
        batch 至少包含:
        - input_ids: [B, T]
        - attention_mask: [B, T] (如果你没用，可以不传)
        - labels: [B, T]

        通常你在 Dataset 里就会根据 tokenizer.encode 好这些东西。
        """
        # 把张量移到和模型一样的 device
        model_inputs = {
            k: v.to(self.device) for k, v in batch.items()
        }

        # 大部分 CausalLM（包括 Qwen）都支持 labels 参数，并自动计算交叉熵 loss
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        return loss

    # ========= 推理：给 prompt，生成一段文本 =========
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        **gen_kwargs: Any,
    ) -> str:
        """
        用于“单轮”生成：你给一段纯文本 prompt，它给你生成一段回复文本。
        后面 chat() 会先把多轮 messages 变成一个 prompt，再调这个函数。
        """
        self.model.eval()

        # 编码到张量
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        # 一些常用的生成参数（可以在调用时覆盖）：
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
            )[0]  # [T_total]

        # 只 decode 新生成的部分，避免把 prompt 再打印一遍
        new_token_ids = output_ids[input_len:]
        text = self.tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
        )
        return text

    # ========= 多轮对话：重写 BaseLLM.chat 的行为 =========
    def _build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        对 Qwen 来说，推荐使用 tokenizer.apply_chat_template，
        这样就能用到官方内置的对话模板（包括 system / user / assistant 角色）。
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # 如果你不需要“思维链/思考模式”，就不要加 enable_thinking
            # enable_thinking=True,
        )
        return text

    # 也可以选择不 override chat，用父类的 chat() 即可
    # 由于我们 override 了 _build_prompt_from_messages，
    # BaseLLM.chat 会先调用这个函数，再调 generate(prompt)。

    # ========= 保存：方便微调完之后下次直接加载 =========
    def save_pretrained(self, save_dir: str):
        """
        将模型和 tokenizer 保存到本地目录，方便下次直接从这个目录加载。
        """
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
