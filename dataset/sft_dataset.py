import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


@dataclass
class SFTConfig:
    """监督微调数据集配置类"""
    dataset_name: str                          # 数据集名称
    max_seq_length: int = 2048                # 最大序列长度
    train_ratio: float = 0.8                  # 训练集比例
    val_ratio: float = 0.1                    # 验证集比例
    test_ratio: float = 0.1                   # 测试集比例
    shuffle: bool = True                      # 是否打乱数据
    seed: int = 42                            # 随机种子
    prompt_template: Optional[str] = None     # 自定义提示模板
    response_template: Optional[str] = None   # 自定义回复模板
    tokenizer_name: Optional[str] = None      # 分词器名称（用于后续集成）


@dataclass
class SFTDataItem:
    """单条监督微调数据项"""
    prompt: str                               # 输入提示
    response: str                             # 期望回复
    input_ids: Optional[List[int]] = field(default=None)    # 编码后的输入ID
    attention_mask: Optional[List[int]] = field(default=None) # 注意力掩码
    labels: Optional[List[int]] = field(default=None)        # 标签（用于监督学习）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


class SFTDataset(Dataset):
    """
    监督微调数据集类 (Supervised Fine-Tuning Dataset)
    
    该类提供了完整的监督微调数据集处理功能，包括：
    - 数据加载和预处理
    - 序列长度限制和截断
    - 训练/验证/测试集分割
    - 批处理和迭代
    - 与prompt模板系统集成
    
    适用于大语言模型的监督微调训练流程。
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        config: SFTConfig,
        split: str = "train",
        tokenizer=None,
        **kwargs
    ):
        """
        初始化SFTDataset
        
        Args:
            data_path: 数据文件路径（支持JSONL、JSON、CSV格式）
            config: 数据集配置
            split: 数据分割类型 ("train", "val", "test")
            tokenizer: 分词器（可选，用于预处理）
            **kwargs: 额外的参数
        """
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.data_items: List[SFTDataItem] = []
        
        # 验证配置
        self._validate_config()
        
        # 加载原始数据
        self.raw_data = self._load_raw_data()
        
        # 预处理数据
        self._preprocess_data()
        
        # 分割数据集
        self._split_dataset()
        
        print(f"成功加载 {len(self.data_items)} 条{split}数据")
    
    def _validate_config(self):
        """验证配置参数的合法性"""
        assert 0 < self.config.train_ratio <= 1, "训练集比例应在(0,1]范围内"
        assert 0 <= self.config.val_ratio <= 1, "验证集比例应在[0,1]范围内"
        assert 0 <= self.config.test_ratio <= 1, "测试集比例应在[0,1]范围内"
        
        total_ratio = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        assert abs(total_ratio - 1.0) < 1e-6, f"分割比例总和应为1，实际为{total_ratio}"
        
        assert self.config.max_seq_length > 0, "最大序列长度必须大于0"
    
    def _load_raw_data(self) -> List[Dict]:
        """加载原始数据文件"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        suffix = self.data_path.suffix.lower()
        
        if suffix == '.jsonl':
            return self._load_jsonl()
        elif suffix == '.json':
            return self._load_json()
        elif suffix == '.csv':
            return self._load_csv()
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _load_jsonl(self) -> List[Dict]:
        """加载JSONL格式数据"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"警告: 跳过无效的JSON行: {line[:50]}... 错误: {e}")
        return data
    
    def _load_json(self) -> List[Dict]:
        """加载JSON格式数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                if 'data' in data:
                    return data['data']
                elif 'conversations' in data:
                    return data['conversations']
                else:
                    return [data]
            return data
    
    def _load_csv(self) -> List[Dict]:
        """加载CSV格式数据"""
        df = pd.read_csv(self.data_path)
        return df.to_dict('records')
    
    def _preprocess_data(self):
        """预处理数据"""
        for item in self.raw_data:
            try:
                # 提取prompt和response
                prompt, response = self._extract_prompt_response(item)
                
                # 创建数据项
                data_item = SFTDataItem(
                    prompt=prompt,
                    response=response,
                    metadata={
                        'original_index': len(self.data_items),
                        'data_source': str(self.data_path),
                        'split': self.split
                    }
                )
                
                # 如果提供了分词器，进行编码
                if self.tokenizer:
                    self._tokenize_item(data_item)
                
                self.data_items.append(data_item)
                
            except Exception as e:
                print(f"警告: 跳过无效数据项: {item} 错误: {e}")
                continue
    
    def _extract_prompt_response(self, item: Dict) -> tuple:
        """从原始数据中提取prompt和response"""
        
        # 常见的数据格式模式
        patterns = [
            # 标准格式
            ('prompt', 'response'),
            ('input', 'output'),
            ('question', 'answer'),
            ('instruction', 'completion'),
            # 对话格式
            ('user', 'assistant'),
            ('human', 'gpt'),
            # Alpaca格式
            ('instruction', 'input', 'output'),
            ('instruction', 'output'),
        ]
        
        for pattern in patterns:
            if len(pattern) == 2:
                prompt_key, response_key = pattern
                if prompt_key in item and response_key in item:
                    prompt = str(item[prompt_key]) if item[prompt_key] else ""
                    response = str(item[response_key]) if item[response_key] else ""
                    
                    # 应用自定义模板（如果提供）
                    if self.config.prompt_template:
                        prompt = self.config.prompt_template.format(prompt=prompt)
                    if self.config.response_template:
                        response = self.config.response_template.format(response=response)
                    
                    return prompt, response
            
            elif len(pattern) == 3:
                instruction_key, input_key, output_key = pattern
                if all(key in item for key in pattern):
                    instruction = str(item[instruction_key])
                    input_text = str(item.get(input_key, ""))
                    output_text = str(item[output_key])
                    
                    # 构建标准格式
                    if input_text and input_text.strip():
                        prompt = f"{instruction}\n\n输入: {input_text}"
                    else:
                        prompt = instruction
                    
                    response = output_text
                    
                    # 应用自定义模板
                    if self.config.prompt_template:
                        prompt = self.config.prompt_template.format(prompt=prompt)
                    if self.config.response_template:
                        response = self.config.response_template.format(response=response)
                    
                    return prompt, response
        
        # 如果没有匹配到任何模式，尝试直接使用所有字段
        raise ValueError(f"无法从数据项中提取prompt和response: {item}")
    
    def _tokenize_item(self, item: SFTDataItem):
        """对数据项进行分词"""
        if not self.tokenizer:
            return
        
        try:
            # 组合prompt和response
            full_text = item.prompt + item.response
            
            # 分词
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length,
                return_tensors=None
            )
            
            item.input_ids = encoding['input_ids']
            item.attention_mask = encoding['attention_mask']
            
            # 创建标签（prompt部分用-100填充，不参与loss计算）
            prompt_length = len(self.tokenizer.encode(item.prompt, add_special_tokens=False))
            labels = [-100] * prompt_length + item.input_ids[prompt_length:]
            
            # 确保标签长度与input_ids一致
            if len(labels) != len(item.input_ids):
                labels = labels[:len(item.input_ids)]
            
            item.labels = labels
            
        except Exception as e:
            print(f"警告: 分词失败 - {e}")
            # 分词失败时使用默认值
            item.input_ids = []
            item.attention_mask = []
            item.labels = []
    
    def _split_dataset(self):
        """分割数据集"""
        if self.split == "train":
            return  # 不需要分割
        
        # 计算分割点
        total_size = len(self.raw_data)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        
        # 设置随机种子
        random.seed(self.config.seed)
        indices = list(range(total_size))
        
        if self.config.shuffle:
            random.shuffle(indices)
        
        if self.split == "val":
            # 验证集：训练集之后的部分
            start_idx = train_size
            end_idx = train_size + val_size
        elif self.split == "test":
            # 测试集：最后部分
            start_idx = train_size + val_size
            end_idx = total_size
        else:
            raise ValueError(f"未知的分割类型: {self.split}")
        
        # 选择对应索引的数据项
        selected_indices = indices[start_idx:end_idx]
        self.data_items = [self.data_items[i] for i in selected_indices]
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> SFTDataItem:
        """获取数据项"""
        return self.data_items[idx]
    
    def get_batch(self, batch_size: int, shuffle: bool = None) -> Iterator[List[SFTDataItem]]:
        """获取批次数据"""
        shuffle = shuffle if shuffle is not None else (self.split == "train")
        
        indices = list(range(len(self.data_items)))
        if shuffle:
            random.seed(self.config.seed)
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield [self.data_items[j] for j in batch_indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.data_items:
            return {}
        
        prompt_lengths = [len(item.prompt) for item in self.data_items]
        response_lengths = [len(item.response) for item in self.data_items]
        total_lengths = [len(item.prompt) + len(item.response) for item in self.data_items]
        
        return {
            'total_samples': len(self.data_items),
            'prompt_stats': {
                'mean': sum(prompt_lengths) / len(prompt_lengths),
                'min': min(prompt_lengths),
                'max': max(prompt_lengths)
            },
            'response_stats': {
                'mean': sum(response_lengths) / len(response_lengths),
                'min': min(response_lengths),
                'max': max(response_lengths)
            },
            'total_length_stats': {
                'mean': sum(total_lengths) / len(total_lengths),
                'min': min(total_lengths),
                'max': max(total_lengths),
                'samples_exceeding_limit': sum(1 for l in total_lengths if l > self.config.max_seq_length)
            },
            'split': self.split,
            'data_source': str(self.data_path)
        }
    
    def save_processed(self, output_path: Union[str, Path]):
        """保存处理后的数据"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_data = []
        for item in self.data_items:
            data_dict = {
                'prompt': item.prompt,
                'response': item.response,
                'input_ids': item.input_ids,
                'attention_mask': item.attention_mask,
                'labels': item.labels,
                'metadata': item.metadata
            }
            processed_data.append(data_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"处理后的数据已保存到: {output_path}")


def create_sft_dataloader(
    dataset: SFTDataset,
    batch_size: int = 8,
    shuffle: bool = None,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    创建PyTorch DataLoader
    
    Args:
        dataset: SFTDataset实例
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        **kwargs: DataLoader的其他参数
    
    Returns:
        DataLoader实例
    """
    from torch.utils.data import DataLoader
    
    shuffle = shuffle if shuffle is not None else (dataset.split == "train")
    
    def collate_fn(batch: List[SFTDataItem]) -> Dict[str, torch.Tensor]:
        """自定义批次处理函数"""
        if dataset.tokenizer is None:
            # 没有分词器时返回原始数据
            return {
                'prompts': [item.prompt for item in batch],
                'responses': [item.response for item in batch],
                'metadata': [item.metadata for item in batch]
            }
        
        # 有分词器时返回张量
        input_ids = [item.input_ids for item in batch if item.input_ids]
        attention_masks = [item.attention_mask for item in batch if item.attention_mask]
        labels = [item.labels for item in batch if item.labels]
        
        # 填充到相同长度
        max_length = max(len(ids) for ids in input_ids) if input_ids else 0
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for ids, mask, lbls in zip(input_ids, attention_masks, labels):
            padding_length = max_length - len(ids)
            padded_input_ids.append(ids + [dataset.tokenizer.pad_token_id] * padding_length)
            padded_attention_masks.append(mask + [0] * padding_length)
            padded_labels.append(lbls + [-100] * padding_length)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )


# 使用示例
if __name__ == "__main__":
    # 示例用法
    config = SFTConfig(
        dataset_name="example_sft",
        max_seq_length=1024,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        prompt_template="请回答以下问题：{prompt}",
        response_template="回答：{response}"
    )
    
    # 假设有一个训练数据文件
    # dataset = SFTDataset("path/to/data.jsonl", config, split="train")
    
    # 获取统计信息
    # stats = dataset.get_statistics()
    # print("数据集统计:", stats)
    
    # 获取批次数据
    # for batch in dataset.get_batch(batch_size=4):
    #     print(f"批次大小: {len(batch)}")
    #     for item in batch:
    #         print(f"Prompt长度: {len(item.prompt)}, Response长度: {len(item.response)}")
    
    print("SFTDataset类已定义完成！")