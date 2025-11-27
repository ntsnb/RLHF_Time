"""
CoT_maker.py - Chain of Thought (CoT) 提示模板生成器

该模块提供了专门的Chain of Thought (CoT)风格提示模板生成器，
用于时间序列预测任务的深入分析和推理过程。
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List

import pandas as pd


# =========================
# 1) CoT风格的Task Definition模板
# =========================

@dataclass
class CoTPromptTemplate:
    """Chain of Thought风格的提示模板"""
    
    task_type: str                    # 任务类型标识
    template_name: str               # 模板名称
    task_definition: str             # 任务定义
    analysis_steps: List[str]        # 分析步骤
    format_instruction: str          # 输出格式指令
    reasoning_structure: str         # 推理结构模板


# 预定义的CoT模板
COT_TEMPLATES: Dict[str, CoTPromptTemplate] = {
    "time_series_forecast": CoTPromptTemplate(
        task_type="time_series_forecast",
        template_name="时间序列预测推理模板",
        task_definition=(
            "你是一个专业的时间序列分析专家。你的任务是分析多变量时间序列数据，"
            "深入理解各变量之间的关系，并基于历史数据预测未来值。"
        ),
        analysis_steps=[
            "1. 数据概览和变量识别",
            "2. 时间序列特性分析（趋势、季节性、周期性）",
            "3. 变量间相关性分析（皮尔逊相关性、滞后相关性）",
            "4. 目标变量与协变量的时滞关系分析",
            "5. 当前环境和 regimes 识别",
            "6. 预测策略设计",
            "7. 不确定性和风险评估"
        ],
        format_instruction=(
            "请按照上述分析步骤，深入分析给定的时间序列数据。"
            "在每个分析步骤中：\n"
            "- 提供具体的分析结果和数值\n"
            "- 解释发现的模式或异常\n"
            "- 说明这些发现对预测的指导意义\n\n"
            "最后提供未来{pred_len}个时间步的完整预测结果。"
            "确保所有预测结果都是基于数据的深度分析和推理得出。"
        ),
        reasoning_structure=(
            "<think>\n"
            "{analysis_process}\n"
            "</think>\n"
            "{final_answer}"
        )
    ),
    
    "environment_analysis": CoTPromptTemplate(
        task_type="environment_analysis",
        template_name="环境分析推理模板",
        task_definition=(
            "你是一个环境系统建模专家。你的任务是从时间序列数据中识别和理解"
            "系统的运行环境、regime特征和内在机制。"
        ),
        analysis_steps=[
            "1. 环境状态变量识别",
            "2. regime shifts 和关键转折点检测",
            "3. 系统稳定性和动态平衡分析",
            "4. 外部冲击和内在调节机制",
            "5. 系统脆弱性和韧性评估"
        ],
        format_instruction=(
            "请基于提供的时间序列数据，系统性地分析系统的运行环境。"
            "重点关注：\n"
            "- 环境的时变特征\n"
            "- 可能的regime shifts\n"
            "- 系统对变化的响应模式\n"
            "- 环境变化对目标变量的影响机制"
        ),
        reasoning_structure=(
            "<think>\n"
            "{environmental_analysis}\n"
            "</think>\n"
            "{prediction_based_on_environment}"
        )
    ),
    
    "covariate_analysis": CoTPromptTemplate(
        task_type="covariate_analysis",
        template_name="协变量分析推理模板",
        task_definition=(
            "你是一个多元统计分析专家。你的任务是对时间序列中的目标变量"
            "和所有协变量进行深入的相关性分析和因果关系探索。"
        ),
        analysis_steps=[
            "1. 变量基本统计特征分析",
            "2. 变量间静态相关性分析",
            "3. 滞后相关性和时间依赖分析",
            "4. 协变量对目标变量的影响强度评估",
            "5. 潜在因果关系识别",
            "6. 多变量相互作用网络分析"
        ],
        format_instruction=(
            "请对时间序列中的所有变量进行全面分析：\n"
            "- 识别主要协变量及其影响模式\n"
            "- 分析目标变量与各协变量的动态关系\n"
            "- 评估变量间的相互依赖性\n"
            "- 提供基于分析结果的预测策略建议"
        ),
        reasoning_structure=(
            "<think>\n"
            "{covariate_analysis_results}\n"
            "</think>\n"
            "{prediction_using_covariate_knowledge}"
        )
    )
}


# =========================
# 2) 增强的数据集配置
# =========================

@dataclass
class CoTDatasetConfig:
    """CoT风格的增强数据集配置"""
    dataset_id: str
    dataset_name: str
    dataset_description: str
    channel_info: str
    target_variable: str             # 明确指定目标变量
    known_covariates: List[str]      # 已知的协变量列表
    time_unit: str = "time step"
    input_len: int = 96
    pred_len: int = 96
    supported_pred_lengths: List[int] = None
    
    def __post_init__(self):
        if self.supported_pred_lengths is None:
            self.supported_pred_lengths = [24, 48, 96, 192, 336, 720]


# 扩展数据集配置
COT_DATASET_CONFIGS: Dict[str, CoTDatasetConfig] = {
    "ETTh1": CoTDatasetConfig(
        dataset_id="ETTh1",
        dataset_name="ETTh1",
        dataset_description=(
            "ETTh1数据集包含电力变压器温度监测数据，具有多变量时序特征。"
            "数据每小时记录一次，包含OT（油温）、HUFL（上层油温）、LULL（下层油温）、"
            "和其他相关传感器数据。"
        ),
        channel_info=(
            "多变量时间序列数据，包含OT（油温）、HUFL（上层油温）、LULL（下层油温）、"
            "LUFL、LLF、UFL、OTU、UHU 等完整变量集合"
        ),
        target_variable="HUFL",  # 指定HUFL为主要目标变量
        known_covariates=["OT", "LULL", "LUFL", "LLF", "UFL", "OTU", "UHU"],  # 其他变量作为协变量
        time_unit="hour",
        input_len=96,
        pred_len=96,
        supported_pred_lengths=[24, 48, 96, 192, 336, 720]
    ),
    
    "ETTh2": CoTDatasetConfig(
        dataset_id="ETTh2", 
        dataset_name="ETTh2",
        dataset_description=(
            "ETTh2是另一个电力变压器温度监测数据集，数据结构与ETTh1相似但时间范围不同。"
        ),
        channel_info=(
            "多变量时间序列数据，包含油温、负载、温度和传感器数据"
        ),
        target_variable="HUFL",
        known_covariates=["OT", "LULL", "LUFL", "LLF", "UFL", "OTU", "UHU"],
        time_unit="hour",
        input_len=96,
        pred_len=96,
        supported_pred_lengths=[24, 48, 96, 192, 336, 720]
    ),
    
    "Electricity": CoTDatasetConfig(
        dataset_id="Electricity",
        dataset_name="Electricity",
        dataset_description=(
            "Electricity数据集包含多个用户的电力消费数据，每小时记录一次。"
            "每个通道对应一个用户的电力消费记录。"
        ),
        channel_info=(
            "多变量时间序列，其中每个通道对应一个用户的电力消费"
        ),
        target_variable="MT_001",  # 默认为第一个用户作为目标
        known_covariates=[],  # 会在数据加载时动态确定
        time_unit="hour",
        input_len=96,
        pred_len=96,
        supported_pred_lengths=[24, 48, 96, 192, 336, 720]
    )
}


# =========================
# 3) CoT提示生成器
# =========================

class CoTTimeSeriesPromptGenerator:
    """
    Chain of Thought风格的时间序列提示生成器
    
    特点：
    - 强调推理过程而非直接输出
    - 引导模型进行系统性分析
    - 支持多种推理模板
    - 明确区分目标和协变量
    """
    
    def __init__(
        self,
        dataset_configs: Dict[str, CoTDatasetConfig],
        cot_templates: Dict[str, CoTPromptTemplate],
        default_template: str = "time_series_forecast"
    ):
        self.dataset_configs = dataset_configs
        self.cot_templates = cot_templates
        self.default_template = default_template
        
        # 默认的CoT格式化指令
        self.default_cot_format = (
            "请按照上述分析步骤进行深入推理：\n"
            "1. 每个步骤都要提供具体的数值分析和发现\n"
            "2. 明确说明推理逻辑和结论依据\n"
            "3. 解释发现对最终预测的影响\n"
            "4. 确保预测结果与前期分析保持一致\n\n"
            "最终预测格式：\n"
            "<answer>\n"
            "```\n完整预测结果（包含所有变量）：\n{timestamp_column_name} {target_variable} {other_covariates}\n{完整的历史数据格式，一行一个时间步}\n```\n"
            "</answer>"
        )
    
    def _pluralize_time_unit(self, time_unit: str) -> str:
        """时间单位复数化"""
        if time_unit.endswith("s"):
            return time_unit
        return time_unit + "s"
    
    def build_cot_prompt(
        self,
        dataset_id: str,
        historical_data: str,
        template_type: Optional[str] = None,
        target_variable: Optional[str] = None,
        covariates: Optional[List[str]] = None
    ) -> str:
        """
        构建CoT风格的提示
        
        Args:
            dataset_id: 数据集标识
            historical_data: 历史数据字符串
            template_type: CoT模板类型
            target_variable: 目标变量名
            covariates: 协变量列表
        
        Returns:
            构建的CoT提示字符串
        """
        if dataset_id not in self.dataset_configs:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")
        
        cfg = self.dataset_configs[dataset_id]
        
        # 选择模板
        template_type = template_type or self.default_template
        if template_type not in self.cot_templates:
            raise ValueError(f"Unknown template_type: {template_type}")
        
        cot_template = self.cot_templates[template_type]
        
        # 获取目标变量和协变量信息
        target_var = target_variable or cfg.target_variable
        known_covars = covariates or cfg.known_covariates
        
        if not known_covars and template_type == "covariate_analysis":
            # 如果没有已知的协变量，需要从数据中推断
            # 这里应该有一个动态推断的逻辑
            pass
        
        time_unit_plural = self._pluralize_time_unit(cfg.time_unit)
        
        # 构建分析步骤说明
        analysis_steps_text = "\n".join([f"- {step}" for step in cot_template.analysis_steps])
        
        # 构建完整的提示
        prompt = (
            f"{cot_template.task_definition}\n\n"
            f"数据集信息：\n"
            f"- 数据集名称：{cfg.dataset_name}\n"
            f"- 数据描述：{cfg.dataset_description}\n"
            f"- 目标变量：{target_var}\n"
            f"- 主要协变量：{', '.join(known_covars) if known_covars else '需要从数据中识别'}\n"
            f"- 时间单位：{cfg.time_unit}\n\n"
            f"分析步骤要求：\n{analysis_steps_text}\n\n"
            f"历史数据（过去{cfg.input_len}个{time_unit_plural}）：\n"
            f"{historical_data}\n\n"
            f"{cot_template.format_instruction.format(pred_len=cfg.pred_len, time_unit_plural=time_unit_plural)}\n\n"
            f"CoT推理模板：\n{cot_template.reasoning_structure}"
        )
        
        return prompt
    
    def generate_multiple_templates(
        self,
        dataset_id: str,
        historical_data: str,
        target_variable: Optional[str] = None,
        covariates: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        生成多种CoT模板的提示
        
        Returns:
            包含多种模板类型的提示字典
        """
        results = {}
        
        for template_type in self.cot_templates.keys():
            prompt = self.build_cot_prompt(
                dataset_id=dataset_id,
                historical_data=historical_data,
                template_type=template_type,
                target_variable=target_variable,
                covariates=covariates
            )
            results[template_type] = prompt
        
        return results


# =========================
# 4) 辅助函数：数据预处理和格式化
# =========================

def format_data_for_cot(
    df: pd.DataFrame,
    target_variable: str,
    date_col: str = "date",
    include_all_variables: bool = True
) -> str:
    """
    为CoT分析格式化了数据
    
    Args:
        df: 输入数据框
        target_variable: 目标变量名
        date_col: 日期列名
        include_all_variables: 是否包含所有变量
    
    Returns:
        格式化后的数据字符串
    """
    if include_all_variables:
        # 包含所有列
        columns = df.columns.tolist()
        value_cols = [col for col in columns if col != date_col]
    else:
        # 只包含目标变量
        value_cols = [target_variable]
    
    # 使用prompt_maker中的make_plain_table函数
    from .prompt_maker import make_plain_table
    return make_plain_table(df, date_col, value_cols)


# =========================
# 5) CLI 主函数
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="生成Chain of Thought风格的时间序列预测提示"
    )
    
    parser.add_argument("--csv_path", type=str, required=True,
                        help="输入的CSV文件路径")
    parser.add_argument("--dataset_id", type=str, default="ETTh1",
                        help="数据集标识")
    parser.add_argument("--template_type", type=str, default="time_series_forecast",
                        choices=list(COT_TEMPLATES.keys()),
                        help="CoT模板类型")
    parser.add_argument("--target_variable", type=str, default=None,
                        help="指定目标变量")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出提示文件路径")
    parser.add_argument("--input_len", type=int, default=96,
                        help="历史窗口长度")
    parser.add_argument("--pred_len", type=int, default=96,
                        help="预测长度")
    
    args = parser.parse_args()
    
    # 加载数据
    df = pd.read_csv(args.csv_path)
    
    # 检测日期列
    if "date" in df.columns:
        date_col = "date"
    else:
        date_col = df.columns[0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 准备配置
    if args.dataset_id in COT_DATASET_CONFIGS:
        cfg = COT_DATASET_CONFIGS[args.dataset_id]
        cfg.input_len = args.input_len
        cfg.pred_len = args.pred_len
    else:
        raise ValueError(f"不支持的数据集: {args.dataset_id}")
    
    # 创建CoT生成器
    generator = CoTTimeSeriesPromptGenerator(COT_DATASET_CONFIGS, COT_TEMPLATES)
    
    # 提取示例数据窗口
    sample_data = df.head(args.input_len)
    
    # 格式化数据
    formatted_data = format_data_for_cot(
        sample_data, 
        target_variable=args.target_variable or cfg.target_variable,
        date_col=date_col
    )
    
    # 生成CoT提示
    cot_prompt = generator.build_cot_prompt(
        dataset_id=args.dataset_id,
        historical_data=formatted_data,
        template_type=args.template_type,
        target_variable=args.target_variable
    )
    
    # 保存提示
    output_data = {
        "prompt": cot_prompt,
        "template_type": args.template_type,
        "dataset_id": args.dataset_id,
        "metadata": {
            "target_variable": args.target_variable or cfg.target_variable,
            "input_len": args.input_len,
            "pred_len": args.pred_len
        }
    }
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"CoT提示已生成并保存到: {args.output_path}")
    print(f"模板类型: {args.template_type}")
    print(f"目标变量: {args.target_variable or cfg.target_variable}")


if __name__ == "__main__":
    main()