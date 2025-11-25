import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List

import pandas as pd


# =========================
# 1) 数据集级配置结构
# =========================

@dataclass
class DatasetPromptConfig:
    dataset_id: str               # 内部 id，如 "ETTh1"
    dataset_name: str             # 在 prompt 中展示的名字，如 "ETTh1"
    dataset_description: str      # Dataset Description 文本
    channel_info: str             # Channel Information 文本
    time_unit: str = "time step"  # "hour" / "15-minute interval" / "day"...
    input_len: int = 96           # 历史长度
    pred_len: int = 96            # 预测长度


# 示例：ETTh1 + Electricity，可以按需继续添加
DATASET_CONFIGS: Dict[str, DatasetPromptConfig] = {
    "ETTh1": DatasetPromptConfig(
        dataset_id="ETTh1",
        dataset_name="ETTh1",
        dataset_description=(
            "The ETTh1 dataset consists of hourly electricity transformer temperature "
            "measurements collected over two years. Each sample contains multiple exogenous "
            "variables such as load, ambient temperature, and other operational indicators."
        ),
        channel_info=(
            "multivariate time-series with all available channels, including OT (oil temperature), "
            "load, ambient temperature and other related measurements from the complete dataset"
        ),
        time_unit="hour",
        input_len=96,
        pred_len=96,
    ),
    "Electricity": DatasetPromptConfig(
        dataset_id="Electricity",
        dataset_name="Electricity",
        dataset_description=(
            "The Electricity dataset contains hourly electricity consumption measurements "
            "for multiple clients over several years. Each time step records the consumption "
            "of each client."
        ),
        channel_info=(
            "multivariate time-series where each channel corresponds to the electricity "
            "consumption of a single client"
        ),
        time_unit="hour",
        input_len=96,
        pred_len=96,
    ),
}


# =========================
# 2) Prompt 生成 Pipeline
# =========================

class TimeR1PromptPipeline:
    """
    基于数据集的 Time-R1 训练模板 prompt 生成器。
    - 固定不变：Task Definition + 总体结构 + 格式说明
    - 可配：各数据集的描述、通道信息、时间粒度、窗口长度等
    """

    def __init__(
        self,
        dataset_configs: Dict[str, DatasetPromptConfig],
        task_definition: Optional[str] = None,
        format_instruction_with_reasoning: Optional[str] = None,
    ):
        self.dataset_configs = dataset_configs

        # 默认 Task Definition（可以在外部传入覆盖）
        self.task_definition = task_definition or (
            "Your task is to forecast future values of a multivariate time series "
            "given its historical observations in a specified dataset."
        )

        # 默认 Format Instruction（带 <think> 和 <answer>，尽量贴近你给的示例）
        self.format_instruction_with_reasoning = format_instruction_with_reasoning or (
            "Please give me the complete data for the next {pred_len} recorded "
            "{time_unit_plural}. Remember to give me the complete data. "
            "You must first conduct reasoning inside <think>...</think>. "
            "When you have the final answer, you can output the answer inside "
            "<answer>...</answer>.\n"
            "Output format for your answer is:\n"
            "<think>\n"
            "Explain the process by which you reach the final answer.\n"
            "</think>\n"
            "<answer>\n"
            "```\nYour forecast data for the next {pred_len} recorded {time_unit_plural}.\n```\n"
            "</answer>"
        )

        # Prompt 的骨架模板：基本不变
        self.base_prompt_template = (
            "{task_definition}\n\n"
            "Here is the {channel_info} data of the {dataset_name} dataset. "
            "The {dataset_name} dataset is {dataset_description}.\n\n"
            "I will now give you historical time series data for the past {input_len} recorded "
            "{time_unit_plural}, and please help me forecast the time series data for the "
            "next {pred_len} recorded {time_unit_plural}. "
            "The historical time series data is as follows:\n"
            "{historical_data}\n\n"
            "{format_instruction}\n"
        )

    def _pluralize_time_unit(self, time_unit: str) -> str:
        """
        简单复数化，例如 'hour' -> 'hours'。
        """
        if time_unit.endswith("s"):
            return time_unit
        return time_unit + "s"

    def build_prompt(
        self,
        dataset_id: str,
        historical_data_str: str,
        *,
        override_task_definition: Optional[str] = None,
        override_format_instruction: Optional[str] = None,
    ) -> str:
        """
        构造单条样本的完整 prompt。
        """
        if dataset_id not in self.dataset_configs:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")

        cfg = self.dataset_configs[dataset_id]

        task_def = override_task_definition or self.task_definition

        # 选择 Format Instruction：默认用带 reasoning 的版本
        fmt_instr_template = (
            override_format_instruction or self.format_instruction_with_reasoning
        )

        time_unit_plural = self._pluralize_time_unit(cfg.time_unit)

        format_instruction = fmt_instr_template.format(
            pred_len=cfg.pred_len,
            time_unit_plural=time_unit_plural,
        )

        prompt = self.base_prompt_template.format(
            task_definition=task_def,
            channel_info=cfg.channel_info,
            dataset_name=cfg.dataset_name,
            dataset_description=cfg.dataset_description,
            input_len=cfg.input_len,
            pred_len=cfg.pred_len,
            time_unit_plural=time_unit_plural,
            historical_data=historical_data_str,
            format_instruction=format_instruction,
        )

        return prompt


# =========================
# 3) 表格序列化函数（无 fence）
# =========================

def make_plain_table(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
) -> str:
    """
    生成不带 ``` 的纯文本表格：

     date HUFL
    2016-08-26 00:00:00 18.018
    ...

    和你 ground_truth 示例保持一致：第一行有一个前导空格。
    明确标明这是时间序列数据的列。
    """
    lines = []

    # 表头：前导空格 + 列名，并明确标注时间序列信息
    header = " " + " ".join([date_col] + value_cols)
    lines.append(header)

    for _, row in df.iterrows():
        ts = pd.to_datetime(row[date_col])
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        values = []
        for col in value_cols:
            val = row[col]
            try:
                val_float = float(val)
            except (TypeError, ValueError):
                val_float = float("nan")
            values.append(f"{val_float:g}")

        line = " ".join([ts_str] + values)
        lines.append(line)

    return "\n".join(lines)


def make_historical_block(
    hist_df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
) -> str:
    """
    历史窗口 -> 带 ``` 的 code block，放进 prompt 中。
    """
    table = make_plain_table(hist_df, date_col=date_col, value_cols=value_cols)
    block = "```\n" + table + "\n```"
    return block


# =========================
# 4) 从文件名智能猜 dataset_id（可被命令行显式覆盖）
# =========================

def infer_dataset_id_from_path(csv_path: Path) -> Optional[str]:
    name = csv_path.name.lower()
    if "etth1" in name:
        return "ETTh1"
    if "electricity" in name:
        return "Electricity"
    return None


# =========================
# 5) CLI 主流程
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Convert standard time-series CSV datasets (e.g., ETTh1, Electricity) "
                    "into Time-R1 style prompt JSONL + ground truth JSONL."
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="输入的 CSV 文件路径，例如 ETTh1.csv 或 electricity.csv")
    parser.add_argument("--prompt_output_path", type=str, required=True,
                        help="输出 prompt JSONL 文件路径，例如 prompts.jsonl")
    parser.add_argument("--answer_output_path", type=str, required=True,
                        help="输出 ground truth JSONL 文件路径，例如 answers.jsonl")
    parser.add_argument("--dataset_id", type=str, default=None,
                        help="数据集 id，用于选择 prompt 配置，例如 ETTh1 或 Electricity。"
                             "若不提供，则尝试根据文件名自动推断。")
    parser.add_argument("--value_cols", type=str, default="",
                        help="逗号分隔的列名，只会把这些列放入历史数据和 ground_truth。"
                             "默认：ETTh1 只用 HUFL，其它数据集使用除日期列外的全部列。")
    parser.add_argument("--input_len", type=int, default=None,
                        help="历史窗口长度，若不指定则使用该数据集配置中的默认值。")
    parser.add_argument("--pred_len", type=int, default=None,
                        help="预测窗口长度，若不指定则使用该数据集配置中的默认值。")
    parser.add_argument("--stride", type=int, default=1,
                        help="滑动步长，默认 1 表示每个时间步都生成一个样本。")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    prompt_output_path = Path(args.prompt_output_path)
    answer_output_path = Path(args.answer_output_path)

    # 1) 确定 dataset_id
    dataset_id = args.dataset_id or infer_dataset_id_from_path(csv_path)
    if dataset_id is None:
        raise ValueError(
            "无法根据文件名推断数据集类型，请显式指定 --dataset_id，例如 --dataset_id ETTh1"
        )

    if dataset_id not in DATASET_CONFIGS:
        raise ValueError(
            f"dataset_id={dataset_id} 在 DATASET_CONFIGS 中没有配置，请先添加相应的数据集描述。"
        )

    # 2) 读 CSV
    df = pd.read_csv(csv_path)

    # 检测时间列
    if "date" in df.columns:
        date_col = "date"
    else:
        # 若没有 date，则默认第一列为时间
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])

    # 3) 决定使用哪些数值列（value_cols）
    if args.value_cols:
        # 用户显式指明：列名用逗号分隔
        raw_cols = [c.strip() for c in args.value_cols.split(",") if c.strip()]
        for c in raw_cols:
            if c not in df.columns:
                raise ValueError(
                    f"value_cols 包含 {c}，但该列不在 CSV 中。现有列：{df.columns.tolist()}"
                )
        value_cols = raw_cols
    else:
        # 没有显式指明 -> 用默认策略
        # 对于所有数据集都使用除时间列外的全部列
        # 这样确保所有列的数据都被包含在prompt中
        value_cols = [c for c in df.columns if c != date_col]

    # 只保留需要的列
    cols_to_keep = [date_col] + value_cols
    df = df[cols_to_keep].copy()

    total_len = len(df)

    # 4) 根据配置确定窗口长度（可被命令行覆盖）
    cfg = DATASET_CONFIGS[dataset_id]
    input_len = args.input_len if args.input_len is not None else cfg.input_len
    pred_len = args.pred_len if args.pred_len is not None else cfg.pred_len

    # 同步回 config，方便 pipeline 在文本里引用
    cfg.input_len = input_len
    cfg.pred_len = pred_len

    window_len = input_len + pred_len

    print(f"数据集: {dataset_id}")
    print(f"总时间步数: {total_len}")
    print(f"每个样本窗口: input_len={input_len}, pred_len={pred_len} -> window_len={window_len}")

    if total_len < window_len:
        raise ValueError("数据长度小于一个窗口长度，无法切分，请检查 input_len/pred_len。")

    # 5) 初始化 Prompt Pipeline
    pipeline = TimeR1PromptPipeline(DATASET_CONFIGS)

    # 6) 滑窗切分 + 生成两个 JSONL
    samples_written = 0
    with prompt_output_path.open("w", encoding="utf-8") as f_prompt, \
         answer_output_path.open("w", encoding="utf-8") as f_answer:

        # [start, start+input_len) -> 历史
        # [start+input_len, start+input_len+pred_len) -> 未来
        for start in range(0, total_len - window_len + 1, args.stride):
            end_hist = start + input_len
            end_total = start + window_len

            hist_df = df.iloc[start:end_hist]
            fut_df = df.iloc[end_hist:end_total]

            # 1) 历史数据块 -> 带 ``` 的 code block
            hist_block = make_historical_block(
                hist_df=hist_df,
                date_col=date_col,
                value_cols=value_cols,
            )

            # 2) 拼完整 prompt content
            content = pipeline.build_prompt(
                dataset_id=dataset_id,
                historical_data_str=hist_block,
            )

            # 3) prompt 样本
            prompt_sample = {
                "content": content,
                "role": "user",
            }

            # 4) ground_truth：未来窗口 -> 不带 ``` 的纯表格
            gt_str = make_plain_table(
                df=fut_df,
                date_col=date_col,
                value_cols=value_cols,
            )
            answer_sample = {
                "ground_truth": gt_str,
                "style": "rule",
            }

            # 写入 JSONL
            f_prompt.write(json.dumps(prompt_sample, ensure_ascii=False) + "\n")
            f_answer.write(json.dumps(answer_sample, ensure_ascii=False) + "\n")

            samples_written += 1

    print(f"已生成样本条数: {samples_written}")
    print(f"prompt JSONL: {prompt_output_path}")
    print(f"answer JSONL: {answer_output_path}")


if __name__ == "__main__":
    main()
