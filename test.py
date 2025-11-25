# from modelscope import AutoModelForCausalLM, AutoTokenizer

# model_name = "/mnt/sda/home/niutiansen/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

# # prepare the model input
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# # conduct text completion
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# # parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
# print("content:", content)

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# test_chat_qwen3.py
from models.Qwen import Qwen3LLM
from models.Llama import Llama3LLM

def main():
    # 1) 初始化模型：先用原始 Qwen3-8B 测试
    #    如果你以后微调保存到 ./checkpoints/qwen3-8b-finetune
    #    那这里的 model_name 换成那个本地路径即可。
    model = Qwen3LLM(
        model_name="/mnt/sda/home/niutiansen/.cache/modelscope/hub/models/Qwen/Qwen3-8B",        # 或 "Qwen/Qwen3-8B-Instruct" 看你下的哪个
        device="cuda",                     # 没有GPU就改成 "cpu"（会很慢）
        torch_dtype="bfloat16",            # 显存不足可以换 "float16"，再不行就 "auto"
    )

    # model = Llama3LLM(
    #     model_name="/mnt/sda/home/niutiansen/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct",  # 或 "meta-llama/Meta-Llama-3-8B-Instruct"
    #     device="cuda",
    #     torch_dtype="bfloat16",   # 显存不太够可以换 "float16" 或 "auto"
    # )
    for name, param in model.named_parameters():
        print(name)
        # print(param)

    # 2) 冻结参数：防止误操作训练，只做推理
    # model.freeze()


    # 1) 冻结前 4 层
    model.freeze_layers([0, 1, 2, 3])

    # 2) 拿第 1 层 和 最后一层 的 hidden states（最后一个 token）
    hs = model.get_hidden_states(
        prompt="这是一个测试用的句子。",
        layer_indices=[1, -1],      # -1 表示最后一层
        last_token_only=True,
    )
    print("第1层 hidden:", hs[1].shape)    # [1, D]
    print("最后一层 hidden:", hs[-1].shape)

    # 3) 准备一段对话
    messages = [
        {
            "role": "system",
            "content": "你是一个耐心、专业、用中文回答的深度学习助手。",
        },
        {
            "role": "user",
            "content": "你好，请用三句话介绍一下你自己。",
        },
    ]

    # 4) 调用 chat
    reply = model.chat(messages, max_new_tokens=256)
    print("模型回复：\n", reply)


if __name__ == "__main__":
    main()
