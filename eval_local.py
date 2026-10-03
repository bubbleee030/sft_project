import os
import torch
import json
from tqdm import tqdm
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────
# Dataloader
# ──────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, qid_prompt_list: List[Tuple[str, str, str]]):
        self.data = qid_prompt_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]  # (qid, prompt)

def make_collate_fn(tokenizer):
    def collate_fn(batch):
        qids, prompts, resp = zip(*batch)

        prompts_text = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize = False, add_generation_prompt = True) for prompt in prompts]

        tokenized_inputs = tokenizer(prompts_text, padding=True, return_tensors="pt")
        return qids, tokenized_inputs, prompts, resp
    return collate_fn

# ──────────────────────────────────────────────
# Prompt model
# ──────────────────────────────────────────────

def ask_llm(model,
            tokenizer,
            dataloader: DataLoader,
            output_path: str, 
            max_new_tokens: int = 1024, 
            to_remove_think_tags: bool = True,
            think_tags: str = '</think>'):
    """
    Returns ONLY the newly generated continuation, without echoing the prompt.
    """

    for qids, tokenized_inputs, prompts, resp in tqdm(dataloader, desc="Processing batches", dynamic_ncols=True):
        tokenized_inputs = {k: v.to("cuda") for k, v in tokenized_inputs.items()}   # move input to GPU

        with torch.no_grad():
            out = model.generate(
                **tokenized_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,               # greedy; change to True for sampling
                temperature=None,
                top_p = None,              
                return_dict_in_generate=False, # return generation only, no metadata
            )

        # out = prompt + response
        # extract response only
        input_len = tokenized_inputs['input_ids'].shape[-1] 
        batch_size, seq_len = out.shape
        response = out[:, input_len:]
        # range_ids = torch.arange(seq_len).unsqueeze(0).to(out.device) # shape = [1, seq_len]
        # input_lengths = input_lengths.unsqueeze(1)  # shape = [batch_size, 1]
        # mask = range_ids >= input_lengths # shape = [batch_size, seq_len]
        # masked_out = torch.where(mask, out, pad_token_tensor) # mask input prompts with pad_token

        # decode and save results
        decoded_outputs = tokenizer.batch_decode(response, skip_special_tokens=True)
        if to_remove_think_tags:
            decoded_outputs = [remove_think_tags(output, think_tags) for output in decoded_outputs]
        save_to_jsonl(output_path, qids, prompts, resp, decoded_outputs)
    
    print(f'Model response saved to {output_path}')

def save_to_jsonl(output_path, qids, prompts, resp, decoded_outputs):
    mode = 'w'
    if os.path.exists(output_path):
        mode = 'a'
    with open(output_path, mode) as file:
        for qid, prompt, resp, model_resp in zip(qids, prompts, resp, decoded_outputs):
            file.write(json.dumps({'qid': qid, 'prompt': prompt, 'resp': resp, 'model_resp': model_resp.strip()}, ensure_ascii=False))
            file.write('\n')

def handle_dtype(dtype: str):
    if dtype == 'float32':
        pt_dtype = torch.float32
    elif dtype == 'float16':
        pt_dtype = torch.float16
    elif dtype == 'bf16':
        pt_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return pt_dtype

def remove_think_tags(data, think_tags):
    """Remove prepending thinking process enclosed by <think> ... </think>. If </think> not in the ouput, discard the data by making it ''

    Args:
        data (str): Output from judge model
        think_tags (str): tag used to seperate thinking process and model response

    Returns:
        str: Output after removign <think> ... </think>
    """

    if think_tags in data:
        return data.split(think_tags)[1].strip()
    else:
        return data

def main(
    model_path: str,
    data_path: str,
    output_path: str,
    batch_size: int,
    max_new_tokens: int,
    torch_dtype: str = 'float16',
    adapter_path: str = None,
    prompt_col: str='prompt',
    resp_col: str='resp',
    remove_think_tags: bool = True,
    think_tags: str = '</think>',
    use_sharegpt_format: bool = True,
    temperature: float = 0.0,
    top_p: float = 1.0,
    ):

    dtype = handle_dtype(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side = 'left')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = dtype,                # FP16 weights
        device_map = {"": "cuda"},          # put entire model on GPU
        local_files_only = True,
    )
    
    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model.load_adapter(adapter_path, adapter_name="default")
        print("LoRA adapter loaded successfully.")
        
    model.eval()

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            if use_sharegpt_format:
                # 處理 ShareGPT 格式
                messages = item.get('messages', [])
                user_message = None
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_message = msg.get('content', '')
                        break
                
                if user_message:
                    # 生成一個唯一的 qid
                    qid = f"sharegpt_{idx}"
                    # 使用 user 的 content 作為 prompt
                    prompt = user_message
                    # 沒有參考答案，設為空字串
                    resp = ""
                    data.append((qid, prompt, resp))
            else:
                # 原有格式
                data.append((item['qid'], item[prompt_col], item[resp_col]))

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    collate_fn = make_collate_fn(tokenizer)
    dataset = PromptDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    ask_llm(model, tokenizer, dataloader, output_path, max_new_tokens = max_new_tokens, to_remove_think_tags = remove_think_tags, think_tags = think_tags)

if __name__ == "__main__":
    import fire
    fire.Fire(main)