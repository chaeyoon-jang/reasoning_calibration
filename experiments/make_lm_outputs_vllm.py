import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json 
import datasets
import pandas as pd
from tqdm.autonotebook import tqdm 
from transformers import set_seed
from vllm import LLM, SamplingParams 

from src.logging import entrypoint
from src.llm_model_utils import create_tokenizer
from src.prompt_hub import confidence_prompts, confidence_prompts_no_reasoning


def generate_outputs(
    model,
    tokenizer,
    base_prompt,
    questions,
    batch_size,
    sampling_params,
    template_add=True):
    
    if template_add:    
        if (
            tokenizer.name_or_path
            and (("Llama-3" in tokenizer.name_or_path) or ("Qwen" in tokenizer.name_or_path))
            and ("Instruct" in tokenizer.name_or_path)
        ):
            msgs = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": base_prompt}]
            
            base_prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
    
    questions = [base_prompt.replace("<question>", q) for q in questions]

    ## Generate outputs.
    answers = []
    for idx in tqdm(range(0, len(questions), batch_size)):
            
        if idx + batch_size > len(questions):
            batch_prompt = questions[idx:]
        else:
            batch_prompt = questions[idx:idx+batch_size]
        
        outputs = model.generate(batch_prompt, 
                                 sampling_params,
                                 use_tqdm=False)
        outputs = [outputs[i].outputs[0].text for i in range(len(outputs))]
        answers.extend(outputs)
    
        if idx == 0:
            print(f"Input prompt:\n {questions[0]}")
            print(f"Output:\n {answers[0]}")
        
    return questions, answers


@entrypoint(with_wandb=False)
def main(
    seed: int = 0,
    log_dir: str = None,
    data_dir: str = "./make_math_data", #data/MATH/train
    data_name: str = "gsm", # math
    data_type: str = "train",
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    max_new_tokens: int = 1024,
    do_sample: bool = False, 
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    c_type: str = "base",
    reasoning: bool = False,
    template_add: bool = True,
):
    
    set_seed(seed)        
    ############################# Loading datasets #############################
    #data_path = os.path.join(data_dir, data_name + '_' + data_type + ".json")
    #if os.path.exists(data_path):
    #    data = json.load(open(data_path))
        
    #else:
    #    raise FileNotFoundError(f"No files found named: {data_path}")  
    if data_name == "gsm":
        data = datasets.load_dataset('openai/gsm8k', 'main')[data_type]
    
    elif data_name == "mmlu":
        
        data_paths = os.listdir(data_dir)
        data = []
        for p in data_paths:
            temp_data = pd.read_csv(os.path.join(data_dir, p), header=None)
            temp_data.columns = ['question', 'A', 'B', 'C', 'D', 'true_answer']
            answer = [temp_data[a][i] for i, a in enumerate(temp_data['true_answer'])]
            question = temp_data['question']
            gt_data = [{'question': question[i], 'answer': answer[i]} for i in range(len(answer))]
            data.extend(gt_data)
    
    elif data_name == "math":
        #data = pd.read_csv(f'/mnt/home/chaeyun-jang/reasoning_calibration/syn_data/addition_game_{c_type}_train.csv')
        data = pd.read_csv(f"./data/syn_test/test.csv")
        data = [{'question': data['question'][i], 'answer': data['answer'][i]} for i in range(len(data))]
        
    if data_name == "squad":
        ds = datasets.load_dataset("rajpurkar/squad")[data_type]
        def extract_example(example):
            return {
                'question': example['question'],
                'answer': example['answers']['text'][0] if example['answers']['text'] else ""
            }
        data = ds.map(extract_example, remove_columns=ds.column_names)
    ######################## Loading tokenizer & model #########################
    tokenizer = create_tokenizer(model_name)
    model = LLM(model_name)
    
    if not do_sample:
        temperature = 0.0 
        
    sampling_params = SamplingParams(
        max_tokens = max_new_tokens,
        temperature = temperature,
        top_p = top_p,
        seed = seed
    )
    
    ########################### Generating outputs #############################    
    base_prompt = confidence_prompts[c_type] if reasoning else confidence_prompts_no_reasoning[c_type]
    
    questions, pred_answers = generate_outputs(
        model,
        tokenizer,
        base_prompt, 
        [d['question'] for d in data],
        batch_size,
        sampling_params,
        template_add)
    
    true_answers = [d['answer'] for d in data]
    
    df = pd.DataFrame({
        "question": questions,
        "true_answer": true_answers,
        "pred_answer": pred_answers
    })
    
    os.makedirs(f"{log_dir}/{c_type}_zs", exist_ok=True)
    df.to_csv(
        os.path.join(
            f"{log_dir}/{c_type}_zs", data_name + '_' + data_type + '.csv'
            ), 
        index=False)
        

if __name__ == "__main__":
    import fire
    fire.Fire(main)