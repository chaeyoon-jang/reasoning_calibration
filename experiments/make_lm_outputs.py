import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import wandb
import datasets
import pandas as pd
from datasets import Dataset
from transformers import GenerationConfig, set_seed

from src.logging import entrypoint
from src.generate_utils import (
    generate_outputs
    )
from src.llm_model_utils import (
    create_model,
    create_tokenizer
    )   
from src.peft_utils import get_lora_model
from src.llm_data_utils import get_loader 
from src.prompt_hub import confidence_prompts, confidence_prompts_no_reasoning



def make_outputs(
    accelerator,
    model,
    tokenizer,
    base_prompt,
    dataset,
    batch_size,
    generation_config,
    suffix=False):
    
    
    dataset = dataset.map(
        lambda x: {"question": 
            base_prompt.replace("<question>", x["question"])
            })
    
    loader = get_loader(dataset, batch_size=batch_size,
                pin_memory=True, accelerator=accelerator)
    
    outputs = generate_outputs(
        accelerator,
        model,
        tokenizer,
        loader,
        generation_config,
        "question")
    
    print(outputs[0])
    
    df = pd.DataFrame({
        "question": [d["question"] for d in dataset], 
        "pred_answer": outputs, 
        "true_answer": [d["answer"] for d in dataset]
    })
    
    dataset = dataset.add_column("pred_answer", outputs)
    
    if suffix:
        
        import torch; torch.cuda.empty_cache()
        import gc; gc.collect()
        
        conf_dataset = dataset.map(
            lambda x: {"conf_input":
                tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}
                                               ,{"role": "user", "content": x["question"]}],
                                            tokenize=False,
                                            add_generation_prompt=True
                                            ) + x["pred_answer"] + "\nPlease respond with a score from 0 to 100 in `<confidence> </confidence>` tags. How confident are you in your previous answer?" #"<confidence>"
            }
        )

        conf_loader = get_loader(conf_dataset, batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        conf_outputs = generate_outputs(
            accelerator,
            model,
            tokenizer,
            conf_loader,
            generation_config,
            "conf_input",
            skip_template=True
        )   
        print(conf_outputs[0])
        df['conf'] = conf_outputs 
        
    return df 
    

@entrypoint(with_accelerator=True, with_wandb=False)
def main(
    seed: int=0,
    accelerator = None,
    log_dir: str = None,
    data_dir: str = "./data/MATH/test",
    data_name: str = "gsm",
    data_type: str = "test",
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    query_peft_dir: str = None,
    int8: bool = False,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    c_type= "system",  
    reasoning: bool = False,
    suffix: bool = False,
):
    
    set_seed(seed) # for sampling method in ling and number.
        
    ############################# Loading datasets #############################
    if data_name == "gsm":
        data = datasets.load_dataset('openai/gsm8k', 'main')[data_type]

    elif data_name == "math":
        '''
        data_paths = os.listdir(data_dir)
        data = []
        for category in data_paths:
            temp_paths = os.listdir(os.path.join(data_dir, category))
            for p in temp_paths:
                temp_data = json.load(open(os.path.join(data_dir, category, p)))
                data.append({'question': temp_data['problem'], 'answer': temp_data['solution']})
                
        print(len(data))
        print(data[0])
        '''
        data = pd.read_csv(f"./data/syn_test/test.csv")
        data = [{'question': data['question'][i], 'answer': data['answer'][i]} for i in range(len(data))]
        
        data = Dataset.from_list(data)
        
    ######################## Loading tokenizer & model #########################
    tokenizer = create_tokenizer(model_name)
    model = create_model(
        model_name,
        tokenizer=tokenizer,
        use_int8=int8,
        device_map="auto",
    )
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample, 
        temperature=temperature,
        top_p=top_p
    )
    
    ############################# Loading PEFT model ###########################
    ############################# and classifier head ##########################
    if query_peft_dir:
        model = get_lora_model(
            model,
            peft_id_or_dir=query_peft_dir,
            is_trainable=False,
            adapter_name="query",
        )
        
    ########################### Generating outputs #############################
    model.eval()        
    df = make_outputs(
        accelerator,
        model,
        tokenizer,
        confidence_prompts[c_type] if reasoning else confidence_prompts_no_reasoning[c_type],
        data,
        batch_size,
        generation_config,
        suffix
    )
    
    os.makedirs(f"{log_dir}/{c_type}_evaluation", exist_ok=True)
    df.to_csv(
        os.path.join(
            f"{log_dir}/{c_type}_evaluation", data_name + '_' + data_type + '.csv'
            ), 
        index=False)

if __name__ == "__main__":
    import fire
    fire.Fire(main)