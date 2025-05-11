import os
import re
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm

from src.llm_model_utils import create_tokenizer
from src.prompt_hub import PARSING_PROMPT


def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_number_xml_confidence(text: str) -> float:
    try:
        confidence = text.split("<confidence>")[-1]
        confidence = confidence.split("</confidence>")[0]
        import re; cleaned = re.sub(r'[^0-9.]', '', confidence)
        return float(cleaned)
    except:
        return -1


def calculate_ece(new_df):
    rtf = list(new_df[new_df['prob']!=-1]['tf'])
    rprob = [0.01* p for p in list(new_df[new_df['prob']!=-1]['prob'])]
    print(f"Available data #: {len(rtf)}")
    def compute_ece(y_true, y_prob, n_bins=10):
        import numpy as np
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges, right=True) - 1

        ece = 0.0
        n = len(y_true)

        for i in range(n_bins):
            bin_mask = bin_indices == i
            bin_size = np.sum(bin_mask)

            if bin_size > 0:
                avg_confidence = np.mean(y_prob[bin_mask])
                avg_accuracy = np.mean(y_true[bin_mask])
                ece += (bin_size / n) * np.abs(avg_confidence - avg_accuracy)

        return ece

    our_ece = compute_ece(rtf, rprob, n_bins=15)
    print(f"ECE: {our_ece}")
    
    
def batch_generate(
    input_texts, 
    model,
    sampling_params,
    batch_size=64
    ):
    results = []
    for i in tqdm(range(0, len(input_texts), batch_size)):
        
        if i + batch_size > len(input_texts):
            batch_texts = input_texts[i:]
        
        else:
            batch_texts = input_texts[i:i+batch_size]
        
        outputs = model.generate(
            batch_texts,
            sampling_params,
            use_tqdm=False)
        outputs = [outputs[i].outputs[0].text for i in range(len(outputs))]
        if i == 0:
            print(outputs[0])
        results.extend(outputs)
    return results


def process_and_save(
    model,
    sampling_params,
    tokenizer,
    data,
    save_path,
    conf_col='pred_answer',
    parsing_prompt=PARSING_PROMPT
    ):
    
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": parsing_prompt}]
    
    parsing_prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    
    df = data.copy()
    pred_answer_text = [extract_xml_answer(t) for t in list(df['pred_answer'])]
    prompts = [parsing_prompt.format(model_answer=ans) for ans in pred_answer_text]
    
    parsed_results = batch_generate(prompts, 
                                model,
                                sampling_params,
                                batch_size=32)
    
    df['parsed_answer'] = parsed_results
    
    df.to_csv(f'{save_path}_parsed.csv', index=False)
    
    true_answer_text = [re.sub(r'[^0-9.]', '', extract_hash_answer(t)) for t in list(data['true_answer'])]
    parsed_answer_text = [a.split('**Model\'s Final Answer is:**')[-1].split('\n')[0] for a in parsed_results]
    
    new_parsed_answer_text = []
    for a in parsed_answer_text:
        try:
            new_parsed_answer_text.append(re.sub(r'[^0-9.]', '', a))
        except:
            new_parsed_answer_text.append('')

    tf = [1 if str(g) == str(p) else 0 for g, p in zip(true_answer_text, new_parsed_answer_text)]
    
    print(f"Accuracy: {sum(tf) / len(tf)}")
    
    conf = [extract_number_xml_confidence(t) for t in list(data[conf_col])]
    conf_labels = []
    for c in conf:
        try:
            conf_labels.append(float(c))
        except:
            conf_labels.append(-1)
    
    new_df = pd.DataFrame()
    new_df['tf'] = tf 
    new_df['prob'] = conf_labels

    calculate_ece(new_df)
    
            
def main(data_path=None,
         save_path='./logs/evaluation_results',
         data_type="zs_base",
         conf_col='pred_answer',):
    
    data = pd.read_csv(data_path)
    tokenizer = create_tokenizer("meta-llama/Llama-3.2-3B-Instruct")
    model = LLM(model="meta-llama/Llama-3.2-3B-Instruct")

    sampling_params = SamplingParams(
        max_tokens = 20,
        temperature = 0.0,
        top_p = 1.0,
        seed = 0
    )

    process_and_save(model,
                     sampling_params,
                     tokenizer,
                     data, 
                     f"{save_path}/{data_type}",
                     conf_col)