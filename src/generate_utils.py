import gc
import torch
from tqdm.auto import tqdm
from peft import PeftModel
import torch.nn.functional as F
from .llm_data_utils import LabeledStringDataCollator, get_token_vec

def wrapped_generate_output(
    model,
    tokenizer,
    generation_inputs,
    generation_config):
    
    while True:
        try:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            generation_outputs = model.generate(
                **generation_inputs, 
                eos_token_id=terminators,
                generation_config=generation_config
            )
            return generation_outputs
        except Exception as e:
            generation_outputs = []
            new_bs = max(1, generation_inputs["input_ids"].size(0) // 2)
            for i in range(0, generation_inputs["input_ids"].size(0), new_bs):
                inputs = {k: v[i : i + new_bs] for k, v in generation_inputs.items()}
                _outputs = wrapped_generate_output(model, inputs, generation_config)
                generation_outputs.append(_outputs)
            return torch.cat(generation_outputs, dim=0)


def generate_outputs(
    accelerator,
    model,
    tokenizer,
    loader,
    generation_config,
    input_col_name="prompt",
    skip_template=False):
    
    collate_fn = LabeledStringDataCollator(tokenizer, skip_template=skip_template)
    
    results = []
    for inputs in tqdm(loader):
        inputs = inputs[input_col_name]
        
        generation_inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()
            }

        if isinstance(model, PeftModel):
            model.set_adapter("query")
            
        generation_outputs = wrapped_generate_output(model,
                                          tokenizer,
                                          generation_inputs,
                                          generation_config)
        
        generations = tokenizer.batch_decode(
            generation_outputs[:, generation_inputs.get("input_ids").size(-1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            )
        
        results.extend(generations)
        del generation_outputs
        gc.collect()
        torch.cuda.empty_cache()
    
    return results