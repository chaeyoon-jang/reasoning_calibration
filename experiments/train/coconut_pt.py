import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd 
from tqdm.auto import tqdm
from dataclasses import dataclass, field

import torch
from torch.utils.data import default_collate

from transformers import set_seed

from transformers.trainer import (
    TRAINING_ARGS_NAME,
    logger,
    Trainer,
    TrainingArguments,
)

from src.logging import (
    entrypoint,
    WandbConfigUpdateCallback
)
from src.distributed import AcceleratorState
from src.llm_model_utils import (
    create_model,
    create_tokenizer
    )
from src.peft_utils import get_lora_model
from src.llm_data_utils import (
    LabeledStringDataCollatorCoconut,
    )
from src.coconut import Coconut
from datasets import Dataset    

class CalibrationTuner(Trainer):
    @dataclass
    class Args(TrainingArguments):
        fp16: bool = field(default=not torch.cuda.is_bf16_supported())
        bf16: bool = field(default=torch.cuda.is_bf16_supported())
        ddp_find_unused_parameters: bool = field(default=False)
        log_on_each_node: bool = field(default=False)
        eval_strategy: str = field(default="steps")
        dataloader_num_workers: int = field(default=4)
        optim: str = field(default="adamw_torch")
        lr: float = field(default=1e-4)
        lr_scheduler_type: str = field(default="cosine")
        weight_decay: float = field(default=0.0)
        warmup_ratio: float = field(default=0.0)
        gradient_accumulation_steps: int = field(default=1)
        report_to: str = field(default="wandb")
        ## Custom args.
        ref_adapter_name: str = field(default="_ref")
        kl_type: str = field(default="jsd")
        kl_decay: float = field(default=0.0)
        
    def __init__(
        self,
        args=None,
        train_dataset=None,
        tokenizer=None,
        c_type=None,
        **kwargs,
    ):
        args.label_names = train_dataset.column_names

        self._collate_fn = LabeledStringDataCollatorCoconut(tokenizer, skip_template=True)
        self.c_type = c_type
        
        super().__init__(
            **kwargs,
            args=args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            data_collator=default_collate,
        )

    def _wrap_model(self, *args, **kwargs):
        return super()._wrap_model(*args, **kwargs)

    def compute_conf_loss(self, model, inputs, conf_targets):
        
        conf_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(inputs, conf_targets).items()
        }
        
        conf_outputs = model(**conf_inputs) 
        return conf_outputs.loss

    def compute_loss(self, 
                     model, 
                     inputs, 
                     return_outputs=False, 
                     return_metrics=False,
                     num_items_in_batch=None):
        
        ## <think> ... </think> -> mask
        ## <answer> ... </answer> -> target kld loss
        ## <confidence> ... </confidence> -> target confidence loss
        
        answer_prompts = inputs['input_prompt']
        answer_predictions = inputs['answer_label']
        
        ## confidence loss
        conf_loss = self.compute_conf_loss(
            model,
            answer_prompts,
            answer_predictions,
        )
        
        loss_metrics = {
            "conf_loss": conf_loss.detach().item(),
        }
        
        if return_metrics:
            return loss_metrics

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)
            
        loss = conf_loss  
        return (loss, None) if return_outputs else loss

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_metrics = {"conf_loss": []}

        for inputs in tqdm(eval_dataloader, leave=False):
            B = len(inputs.get("input_prompt"))

            with torch.inference_mode():
                loss_metrics = self.compute_loss(
                    self.model_wrapped, inputs, return_metrics=True
                )

            loss_metrics = {
                k: torch.zeros(B)
                .index_fill_(0, torch.tensor([0]).long(), v * B)
                .to(self.accelerator.device)
                for k, v in loss_metrics.items()
            }

            [
                all_metrics[l].append(v)
                for l, v in zip(
                    all_metrics.keys(),
                    self.accelerator.gather_for_metrics(
                        tuple(loss_metrics[k] for k in all_metrics.keys())
                    ),
                )
            ]

        all_metrics = {k: torch.cat(v, dim=0) for k, v in all_metrics.items()}
        N = all_metrics["conf_loss"].size(0)

        all_metrics = {
            f"{metric_key_prefix}_{k}": (v[v.nonzero().squeeze(-1)].sum() / N).item()
            for k, v in all_metrics.items()
        }
        all_metrics[f"{metric_key_prefix}_N"] = N

        self.log(all_metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, all_metrics
        )

        return all_metrics

    def _save(self, output_dir=None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()
            state_dict.update(
                {".".join(k.split(".")[2:]): v for k, v in state_dict.items()}
            )

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.base_causallm.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            selected_adapters=["default"],
            save_embedding_layers=False,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


@entrypoint
def main(
    seed=0,
    log_dir=None,
    dataset=None,
    data_dir="data/processed",
    max_token_length=None,
    num_workers=4,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    int8=False,
    lora_rank=64, #128,
    lora_alpha=32,
    lora_dropout=0.1,
    peft_dir=None,
    ref_peft_dir=None,
    batch_size=2,
    lr=1e-4,
    warmup_ratio=0.0,
    kl_decay=1.0,
    max_steps=1000,
    gradient_accumulation_steps=8,
    c_type="base", # random
):
    
    set_seed(seed)
    
    accelerator = AcceleratorState()
    
    ## Load data
    with accelerator.main_process_first():
        all_train_data, all_valid_data = [], []
        stages = [1,2,3,4,5,6,7,8,9,95]
        for i in stages:
            #all_train_data.append(pd.read_csv(f"./data/prev/prev/coconut/stage_{i}_train.csv"))
            #all_valid_data.append(pd.read_csv(f"./data/prev/prev/coconut/stage_{i}_valid.csv"))
            temp_train = pd.read_csv(f"./data/prev/prev/coconut/stage_{i}_train.csv")
            temp_valid = pd.read_csv(f"./data/prev/prev/coconut/stage_{i}_valid.csv")
            train_data = Dataset.from_pandas(temp_train)
            valid_data = Dataset.from_pandas(temp_valid)
            all_train_data.append(train_data)
            all_valid_data.append(valid_data)
    
    ## Load model
    tokenizer = create_tokenizer(model_name)
    
    model = create_model(
        model_name,
        tokenizer=tokenizer,
        use_int8=int8,
        device_map={"": accelerator.local_process_index}
        )
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    
    for token_id in [latent_id, start_id, end_id]:
        target_embedding = embeddings.weight.data[target_id] 
        embeddings.weight.data[token_id] = target_embedding
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id]
    
    model = get_lora_model(
        model,
        peft_id_or_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=True,
        adapter_name="default",
    )

    model = get_lora_model(
        model,
        peft_id_or_dir=ref_peft_dir or peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=False,
        adapter_name="_ref",
    )
    
    model.set_adapter("default")
    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
        
    print(f"Training model with calibration tuning ({c_type}).")
    idx = 0
    for train_data, valid_data, stage in zip(all_train_data, all_valid_data, stages):
        
        max_steps = 500 * (idx+1) 
        
        print(f"Training stage {stage}...")

        trainer_args = CalibrationTuner.Args(
            seed=seed,
            output_dir=os.path.join(log_dir,f"stage_{stage}"),
            max_steps=max_steps,
            eval_steps=max_steps // 5,
            save_steps=max_steps // 5,
            logging_steps=max(1, max_steps // 200),
            dataloader_num_workers=num_workers,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            kl_decay=kl_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    
        trainer = CalibrationTuner(
            model=coconut_model,
            c_type=c_type,
            args=trainer_args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            tokenizer=tokenizer,
            callbacks=[
                WandbConfigUpdateCallback(
                    dataset=dataset,
                    max_token_length=max_token_length,
                    model_name=model_name,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    peft_dir=peft_dir,
                ),
            ],
        )
        trainer.train()
        
        idx += 1

if __name__ == "__main__":
    import fire
    # ===
    import os 
    os.environ['WANDB_PROJECT'] = 'reasoning_calibration'
    # === 
    fire.Fire(main)