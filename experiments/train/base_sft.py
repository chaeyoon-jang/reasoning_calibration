import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd 
from tqdm.auto import tqdm
from datasets import Dataset
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from torch.distributions import Categorical, kl_divergence

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
from src.generate_utils import (
    LabeledStringDataCollator,
    )
            

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

        self._collate_fn = LabeledStringDataCollator(tokenizer, skip_template=True)
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

    def compute_kl_loss(self, model, inputs, targets):
        if self.args.kl_decay <= 0.0:
            return torch.tensor(0.0)

        ref_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(inputs, targets).items()
        }

        probs = model(**ref_inputs).logits[..., :-1, :].softmax(dim=-1)
        with torch.inference_mode():
            self.model.set_adapter(self.args.ref_adapter_name)

            ref_probs = self.model(**ref_inputs).logits[..., :-1, :].softmax(dim=-1)

            self.model.set_adapter("default")
        
        self.model.train()
        labels = ref_inputs.pop("labels")[..., 1:]
        
        p = Categorical(probs=probs)
        p_ref = Categorical(probs=ref_probs)

        if self.args.kl_type == "reverse_kl":
            kl_loss = kl_divergence(p, p_ref)
        elif self.args.kl_type == "forward_kl":
            kl_loss = kl_divergence(p_ref, p)
        elif self.args.kl_type == "jsd":
            p_mix = Categorical(probs=(probs + ref_probs) / 2)
            kl_loss = (kl_divergence(p, p_mix) + kl_divergence(p_ref, p_mix)) / 2
        else:
            raise NotImplementedError

        loss_mask = labels != -100
        loss = (kl_loss * loss_mask).sum(dim=-1).mean(dim=0)

        return loss

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
        
        conf_prompts = inputs['confidence_input_prompt']
        confidence_labels = inputs['confidence_label']
        
        ## target kld loss 
        kl_loss = self.compute_kl_loss(model, answer_prompts, answer_predictions)
        
        ## confidence loss
        conf_loss = self.compute_conf_loss(
            model,
            conf_prompts,
            confidence_labels,
        )
        
        loss_metrics = {
            "conf_loss": conf_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
        }
        
        if return_metrics:
            return loss_metrics

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)
            
        loss = conf_loss + self.args.kl_decay * kl_loss   
        return (loss, None) if return_outputs else loss

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_metrics = {"conf_loss": [], 
                       "kl_loss": []}

        for inputs in tqdm(eval_dataloader, leave=False):
            B = len(inputs.get("confidence_input_prompt"))

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

        self.model.save_pretrained(
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
    lora_rank=128,
    lora_alpha=32,
    lora_dropout=0.1,
    peft_dir=None,
    ref_peft_dir=None,
    batch_size=1,
    lr=1e-4,
    warmup_ratio=0.0,
    kl_decay=1.0,
    max_steps=2000,
    gradient_accumulation_steps=16,
    c_type="base", # random
    reasoning: bool = False,
    kl_type: str = "answer_kl"
):
    
    set_seed(seed)
    
    accelerator = AcceleratorState()

    trainer_args = CalibrationTuner.Args(
        seed=seed,
        output_dir=os.path.join(log_dir,f"{c_type}_{kl_type}"),
        max_steps=max_steps,
        eval_steps=max_steps // 10,
        save_steps=max_steps // 10,
        logging_steps=max(1, max_steps // 200),
        dataloader_num_workers=num_workers,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        kl_decay=kl_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    with accelerator.main_process_first():
        if "random" in c_type:
        #if c_type == "random":
            if "out" in c_type:
                data_dir_1 = os.path.join(data_dir, "base", "answer_kl")
                data_dir_2 = os.path.join(data_dir, "prefix_out", kl_type)    
                         
            else:
                data_dir_1 = os.path.join(data_dir, "base", kl_type)
                data_dir_2 = os.path.join(data_dir, "prefix", kl_type) 
            
        
            train_data_1 = pd.read_csv(os.path.join(f"{data_dir_1}", "train.csv"))
            train_data_1 = train_data_1.sample(frac=1, random_state=seed).reset_index(drop=True)
            train_data_1 = train_data_1.dropna().reset_index(drop=True)[:int(len(train_data_1) * 0.5)]
            
            valid_data_1 = pd.read_csv(os.path.join(f"{data_dir_1}", "valid.csv"))
            valid_data_1 = valid_data_1.dropna().reset_index(drop=True)[int(len(valid_data_1) * 0.5):]
            
            train_data_2 = pd.read_csv(os.path.join(f"{data_dir_2}", "train.csv"))
            train_data_2 = train_data_2.sample(frac=1, random_state=seed).reset_index(drop=True)
            train_data_2 = train_data_2.dropna().reset_index(drop=True)[:int(len(train_data_2) * 0.5)]
            
            valid_data_2 = pd.read_csv(os.path.join(f"{data_dir_2}", "valid.csv"))
            valid_data_2 = valid_data_2.dropna().reset_index(drop=True)[int(len(valid_data_2) * 0.5):]
            
            train_data = pd.concat([train_data_1, train_data_2], axis=0)
            train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
            valid_data = pd.concat([valid_data_1, valid_data_2], axis=0).reset_index(drop=True)
            
            train_data = Dataset.from_pandas(train_data)
            valid_data = Dataset.from_pandas(valid_data)
        
        else:
            data_dir = os.path.join(data_dir, c_type if reasoning else f"{c_type}", kl_type) 
            
            train_data = pd.read_csv(os.path.join(f"{data_dir}", "train.csv"))
            train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
            train_data = train_data.dropna().reset_index(drop=True)
            valid_data = pd.read_csv(os.path.join(f"{data_dir}", "valid.csv"))
            valid_data = valid_data.dropna().reset_index(drop=True)
            
            train_data = Dataset.from_pandas(train_data)
            valid_data = Dataset.from_pandas(valid_data)
        
    tokenizer = create_tokenizer(model_name)
    
    model = create_model(
        model_name,
        tokenizer=tokenizer,
        use_int8=int8,
        device_map={"": accelerator.local_process_index}
        )
    
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
    
    print(f"Training model with calibration tuning ({c_type}).")
    
    trainer = CalibrationTuner(
        model=model,
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

if __name__ == "__main__":
    import fire
    # ===
    import os 
    os.environ['WANDB_PROJECT'] = 'neurips_calibration'
    # === 
    fire.Fire(main)