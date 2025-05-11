import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
DEFAULT_PAD_TOKEN = "[PAD]"


def resize_token_embeddings(tokenizer, model):
    extra_token_count = len(tokenizer) - model.get_input_embeddings().weight.data.size(
        0
    )
    if extra_token_count:
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data

        input_embeddings[-extra_token_count:] = input_embeddings[
            :-extra_token_count
        ].mean(dim=0, keepdim=True)

        output_embeddings = model.get_output_embeddings().weight.data

        output_embeddings[-extra_token_count:] = output_embeddings[
            :-extra_token_count
        ].mean(dim=0, keepdim=True)
            
            
def create_tokenizer(
    kind,
    model_dir=None,
    padding_side="left",
    model_max_length=4096,
    **kwargs,
):
    
    tokenizer = AutoTokenizer.from_pretrained(
            model_dir or kind,
            padding_side=padding_side,
            model_max_length=model_max_length,
            use_fast=True,
            legacy=False,
            **kwargs,
        )

    tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})

    return tokenizer


def create_model(
    kind,
    torch_dtype=None,
    model_dir=None,
    use_cache=False,
    tokenizer=None,
    use_int8=False,
    use_int4=False,
    **kwargs,
):
    quantization_config = None
    if use_int8 or use_int4:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=use_int4,
            load_in_8bit=use_int8,
        )
    
    if kind == "mistral":
        kind = "mistralai/Mistral-7B-Instruct-v0.3"
        model = AutoModelForCausalLM.from_pretrained(
            model_dir or f"{kind}",
            torch_dtype=torch_dtype
            or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            quantization_config=quantization_config,
            use_cache=use_cache,
            **kwargs,
        )
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir or kind,
            torch_dtype=torch_dtype
            or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            quantization_config=quantization_config,
            use_cache=use_cache,
            **kwargs,
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    resize_token_embeddings(tokenizer, model)

    if use_int8:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    return model


def create_tokenizer_and_model(kind, tokenizer_args=None, **kwargs):
    tokenizer = create_tokenizer(kind, **(tokenizer_args or dict()))
    model = create_model(kind, tokenizer=tokenizer, **kwargs)
    return tokenizer, model