import transformers
from dataclasses import dataclass

from typing import List, Tuple
from torch import LongTensor

import torch
from torch.utils.data import DataLoader, random_split

IGNORE_LABEL = -100

def get_token_vec(tokenizer, c_type="number"):
    vocab = tokenizer.get_vocab()

    def _create_vec(raw_list):
        for t in raw_list:
            assert t in vocab, f"Cannot handle {t} as a single token."

        return torch.tensor([tokenizer(t).input_ids[-1] for t in raw_list])

    if c_type == "ct":
        raw_strings = ["i", "ii"]
        
    elif c_type == "number":
        raw_strings=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    
    elif c_type == "ling":
        raw_strings = ["Unlikely", "Doubtful", "Uncertain", "Ambiguous", 
                       "Probable", "Likely", "Possible", "Specified",
                       "Confirmed", "Certain", "Inevitable"]
    else:
        raise NotImplementedError
    return _create_vec(raw_strings)


@dataclass
class LabeledStringDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    base_prompt: str = "You are a helpful assistant."
    skip_template: bool = False

    @staticmethod
    def get_tokenizer_args(tokenizer):
        return dict(
            padding=True,
            truncation=True,
            max_length=(
                tokenizer.model_max_length
                if hasattr(tokenizer, "model_max_length")
                else None
            ),
            return_tensors="pt",
            return_length=True,
        )

    def __call__(self, prompts, targets=None):
        tokenizer_args = self.get_tokenizer_args(self.tokenizer)
        
        if (
            self.tokenizer.name_or_path
            and ("Llama-3" in self.tokenizer.name_or_path)
            and ("Instruct" in self.tokenizer.name_or_path)
            and not self.skip_template
        ):
            msgs = [
                [
                    {"role": "system", "content": self.base_prompt},
                    {"role": "user", "content": p},
                ]
                for p in prompts
            ]

            prompts = [
                self.tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in msgs
            ]
        
        if targets:
            all_prompts = [p + t for p, t in zip(prompts, targets)]
        else:
            all_prompts = prompts
        
        inputs = self.tokenizer(all_prompts, **tokenizer_args)
        input_lengths = inputs.pop("length")

        if targets:
            un_inputs = self.tokenizer(prompts, **tokenizer_args)
            un_input_lengths = un_inputs.pop("length")

            labels = inputs.get("input_ids").clone()
            for i, l in enumerate(input_lengths - un_input_lengths):
                labels[i, :-l] = IGNORE_LABEL
            inputs["labels"] = labels
        return inputs


@dataclass
class LabeledStringDataCollatorCoconut:
    """Collate strings (optionally with targets) into a batch aligned on the first ``<|latent|>`` token."""

    tokenizer: transformers.PreTrainedTokenizer
    base_prompt: str = "You are a helpful assistant."
    skip_template: bool = False
    ignore_label: int = IGNORE_LABEL

    # ---------------------------------------------------------------------
    # Helper: build tokenizer kwargs but *do not* pad – we handle padding.
    # ---------------------------------------------------------------------
    @staticmethod
    def get_tokenizer_args(tokenizer: transformers.PreTrainedTokenizer) -> dict:
        return dict(
            padding=False,  # manual padding for <|latent|> alignment
            truncation=True,
            max_length=(tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else None),
            return_tensors="pt",  # we convert tensors -> list right after
        )

    # ------------------------------------------------------------------
    # Helper: align every sequence so that its first <|latent|> token sits
    #         at the same column index. Returns (input_ids, attention_mask).
    # ------------------------------------------------------------------
    def _align_to_latent(
        self,
        sequences: List[List[int]],
        pad_id: int,
        latent_id: int,
    ) -> Tuple[LongTensor, LongTensor]:
        """Left‑pad so the first <|latent|> aligns, then right‑pad to max length."""
        prefix_lens = []
        for ids in sequences:
            try:
                prefix_lens.append(ids.index(latent_id))
            except ValueError:
                prefix_lens.append(len(ids))  # no <|latent|> found

        max_prefix = max(prefix_lens)

        left_padded, attn = [], []
        for ids, pre_len in zip(sequences, prefix_lens):
            left_pad = max_prefix - pre_len
            left_padded.append([pad_id] * left_pad + ids)
            attn.append([0] * left_pad + [1] * len(ids))

        max_len = max(len(ids) for ids in left_padded)
        for i in range(len(left_padded)):
            right_pad = max_len - len(left_padded[i])
            left_padded[i].extend([pad_id] * right_pad)
            attn[i].extend([0] * right_pad)

        return torch.tensor(left_padded, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def __call__(self, prompts: List[str], targets: List[str]):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set (e.g. tokenizer.pad_token_id = tokenizer.eos_token_id)")

        latent_token = "<|latent|>"
        latent_id = self.tokenizer.convert_tokens_to_ids(latent_token)
        if latent_id == self.tokenizer.unk_token_id:
            raise ValueError(f"'{latent_token}' token must exist in the tokenizer vocabulary.")

        # 1) optional chat template
        if (
            self.tokenizer.name_or_path
            and "Llama-3" in self.tokenizer.name_or_path
            and "Instruct" in self.tokenizer.name_or_path
            and not self.skip_template
        ):
            messages = [
                [
                    {"role": "system", "content": self.base_prompt},
                    {"role": "user", "content": p},
                ]
                for p in prompts
            ]
            prompts = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages
            ]

        # 2) concat prompts & targets if provided
        if targets is not None:
            all_texts = [p + t for p, t in zip(prompts, targets)]
        else:
            all_texts = prompts

        # 3) tokenise without padding (tensor -> list[int])
        tok_args = self.get_tokenizer_args(self.tokenizer)
        tokenised_batch = [
            self.tokenizer(t, **tok_args)["input_ids"].squeeze(0).tolist()
            for t in all_texts
        ]

        # 4) align on <|latent|>
        input_ids, attention_mask = self._align_to_latent(tokenised_batch, pad_id, latent_id)

        # 5) position_ids = cumsum(attention_mask) - 1, with pads = 0
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        # 6) labels with prompt‑portion ignored
        if targets is not None:
            prompt_only_tok = [
                self.tokenizer(p, **tok_args)["input_ids"].squeeze(0).tolist()
                for p in prompts
            ]
            prompt_ids, _ = self._align_to_latent(prompt_only_tok, pad_id, latent_id)
            labels = input_ids.clone()
            for i in range(len(labels)):
                prompt_len = (prompt_ids[i] != pad_id).sum()
                labels[i, : prompt_len] = self.ignore_label
            batch["labels"] = labels

        return batch


def train_test_split(dataset, test_size=0.2, seed=None):
    N = len(dataset)
    N_test = int(test_size * N)
    N -= N_test

    if seed is not None:
        train, test = random_split(
            dataset, [N, N_test], generator=torch.Generator().manual_seed(seed)
        )
    else:
        train, test = random_split(dataset, [N, N_test])

    return train, test


def get_num_workers(num_workers=4):
    num_gpus_per_host = torch.cuda.device_count()
    if num_gpus_per_host == 0:
        return num_workers
    return (num_workers + num_gpus_per_host - 1) // num_gpus_per_host


def get_loader(dataset, batch_size=128, num_workers=4, accelerator=None, **kwargs):
    num_workers = get_num_workers(num_workers=num_workers)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
    )
    if accelerator is not None:
        loader = accelerator.prepare(loader)

    return loader
