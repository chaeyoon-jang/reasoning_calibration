## base
python -m experiments.train.base_sft --c_type "base" --seed 0 --max_steps 2000 --reasoning True --kl_decay 1.0 --kl_type "answer_kl"
python -m experiments.train.base_sft --c_type "base" --seed 0 --max_steps 2000 --reasoning True --kl_decay 1.0 --kl_type "cot_kl"

## prefix
python -m experiments.train.base_sft --c_type "prefix" --seed 0 --max_steps 2000 --reasoning True --kl_decay 1.0 --kl_type "answer_kl"
python -m experiments.train.base_sft --c_type "prefix" --seed 0 --max_steps 2000 --reasoning True --kl_decay 1.0 --kl_type "cot_kl"

## prefix_out
python -m experiments.train.base_sft --c_type "prefix_out" --seed 0 --max_steps 2000 --reasoning True --kl_decay 1.0 --kl_type "cot_kl"

## random
python -m experiments.train.base_sft --c_type "random" --seed 0 --max_steps 2000 --reasoning True --kl_decay 1.0 --kl_type "cot_kl"
python -m experiments.train.base_sft --c_type "random" --seed 0 --max_steps 2000 --reasoning True --kl_decay 1.0 --kl_type "answer_kl"

python -m experiments.train.base_sft --c_type "random_out" --seed 0 --max_steps 2000 --reasoning True --kl_decay 1.0 --kl_type "cot_kl" 

## kl zero
python -m experiments.train.base_sft --c_type "base" --seed 0 --max_steps 2000 --reasoning True --kl_decay 0.0 --kl_type "cot_kl"
python -m experiments.train.base_sft --c_type "prefix" --seed 0 --max_steps 2000 --reasoning True --kl_decay 0.0 --kl_type "cot_kl"
python -m experiments.train.base_sft --c_type "prefix_out" --seed 0 --max_steps 2000 --reasoning True --kl_decay 0.0 --kl_type "cot_kl"
python -m experiments.train.base_sft --c_type "random" --seed 0 --max_steps 2000 --reasoning True --kl_decay 0.0 --kl_type "answer_kl"