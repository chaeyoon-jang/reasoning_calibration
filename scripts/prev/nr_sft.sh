#python -m experiments.train.base_sft --c_type "system" --seed 0 --batch_size 2 --max_steps 2500
#python -m experiments.train.base_sft --c_type "continuous" --seed 0 --batch_size 2 --max_steps 2500

#python -m experiments.train.base_sft --c_type "system" --seed 1 --batch_size 2 --max_steps 2500
#python -m experiments.train.base_sft --c_type "continuous" --seed 1 --batch_size 2 --max_steps 2500

#python -m experiments.train.base_sft --c_type "system" --seed 2 --batch_size 2 --max_steps 2500
#python -m experiments.train.base_sft --c_type "continuous" --seed 2 --batch_size 2 --max_steps 2500

#python -m experiments.train.base_sft --c_type "system" --seed 0 --batch_size 2 --max_steps 2500 #--model_name "meta-llama/Llama-3.2-1B-Instruct"
python -m experiments.train.base_sft --c_type "continuous" --seed 0 --batch_size 2 --max_steps 2500 #--model_name "meta-llama/Llama-3.2-1B-Instruct"

#python -m experiments.train.base_sft --c_type "system" --seed 1 --batch_size 2 --max_steps 2500 #--model_name "meta-llama/Llama-3.2-1B-Instruct"
python -m experiments.train.base_sft --c_type "continuous" --seed 1 --batch_size 2 --max_steps 2500 #--model_name "meta-llama/Llama-3.2-1B-Instruct"

#python -m experiments.train.base_sft --c_type "system" --seed 2 --batch_size 2 --max_steps 2500 #--model_name "meta-llama/Llama-3.2-1B-Instruct"
python -m experiments.train.base_sft --c_type "continuous" --seed 2 --batch_size 2 --max_steps 2500 #--model_name "meta-llama/Llama-3.2-1B-Instruct"