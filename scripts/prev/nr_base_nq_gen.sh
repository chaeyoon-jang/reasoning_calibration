python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample False --c_type "continuous" --data_type "validation" --data_name "nq" #--model_name "Qwen/Qwen2.5-Math-7B-Instruct"
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample False --c_type "base" --data_type "validation" --data_name "nq"  #--model_name "Qwen/Qwen2.5-Math-7B-Instruct"

python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample False --c_type "continuous" --data_name "nq" 

python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample True --c_type "continuous" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 1 --do_sample True --c_type "continuous" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 2 --do_sample True --c_type "continuous" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 3 --do_sample True --c_type "continuous" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 4 --do_sample True --c_type "continuous" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 5 --do_sample True --c_type "continuous" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 6 --do_sample True --c_type "continuous" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 7 --do_sample True --c_type "continuous" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 8 --do_sample True --c_type "continuous" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 9 --do_sample True --c_type "continuous" --data_name "nq" 

###
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample False --c_type "base" --data_name "nq" 

python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample True --c_type "base" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 1 --do_sample True --c_type "base" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 2 --do_sample True --c_type "base" --data_name "nq"  
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 3 --do_sample True --c_type "base" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 4 --do_sample True --c_type "base" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 5 --do_sample True --c_type "base" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 6 --do_sample True --c_type "base" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 7 --do_sample True --c_type "base" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 8 --do_sample True --c_type "base" --data_name "nq" 
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 9 --do_sample True --c_type "base" --data_name "nq" 