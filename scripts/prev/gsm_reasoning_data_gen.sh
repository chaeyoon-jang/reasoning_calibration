# for test
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 0 --do_sample False --c_type "prefix_out" --data_name "gsm" --data_type "test" --reasoning True

# for data gen
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 0 --do_sample False --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 0 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 1 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 2 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 3 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 4 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 5 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 6 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 7 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 8 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 9 --do_sample True --c_type "prefix_out" --data_name "gsm" --reasoning True

# for test
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 0 --do_sample False --c_type "base" --data_name "math" --data_type "test" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 0 --do_sample False --c_type "infix" --data_name "math" --data_type "test"

# for data gen
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 0 --do_sample False --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 0 --do_sample True --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 1 --do_sample True --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 2 --do_sample True --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 3 --do_sample True --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 4 --do_sample True --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 5 --do_sample True --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 6 --do_sample True --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 7 --do_sample True --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 8 --do_sample True --c_type "base" --data_name "math" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 9 --do_sample True --c_type "base" --data_name "math" --reasoning True