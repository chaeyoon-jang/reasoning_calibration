#python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample True --c_type "continuous" --data_type "test" --reasoning True
#python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample True --c_type "system"     --data_type "test" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample True --c_type "base" --data_type "test" --reasoning True
python -m experiments.make_lm_outputs_vllm --batch_size 32 --seed 0 --do_sample True --c_type "base" --data_type "test" --reasoning False