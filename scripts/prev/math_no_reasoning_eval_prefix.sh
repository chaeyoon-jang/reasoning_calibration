python -m experiments.make_lm_outputs_vllm --batch_size 16 --seed 0 --do_sample False --c_type "prefix" --data_name "math" --data_type "test" --reasoning False
python -m experiments.make_lm_outputs --query_peft_dir '/mnt/home/chaeyun-jang/reasoning_calibration/logs/2025-04-21T10-35-48' --data_name 'math' --c_type 'prefix' --batch_size 32 --reasoning False --seed 0 --do_sample False
