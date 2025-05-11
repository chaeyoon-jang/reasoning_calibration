# zero-shot evaluation
#python -m experiments.make_lm_outputs --data_name 'gsm' --c_type 'base' --batch_size 32 --reasoning True --seed 0 --do_sample False --suffix True
#python -m experiments.make_lm_outputs --data_name 'gsm' --c_type 'prefix' --batch_size 32 --reasoning True --seed 0 --do_sample False 
#python -m experiments.make_lm_outputs --data_name 'gsm' --c_type 'prefix_out' --batch_size 32 --reasoning True --seed 0 --do_sample False

# sft evaluation
python -m experiments.make_lm_outputs --data_name 'gsm' --c_type 'base' --query_peft_dir "/mnt/home/chaeyun-jang/reasoning_calibration/logs/2025-05-10T22-32-36/base_answer_kl" --batch_size 32 --reasoning True --seed 0 --do_sample False --suffix True
python -m experiments.make_lm_outputs --data_name 'gsm' --c_type 'base' --query_peft_dir "/mnt/home/chaeyun-jang/reasoning_calibration/logs/2025-05-11T02-02-31/base_cot_kl" --batch_size 32 --reasoning True --seed 0 --do_sample False --suffix True
python -m experiments.make_lm_outputs --data_name 'gsm' --c_type 'prefix' --query_peft_dir "/mnt/home/chaeyun-jang/reasoning_calibration/logs/2025-05-11T05-32-39/prefix_answer_kl" --batch_size 32 --reasoning True --seed 0 --do_sample False
python -m experiments.make_lm_outputs --data_name 'gsm' --c_type 'prefix' --query_peft_dir "/mnt/home/chaeyun-jang/reasoning_calibration/logs/2025-05-11T09-12-11/prefix_cot_kl" --batch_size 32 --reasoning True --seed 0 --do_sample False
#python -m experiments.make_lm_outputs --data_name 'gsm' --c_type 'prefix_out' --query_peft_dir "/mnt/home/chaeyun-jang/reasoning_calibration/logs/2025-05-11T12-51-46/prefix_out_cot_kl" --batch_size 32 --reasoning True --seed 0 --do_sample False

