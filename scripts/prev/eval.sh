#python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/system_seed_1_2025-04-02T23-56-08' --batch_size 32 --c_type 'system' --reasoning True

python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/continuous_seed_1_2025-04-03T05-27-55' --c_type 'continuous' --batch_size 32 --reasoning True --seed 0 --do_sample True 
python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/continuous_seed_1_2025-04-03T05-27-55' --c_type 'continuous' --batch_size 32 --reasoning True --seed 1 --do_sample True
python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/continuous_seed_1_2025-04-03T05-27-55' --c_type 'continuous' --batch_size 32 --reasoning True --seed 2 --do_sample True
python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/continuous_seed_1_2025-04-03T05-27-55' --c_type 'continuous' --batch_size 32 --reasoning True --seed 3 --do_sample True
python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/continuous_seed_1_2025-04-03T05-27-55' --c_type 'continuous' --batch_size 32 --reasoning True --seed 4 --do_sample True

#python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/system_seed_2_2025-04-03T10-32-11' --batch_size 32 --c_type 'system' --reasoning True
#python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/continuous_seed_2_2025-04-03T16-03-56' --c_type 'continuous' --batch_size 32 --reasoning True

## col name 다른거 바꾸기 

## nr case
#python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/nr_system_seed_0_2025-04-07T14-34-29' --batch_size 32 --c_type 'system'
#python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/nr_continuous_seed_1_2025-04-07T23-04-26' --c_type 'continuous' --batch_size 32

#python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/nr_system_seed_1_2025-04-07T20-11-44' --batch_size 32 --c_type 'system'
#python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/nr_continuous_seed_1_2025-04-07T23-04-26' --c_type 'continuous' --batch_size 32

#python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/nr_system_seed_2_2025-04-08T01-49-17' --batch_size 32 --c_type 'system'
#python -m experiments.make_lm_outputs --query_peft_dir './logs/complete/models/nr_continuous_seed_2_2025-04-08T04-42-13' --c_type 'continuous' --batch_size 32