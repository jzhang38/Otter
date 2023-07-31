
accelerate launch --config_file=pipeline/accelerate_configs/accelerate_config_ddp.yaml  pipeline/train/pretraining_cc3m.py \
--external_save_dir outputs \
--run_name Flamingo-Llama2-Chat7B-CC3M_595K_debug \
--cc3m_shards "pipeline/utils/CC3M_595K_{0..7}.tar" \
--pretrained_model_name_or_path luodian/Flamingo-Llama2-Chat7B-CC3M \
--workers 1 \
--dataset_resampled \
--train_num_samples_cc3m 595000 \
--batch_size_cc3m=32 \
--report_to_wandb \
--wandb_project=flamingo-llama2-pretrain \
--lr_scheduler=cosine \
--learning_rate=5e-5 \
--num_epochs=3 \
--warmup_steps_ratio=0.05 \

python /home/peiyuan/peiyuan/Otter/flamingo/injecting_llama2_EVA02_into_flamingo.py --model_choice 7B --save_root_dir ./




export PYTHONPATH=.
accelerate launch --main_process_port 1235 --config_file=pipeline/accelerate_configs/accelerate_config_ddp.yaml  pipeline/train/pretraining_cc3m.py \
--external_save_dir outputs \
--run_name Flamingo-EVA-E-Llama2-Chat7B-CC3M_595K_betas_0.9_0.95 \
--cc3m_shards "pipeline/utils/CC3M_595K_{0..7}.tar" \
--pretrained_model_name_or_path flamingo-eva02-e-llama2-chat-7B-init \
--workers 1 \
--dataset_resampled \
--train_num_samples_cc3m 595000 \
--batch_size_cc3m=16 \
--report_to_wandb \
--wandb_project=flamingo-llama2-pretrain \
--lr_scheduler=cosine \
--learning_rate=5e-5 \
--num_epochs=3 \
--warmup_steps_ratio=0.05 \
--logging_steps=100 \
--gradient_accumulation_steps=1 \
--beta1=0.9 \
--beta2=0.95 

