

export PYTHONPATH=.
accelerate launch --config_file=pipeline/accelerate_configs/accelerate_config_ddp.yaml \
pipeline/train/pretraining_cc3m.py \
--run_name=flamingo-llama2-cc3m-clip \
--pretrained_model_name_or_path=checkpoints/flamingo-clip-l-llama2-chat-7B-init \
--dataset_resampled \
--batch_size_cc3m=32 \
--num_epochs=6 \
--report_to_wandb \
--cc3m_shards="data/cc3m/{00000..00331}.tar" \
--train_num_samples_cc3m=3000000 \
--wandb_project=flamingo-llama2-pretrain \
--external_save_dir=/home/luodian/projects/checkpoints \
--checkpointing_steps=10000 \
--save_hf_model \
--workers=8 \
--lr_scheduler=cosine \
--delete_previous_checkpoint \
--learning_rate=1e-4 \
--warmup_steps_ratio=0.005


export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file=pipeline/accelerate_configs/accelerate_config_ddp.yaml --main_process_port 1234 \
pipeline/train/pretraining_cc3m.py \
--run_name=flamingo-llama2-cc3m-eva \
--pretrained_model_name_or_path=checkpoints/flamingo-eva02-e-llama2-chat-7B-init \
--dataset_resampled \
--batch_size_cc3m=32 \
--num_epochs=6 \
--report_to_wandb \
--cc3m_shards="data/cc3m/{00000..00331}.tar" \
--train_num_samples_cc3m=3000000 \
--wandb_project=flamingo-llama2-pretrain \
--external_save_dir=/home/luodian/projects/checkpoints \
--checkpointing_steps=10000 \
--save_hf_model \
--workers=8 \
--lr_scheduler=cosine \
--delete_previous_checkpoint \
--learning_rate=1e-4 \
--warmup_steps_ratio=0.005

