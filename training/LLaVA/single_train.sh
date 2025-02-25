export WANDB_NAME='llava-1.6-mistral-7b-r128-a256-lr1e4-without-sg-no-eval-sampling'
export WANDB_USERNAME='aisg-meme'
export WANDB_PROJECT='llava-new'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 llava/train/train_mem.py \
--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
--deepspeed ./scripts/zero2.json \
--model_name_or_path liuhaotian/llava-v1.6-mistral-7b \
--version v1 \
--data_path ../training/train_without_sg_dedup.json \
--image_folder ../dataset \
--vision_tower openai/clip-vit-large-patch14-336 \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio anyres \
--group_by_modality_length True \
--bf16 True \
--output_dir "./checkpoints/$WANDB_NAME" \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 16 \
--eval_strategy "steps" \
--eval_steps 100 \
--eval_accumulation_steps 1 \
--eval_with_sampling False \
--save_strategy "steps" \
--save_steps 50 \
--data_seed 42 \
--learning_rate 1e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 6144 \
--gradient_checkpointing True \
--dataloader_num_workers 256 \
--lazy_preprocess True \
--eval_before_training False \
--report_to wandb