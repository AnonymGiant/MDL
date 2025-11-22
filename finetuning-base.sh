OMP_NUM_THREADS=64 \
torchrun --nproc_per_node=8 \
    --master_port=29502 \
    main_finetune.py \
    --batch_size 1024 \
    --data_path [ImageNet] \
    --model lct_vit_base_patch16 \
    --epochs 100 \
    --blr 1.5e-4 --layer_decay 0.65 \
    --finetune [Pre-trained-CKP] \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --output_dir ./output_dir_base \
    --log_dir ./output_dir_base \
    --num_workers 16
