deepspeed --num_gpus 4 02_train_ds.py --batch_size 64 --epochs 15 --no_use_t_fixup_init --no_torch_dist
