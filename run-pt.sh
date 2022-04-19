CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 1235 --nproc_per_node=4 02_train_ds.py --batch_size 64 --epochs 15 --no_use_t_fixup_init
