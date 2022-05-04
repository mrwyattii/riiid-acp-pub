#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 1235 --nproc_per_node=4 02_train_ds.py --batch_size 56 --epochs 15
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 1235 --nproc_per_node=1 02_train_ds.py --batch_size 48 --epochs 15
