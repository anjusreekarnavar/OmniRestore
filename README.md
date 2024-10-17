

Train the baseline model


python -m torch.distributed.launch pretrain_decoder.py --train_data_path path --val_data_path path --log_dir path   --output_dir path  --decoder_depth value


For creating the pretrained weights run

python -m torch.distributed.launch  main_pretrain.py

for using multigpus include --nproc_per_node=num_gpus


For running aggregator run

python -m torch.distributed.launch moe_trainddp.py
