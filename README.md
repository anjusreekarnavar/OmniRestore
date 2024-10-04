# multi_distortion-based_image_restoration
An image restoration model for handling random multiple distortions
For creating the decoder weights run the code:
python -m torch.distributed.launch main_pretrain.py
Add nproc_per_node =num_gpus, if need to run across multiple gpus 
For running the multidistortion model run 
python -m torch.distributed.launch moe_trainddp.py
