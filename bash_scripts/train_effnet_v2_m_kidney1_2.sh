#!/usr/bin/env

torchrun --nproc_per_node=$N_GPUS segment_vasculature/train.py model=unet_2d_effnet_m_upscale loss=ce_dice_focal dataset=[train/kidneys_1_2,val/kidney_3] transform=medium_float logging.run_name=effnet_m_kidney_1_2 trainer.trainer_hyps.num_epochs=40 trainer.trainer_hyps.debug_iters=3000
