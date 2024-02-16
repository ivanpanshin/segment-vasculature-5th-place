#!/usr/bin/env

torchrun --nproc_per_node=$N_GPUS segment_vasculature/train.py model=unet_2d_effnet_m_upscale loss=ce_dice_focal dataset=[train/kidney_1,val/kidney_3] transform=medium logging.run_name=effnet_m_kidney_1 trainer.trainer_hyps.num_epochs=40
