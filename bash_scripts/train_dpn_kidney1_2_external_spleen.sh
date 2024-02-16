#!/usr/bin/env

torchrun --nproc_per_node=$N_GPUS segment_vasculature/train.py --config-name=train_boundaries model=unet_2d_dpn68_upscale loss=bound_twersky_focal dataset=[train/kidneys_1_2_external_spleen_boundaries,val/kidney_3] transform=medium_float_boundaries logging.run_name=dpn_kidney_1_2_external_spleen_boundaries_twersky trainer.trainer_hyps.num_epochs=40 trainer.trainer_hyps.debug_iters=3000
