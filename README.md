# Segment Vasculature 5th place solution

Gold solution for the [Hacking the Human Vasculature](https://www.kaggle.com/competitions/blood-vessel-segmentation/overview) in 3D competition.

The trained weights could be found in `weights` dir.

The proposed solution could be trained in about a week on a single RTX A6000 Ada.


1. Install packages
```
pip install --upgrade pip
pip install -r requirements.dev.txt
pip install -r requirements.txt
```

2. Download [Kaggle data](https://www.kaggle.com/competitions/blood-vessel-segmentation/data) and place it into `data/kaggle/`.

3. Download external [kidney data](https://human-organ-atlas.esrf.eu/datasets/572253707) + [spleen data](https://human-organ-atlas.esrf.eu/datasets/572244401) in 50um resolution, and place them into `data/external`

4. Preprocess Kaggle + external data.
```
python segment_vasculature/preprocessing/create_3d_tensors.py
```

5. Run MLFlow server
```
bash bash_scripts/run_mlflow_server.sh
```

6. Export the project dir to enable relative imports
```
export PYTHONPATH="${PYTHONPATH}:${ABSOLUTE_PROJECT_PATH}"
```

7. Export number of GPUs that are going to be used for training
```
export N_GPUS=1
```

8. Train effnet_v2_m model for kidney1
```
bash bash_scripts/train_effnet_v2_m_kidney1.sh
```
9. Calculate pseudo labels for kidney_2
```
bash bash_scripts/pseudo_label_kidney2.sh
```
Note: don't forget to insert correct path to weights in `configs/callbacks/test.yaml`

10. Train effnet_v2_m model for kidney1 + kidney2
```
bash bash_scripts/train_effnet_v2_m_kidney1_2.sh
```
11. Calculate pseudo labels for kidney_external
```
bash bash_scripts/pseudo_label_kidney_external.sh
```
Note: don't forget to insert correct path to weights in `configs/callbacks/test.yaml`

12. Train effnet_v2_m model for kidney1 + kidney2 + kidney_external
```
bash bash_scripts/train_effnet_v2_m_kidney1_2_external.sh
```
13. Calculate pseudo labels for spleen_external
```
bash bash_scripts/pseudo_label_spleen_external.sh
```
Note: don't forget to insert correct path to weights in `configs/callbacks/test.yaml`

14. Create boundaries masks for all training data
```
python segment_vasculature/preprocessing/create_boundaries.py
```

15. Train effnet_v2_m model for kidney1 + kidney2 + kidney_external + spleen_external
```
bash bash_scripts/train_effnet_v2_m_kidney1_2_external_spleen.sh
```
16. Train dpn model for kidney1 + kidney2 + kidney_external + spleen_external
```
bash bash_scripts/train_dpn_kidney1_2_external_spleen.sh
```
17. Train maxvit model for kidney1 + kidney2 + kidney_external + spleen_external
```
bash bash_scripts/train_maxvit_kidney1_2_external_spleen.sh
```
