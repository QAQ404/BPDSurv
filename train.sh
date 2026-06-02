CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /your_data_root_dir \
--split_dir tcga_blca \
--model_type BPDSurv \
--which_splits 5foldcv