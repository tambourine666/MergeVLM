CUDA_VISIBLE_DEVICES=6,7 \
python inference_vlms_reasoning.py \
--dataset_name MMMU \
--finetuned_model_name table-llava-v1.5-7b \
--merge_code \
--tensor_parallel_size 4 \
--weight_mask_rate 0.833 \
--scaling_coefficient 0.5 \
--merge_mode online \
--merge_method svd \

