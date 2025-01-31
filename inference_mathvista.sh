CUDA_VISIBLE_DEVICES=6,7 \
python inference_vlms_reasoning.py \
--dataset_name MathVista \
--finetuned_model_name llava-v1.6-vicuna-7b \
--merge_math \
--merge_code \
--merge_table \
--tensor_parallel_size 4 \
--weight_mask_rate 0.832 \
--scaling_coefficient 0.5 \
--merge_mode online \
--merge_method svd \


