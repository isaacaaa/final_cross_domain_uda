python3 x_run_bert_base.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir sorted_data_acl/ \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 256 \
  --output_dir output/ \
  --save_steps -1 \
  --overwrite_output_dir