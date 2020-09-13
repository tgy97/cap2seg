export CUDA_VISIBLE_DEVICES=7
python run_classifier_append_multi_soft_withpixel.py \
  --task_name coco \
  --do_train \
  --consistency_loss \
  --consist_scale 1 \
  --do_eval \
  --do_eval_pixel \
  --data_dir data/binary_all_multi_soft_withpixel/ \
  --bert_model /scratch/tiangy/.pytorch_pretrained_bert/ \
  --max_seq_length 24 \
  --train_batch_size 128 \
  --eval_batch_size 1024 \
  --learning_rate 5e-5 \
  --num_train_epochs 5.0 \
  --output_dir logs/scale1/\
  --cls_dir model_split/split_2/ \
  --emb_file data/datasets/cocostuff/word_vectors/fastnvec.pkl \
  --loss_type bce \
  --tokenizer /scratch/tiangy/.pytorch_pretrained_bert/ \




