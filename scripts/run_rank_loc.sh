CUDA_VISIBLE_DEVICES=$1 python train.py --config config/cocostuff/ZLSS.yaml --experimentid rank_loc --imagedataset cocostuff --use_caption
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/rank_loc/checkpoint_20000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/rank_loc/checkpoint_20000.pth.tar -r gzlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/rank_loc/checkpoint_20000.pth.tar -r gzlss --threshold 0.4

CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/rank_loc/checkpoint_20000.pth.tar -r gzlss --threshold 0.4
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/rank_loc/checkpoint_20000.pth.tar -r gzlss --threshold 0.4
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/rank_loc/checkpoint_20000.pth.tar -r gzlss --threshold 0.4

