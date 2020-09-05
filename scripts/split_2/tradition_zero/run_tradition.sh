
#Zero Label
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/cocostuff/ZLSS.yaml --experimentid split2_tradition --imagedataset cocostuff --use_caption --my_load_from /scratch/tiangy/SPNet/logs/cocostuff/split2_exact_rank/checkpoint_10000.pth.tar
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_1000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_2000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_3000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_4000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_5000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_6000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_7000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_8000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_9000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_tradition/checkpoint_10000.pth.tar -r zlss

#CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_exact_rank/checkpoint_20000.pth.tar -r gzlss
#CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_exact_rank/checkpoint_20000.pth.tar -r gzlss --threshold 0.4


:<<!
#Few Label
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/cocostuff/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 1 --imagedataset cocostuff --inputmix both
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/cocostuff/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 2 --imagedataset cocostuff --inputmix both
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/cocostuff/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 5 --imagedataset cocostuff --inputmix both
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/cocostuff/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 10 --imagedataset cocostuff --inputmix both
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/cocostuff/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 20 --imagedataset cocostuff --inputmix both
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_5b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_5b_0_2000.pth.tar -r gflss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_10b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_10b_0_2000.pth.tar -r gflss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_1b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_1b_0_2000.pth.tar -r gflss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_2b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_2b_0_2000.pth.tar -r gflss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_20b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/FLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/myexp/checkpoint_20000_20b_0_2000.pth.tar -r gflss

!
