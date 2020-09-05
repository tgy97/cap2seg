#Zero Label
#CUDA_VISIBLE_DEVICES=$1 python train.py --config config/cocostuff/ZLSS.yaml --experimentid split2_thresh_append --imagedataset cocostuff --use_caption --read_score
#CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_thresh_append/checkpoint_current.pth.tar -r zlss
#CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_thresh_append/checkpoint_current.pth.tar -r gzlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_thresh_append/checkpoint_current.pth.tar -r gzlss --threshold 0.4
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_thresh_append/checkpoint_current.pth.tar -r gzlss --threshold 0.6
#CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_thresh_append/checkpoint_9000.pth.tar -r zlss
#CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_thresh_append/checkpoint_6000.pth.tar -r zlss
#CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/cocostuff/ZLSS.yaml --imagedataset cocostuff --model-path logs/cocostuff/split2_thresh_append/checkpoint_12000.pth.tar -r zlss


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