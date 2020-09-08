import pickle
import numpy as np
import cv2

from IPython import embed
import pprint
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")
import argparse
from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('/scratch/tiangy/bert/bert_pretrained_model/add_word/')
def gaussian( x , s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )


def mAP_append_local(file,softmax_scale = None):
    print('softmax scale: ', softmax_scale)
    # num_class = 182
    # cls_score = pickle.load(open(file,'rb')) 
    # all_labels  = np.genfromtxt('/scratch/tiangy/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute


    # id2cls = pickle.load(open('/scratch/tiangy/bert/model_split/id2cls_val.pkl','rb'))

    # seen_cls = np.load('/scratch/tiangy/bert/model_split/seen_cls.npy')
    # val_cls = np.load('/scratch/tiangy/bert/model_split/val_cls.npy')
    # novel_cls = np.load('/scratch/tiangy/bert/model_split/novel_cls.npy')

    # class_all = np.concatenate([seen_cls,val_cls,novel_cls])
    # know_class = np.concatenate([seen_cls,val_cls])
    ##  
    num_class = 182
    cls_score = pickle.load(open(file,'rb')) #relative
    location = cls_score[1]
    cls_score = cls_score[0]
    #cls_score = pickle.load(open('/m/tiangy/embeddings/bert/logs/test_pixel_num_margin/evaluation3/cls_score.pkl','rb')) #relative
    all_labels  = np.genfromtxt('data/datasets/cocostuff/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #absolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    #id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    id2cls_val = pickle.load(open('model_split/id2cls_val.pkl','rb'))#absolute
    loc_val = pickle.load(open('data/binary_all_multi_soft_withpixel/val.pkl','rb'))
    #loc_train = pickle.load(open('data/binary_all_multi_soft_withpixel/train.pkl','rb')) #######todo
    def search_loc(cocoid,mod):
        l = loc_val if mod == 'val' else loc_train
        for i in l:
            if i[0] == cocoid:
                return i[3]

    seen_cls = np.load('model_split/split_2/seen_cls.npy')
    novel_cls = np.load('model_split/split_2/novel_cls.npy')

    novel_hard_cls = np.load('model_split/split_2/novel_hard.npy')
    novel_medium_cls = np.load('model_split/split_2/novel_medium.npy')
    novel_easy_cls = np.load('model_split/split_2/novel_easy.npy')

    class_all = np.concatenate([seen_cls,novel_cls])
    know_class = seen_cls
    ##
    mod = 'val'
    if mod =='val':
        id2cls_used = id2cls_val
    elif mod =='train':
        id2cls_used = id2cls

    all_mean_dif = [np.array([])  for i in range(num_class)]
    for cocoid in id2cls_used:
        gt_loc_dic = search_loc(cocoid,mod)
        for i in range(num_class):
            try:
                loc = location[cocoid+'_'+str(i)]
                if not softmax_scale is None:
                    loc = loc*softmax_scale
                loc = np.exp(loc) / (np.exp(loc).sum())
                gt_loc = gt_loc_dic[i]
                gt_loc = gt_loc / gt_loc.sum()
            except:
                continue
            #all_mean_dif[i] = np.append(all_mean_dif[i], np.abs(loc - gt_loc).mean())
            all_mean_dif[i] = np.append(all_mean_dif[i], (loc*gt_loc).sum())

    all_mean_dif = np.array([i.mean() for i in all_mean_dif])
    id2score = [] # n x 182
    gt = []
    for cocoid in id2cls_used:
    #for cocoid in id2cls_val:
        #base = cls_score[cocoid+'_-1']
        score_now = np.zeros(num_class)
        gt_cls = np.zeros(num_class)
        for i in range(num_class):
            # now = cls_score[cocoid+'_'+str(i)]
            # score_now[i] = now
            try:
                now = cls_score[cocoid+'_'+str(i)]
                score_now[i] = now
            except:
                pass
            #cls_dif[i] = np.dot(now,base)/(np.linalg.norm(now)*np.linalg.norm(base))
        gt_cls[id2cls_used[cocoid]] = 1
        id2score.append(score_now)
        gt.append(gt_cls)
    id2score,gt = np.stack(id2score).transpose(), np.stack(gt).transpose() # 182 x n

    AP_all = np.zeros(num_class)
    random_all = np.zeros(num_class)
    for i in class_all:
        pre = id2score[i]
        g = gt[i]
        AP,random = binary_mAP(pre,g)

        AP_all[i] = AP
        random_all[i] = random
        #print(all_labels[i],' ',AP)
    #np.save('/scratch/tiangy/bert/model_split/split_2/class_rank.npy',np.argsort(AP_all))
    # for i in np.argsort(AP_all):
    #   print(tokenizer.tokenize(all_labels[i]),AP_all[i],random_all[i])
    #   if AP_all[i]<random_all[i]:
    #       print('worse than random :',all_labels[i] )

    print('mAP results:')
    used = [  4,   9,  10,  11,  15,  18,  19,  21,  23,  26,  30,  35,  39,
    41,  42,  48,  51,  55,  58,  63,  72,  75,  77,  81,  82,  87,
    89,  90,  94, 100, 102, 107, 110, 113, 115, 117, 119, 124, 125,
    126, 133, 134, 136, 139, 141, 142, 145, 153, 154, 157, 166, 169,
    170, 172, 173, 174, 175, 176, 177, 178]
    #print("mean used class: ","%.3f"%np.nanmean(AP_all[used]),'   ',"%.3f"%np.nanmean(random_all[used]))
    print("mean seen class: ","%.3f"%np.nanmean(AP_all[know_class]),'   ',"%.3f"%np.nanmean(random_all[know_class]))
    print("mean unseen class: ","%.3f"%np.nanmean(AP_all[novel_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_cls]))
    print(' ')
    print('unseen class:')
    for i in novel_cls:
        print(all_labels[i],' : ',"%.3f"%AP_all[i],'   ',"%.3f"%random_all[i])
    print(' ')
    print("mean hard unseen class: ","%.3f"%np.nanmean(AP_all[novel_hard_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_hard_cls]))
    print("mean medium unseen class: ","%.3f"%np.nanmean(AP_all[novel_medium_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_medium_cls]))
    print("mean easy unseen class: ","%.3f"%np.nanmean(AP_all[novel_easy_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_easy_cls]))
    print(' ')
    print("mean seen class diff: ","%.3f"%np.nanmean(all_mean_dif[know_class]))
    print("mean unseen class diff: ","%.3f"%np.mean(all_mean_dif[novel_cls]))
    print("mean hard unseen class diff: ","%.3f"%np.nanmean(all_mean_dif[novel_hard_cls]))
    print("mean medium unseen class diff: ","%.3f"%np.nanmean(all_mean_dif[novel_medium_cls]))
    print("mean easy unseen class diff: ","%.3f"%np.nanmean(all_mean_dif[novel_easy_cls]))

def mAP_append_local_en(file,file2,scale=0.5,softmax_scale = None):
    print('softmax scale: ', softmax_scale)
    print('fusion scale: ', scale)
    # num_class = 182
    # cls_score = pickle.load(open(file,'rb')) 
    # all_labels  = np.genfromtxt('/scratch/tiangy/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute


    # id2cls = pickle.load(open('/scratch/tiangy/bert/model_split/id2cls_val.pkl','rb'))

    # seen_cls = np.load('/scratch/tiangy/bert/model_split/seen_cls.npy')
    # val_cls = np.load('/scratch/tiangy/bert/model_split/val_cls.npy')
    # novel_cls = np.load('/scratch/tiangy/bert/model_split/novel_cls.npy')

    # class_all = np.concatenate([seen_cls,val_cls,novel_cls])
    # know_class = np.concatenate([seen_cls,val_cls])
    ##  
    num_class = 182
    cls_score = pickle.load(open(file,'rb')) #relative
    location = cls_score[1]
    cls_score = cls_score[0]

    cls_score2 = pickle.load(open(file2,'rb')) #relative
    location2 = cls_score2[1]
    cls_score2 = cls_score2[0]
    #cls_score = pickle.load(open('/m/tiangy/embeddings/bert/logs/test_pixel_num_margin/evaluation3/cls_score.pkl','rb')) #relative
    all_labels  = np.genfromtxt('model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    #id2cls = pickle.load(open('data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    id2cls_val = pickle.load(open('model_split/id2cls_val.pkl','rb'))#absolute
    loc_val = pickle.load(open('data/binary_all_multi_soft_withpixel/val.pkl','rb'))
    #loc_train = pickle.load(open('/scratch/tiangy/bert/data/binary_all_multi/train.pkl','rb')) #######todo
    def search_loc(cocoid,mod):
        l = loc_val if mod == 'val' else loc_train
        for i in l:
            if i[0] == cocoid:
                return i[3]

    seen_cls = np.load('model_split/split_2/seen_cls.npy')
    novel_cls = np.load('model_split/split_2/novel_cls.npy')

    novel_hard_cls = np.load('model_split/split_2/novel_hard.npy')
    novel_medium_cls = np.load('model_split/split_2/novel_medium.npy')
    novel_easy_cls = np.load('model_split/split_2/novel_easy.npy')

    class_all = np.concatenate([seen_cls,novel_cls])
    know_class = seen_cls
    ##
    mod = 'val'
    if mod =='val':
        id2cls_used = id2cls_val
    elif mod =='train':
        id2cls_used = id2cls

    all_mean_dif = [np.array([])  for i in range(num_class)]
    miss_count = 0
    for cocoid in id2cls_used:
        gt_loc_dic = search_loc(cocoid,mod)
        for i in novel_cls:
            if not i in gt_loc_dic:
                continue
            try:
                loc = location[cocoid+'_'+str(i)]
                loc2 = location2[cocoid+'_'+str(i)]
                if not softmax_scale is None:
                    loc = loc*softmax_scale
                    loc2 = loc2*softmax_scale
                loc = loc*scale + (1-scale)*loc2
                loc = np.exp(loc) / (np.exp(loc).sum())

                # loc = np.exp(loc) / (np.exp(loc).sum())
                # loc2 = np.exp(loc2) / (np.exp(loc2).sum())
                # loc = loc*scale + (1-scale)*loc2

                gt_loc = gt_loc_dic[i]
                gt_loc = gt_loc / gt_loc.sum()

            except:
                miss_count += 1
                continue

            #all_mean_dif[i] = np.append(all_mean_dif[i], np.abs(loc - gt_loc).mean())
            all_mean_dif[i] = np.append(all_mean_dif[i], (loc*gt_loc).sum())

    all_mean_dif = np.array([i.mean() for i in all_mean_dif])
    id2score = [] # n x 182
    gt = []
    for cocoid in id2cls_used:
    #for cocoid in id2cls_val:
        #base = cls_score[cocoid+'_-1']
        score_now = np.zeros(num_class)
        gt_cls = np.zeros(num_class)
        for i in range(num_class):
            # now = cls_score[cocoid+'_'+str(i)]
            # score_now[i] = now
            try:
                now = cls_score[cocoid+'_'+str(i)]
                score_now[i] = now
            except:
                pass
            #cls_dif[i] = np.dot(now,base)/(np.linalg.norm(now)*np.linalg.norm(base))
        gt_cls[id2cls_used[cocoid]] = 1
        id2score.append(score_now)
        gt.append(gt_cls)
    id2score,gt = np.stack(id2score).transpose(), np.stack(gt).transpose() # 182 x n

    AP_all = np.zeros(num_class)
    random_all = np.zeros(num_class)
    for i in class_all:
        pre = id2score[i]
        g = gt[i]
        AP,random = binary_mAP(pre,g)

        AP_all[i] = AP
        random_all[i] = random
        #print(all_labels[i],' ',AP)
    #np.save('/scratch/tiangy/bert/model_split/split_2/class_rank.npy',np.argsort(AP_all))
    # for i in np.argsort(AP_all):
    #   print(tokenizer.tokenize(all_labels[i]),AP_all[i],random_all[i])
    #   if AP_all[i]<random_all[i]:
    #       print('worse than random :',all_labels[i] )

    print('mAP results:')
    
    used = [  4,   9,  10,  11,  15,  18,  19,  21,  23,  26,  30,  35,  39,
    41,  42,  48,  51,  55,  58,  63,  72,  75,  77,  81,  82,  87,
    89,  90,  94, 100, 102, 107, 110, 113, 115, 117, 119, 124, 125,
    126, 133, 134, 136, 139, 141, 142, 145, 153, 154, 157, 166, 169,
    170, 172, 173, 174, 175, 176, 177, 178]
    #print("mean used class: ","%.3f"%np.nanmean(AP_all[used]),'   ',"%.3f"%np.nanmean(random_all[used]))
    print("mean seen class: ","%.3f"%np.nanmean(AP_all[know_class]),'   ',"%.3f"%np.nanmean(random_all[know_class]))
    print("mean unseen class: ","%.3f"%np.nanmean(AP_all[novel_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_cls]))
    print(' ')
    print('unseen class:')
    for i in novel_cls:
        print(all_labels[i],' : ',"%.3f"%AP_all[i],'   ',"%.3f"%random_all[i])
    print(' ')
    print("mean hard unseen class: ","%.3f"%np.nanmean(AP_all[novel_hard_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_hard_cls]))
    print("mean medium unseen class: ","%.3f"%np.nanmean(AP_all[novel_medium_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_medium_cls]))
    print("mean easy unseen class: ","%.3f"%np.nanmean(AP_all[novel_easy_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_easy_cls]))
    print(' ')
    print('miss count: ',miss_count)
    print("mean seen class diff: ","%.3f"%np.nanmean(all_mean_dif[know_class]))
    print("mean unseen class diff: ","%.3f"%np.mean(all_mean_dif[novel_cls]))
    print("mean hard unseen class diff: ","%.3f"%np.nanmean(all_mean_dif[novel_hard_cls]))
    print("mean medium unseen class diff: ","%.3f"%np.nanmean(all_mean_dif[novel_medium_cls]))
    print("mean easy unseen class diff: ","%.3f"%np.nanmean(all_mean_dif[novel_easy_cls]))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--score_file2",
                        default=None,
                        type=str,
                        required=False)
    args = parser.parse_args()
    """
    use mAP_append_local to calculate single file
    use mAP_append_local_en to merge two file
    """
    mAP_append_local(args.score_file,softmax_scale = 10)
    #mAP_append_local_en(file = args.score_file,file2 = args.score_file2,softmax_scale = 10)
    


