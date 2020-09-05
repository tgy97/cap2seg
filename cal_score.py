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

def cal_rank():
    cls_score = pickle.load(open('/m/tiangy/embeddings/bert/logs/test_pixel_num_margin/evaluation2/cls_score.pkl','rb')) #relative
    all_labels  = np.genfromtxt('/scratch/tiangy/SPNet/data/datasets/cocostuff/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train_pixel_num.pkl','rb')) #absolute

    id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute

    seen_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/seen_cls.npy')
    val_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/val_cls.npy')
    novel_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/novel_cls.npy')

    class_all = np.concatenate([seen_cls,val_cls,novel_cls])
    know_class = np.concatenate([seen_cls,val_cls])

    myGaussian = np.fromiter( (gaussian( x , 1 ) for x in range( -3, 4, 1 ) ), np.float )
    #myFilteredData = np.convolve( myData, myGaussian, mode='same' )

    pixel_num_ = []
    predict_  =  []
    for entry in pixel_num:
        cocoid = entry[0]
        pixel_num_.append(entry[2])
        pre_tmp = np.zeros(entry[2].shape)

        pre_tmp[class_all] = cls_score[cocoid]
        predict_.append(pre_tmp)

    pixel_num_ = np.stack(pixel_num_).transpose()
    predict_ = np.stack(predict_).transpose()
    for i in class_all:
        i = np.random.randint(182)
        plt.cla()

        x1 = pixel_num_[i][pixel_num_[i]!=0]
        x2 = predict_[i][pixel_num_[i]!=0]

        x1=x1[np.argsort(x2)]
        x2=np.sort(x2)
        x2 = np.convolve( x2, myGaussian, mode='same' )

        plt.scatter(x1,x2,s=0.05,alpha = 0.2) 
        plt.savefig('/scratch/tiangy/plt/{}.png'.format(str(all_labels[i])))
        print(all_labels[i])
        embed()

def binary_mAP(score,gt):
    size = gt.size
    true_num = (gt==1).sum()
    random = true_num/size
    if random == 0:
        random = np.nan
    
    sorted_gt = gt[np.argsort(score)]
    AP = 0
    for i,index in enumerate(np.argwhere(sorted_gt==1).flatten()):
        p = (true_num-i) / (size-index)
        AP += p
    AP = AP / true_num
    return AP,random

def mAP(file):
    num_class = 182
    cls_score = pickle.load(open(file,'rb')) #relative
    #cls_score = pickle.load(open('/m/tiangy/embeddings/bert/logs/test_pixel_num_margin/evaluation3/cls_score.pkl','rb')) #relative
    all_labels  = np.genfromtxt('/scratch/tiangy/SPNet/data/datasets/cocostuff/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute

    seen_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/seen_cls.npy')
    val_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/val_cls.npy')
    novel_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/novel_cls.npy')

    class_all = np.concatenate([seen_cls,val_cls,novel_cls])
    know_class = np.concatenate([seen_cls,val_cls]) 

    score_m = []
    gt = []
    for cocoid in cls_score:

        tmp = np.zeros(cls_score[cocoid].shape)
        tmp[class_all] = cls_score[cocoid]
        score_m.append(tmp)
        tmp_2 = np.zeros(num_class)
        #
        if not cocoid in id2cls:
            im = cv2.imread('/m/tiangy/coco_stuff_anno/val2017/'+ '{}.png'.format(str(cocoid)), cv2.IMREAD_GRAYSCALE)
            id2cls[cocoid] = np.setdiff1d(np.unique(im.flatten()),[255] )
        #
        tmp_2[id2cls[cocoid]] = 1
        gt.append(tmp_2)
    score_m = np.stack(score_m).transpose()  # 182 x 112321
    gt = np.stack(gt).transpose() # 182 x 112321
    AP_all = np.zeros(num_class)
    random_all = np.zeros(num_class)
    for i in class_all:
        pre = score_m[i]
        g = gt[i]
        AP,random = binary_mAP(pre,g)
        AP_all[i] = AP
        random_all[i] = random
        #print(all_labels[i],' ',AP)
    for i in np.argsort(AP_all):
        print(tokenizer.tokenize(all_labels[i]),AP_all[i],random_all[i])
        if AP_all[i]<random_all[i]:
            print('worse than random :',all_labels[i] )

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

def mAP_insert(file):
    num_class = 182
    cls_score = pickle.load(open(file,'rb')) 
    all_labels  = np.genfromtxt('/m/tiangy/embeddings/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute


    id2cls = pickle.load(open('/m/tiangy/embeddings/bert/model_split/id2cls_val.pkl','rb'))

    seen_cls = np.load('/m/tiangy/embeddings/bert/model_split/seen_cls.npy')
    val_cls = np.load('/m/tiangy/embeddings/bert/model_split/val_cls.npy')
    novel_cls = np.load('/m/tiangy/embeddings/bert/model_split/novel_cls.npy')

    class_all = np.concatenate([seen_cls,val_cls,novel_cls])
    know_class = np.concatenate([seen_cls,val_cls]) 

    id2dif = [] # n x 182
    gt = []
    for cocoid in id2cls:
        base = cls_score[cocoid+'_-1']
        cls_dif = np.zeros(num_class)
        gt_cls = np.zeros(num_class)
        for i in range(num_class):
            now = cls_score[cocoid+'_'+str(i)]
            
            cls_dif[i] = np.dot(now,base)/(np.linalg.norm(now)*np.linalg.norm(base))
        gt_cls[id2cls[cocoid]] = 1
        id2dif.append(cls_dif)
        gt.append(gt_cls)
    id2dif,gt = np.stack(id2dif).transpose(), np.stack(gt).transpose() # 182 x n

    id2dif = -id2dif  #### score increases as dif decreases

    print('mean appeared class score: ',id2dif[gt==1].mean())
    print('mean unappeared class score: ',id2dif[gt==0].mean())

    AP_all = np.zeros(num_class)
    random_all = np.zeros(num_class)
    for i in class_all:
        pre = id2dif[i]
        g = gt[i]
        AP,random = binary_mAP(pre,g)
        AP_all[i] = AP
        random_all[i] = random
        #print(all_labels[i],' ',AP)
    for i in np.argsort(AP_all):
        print(tokenizer.tokenize(all_labels[i]),AP_all[i],random_all[i])
        if AP_all[i]<random_all[i]:
            print('worse than random :',all_labels[i] )

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

def mAP_insert_caption_level(file): # each distance is cal from single caption
    num_class = 182
    cls_score = pickle.load(open(file,'rb')) 
    all_labels  = np.genfromtxt('/m/tiangy/embeddings/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute


    id2cls = pickle.load(open('/m/tiangy/embeddings/bert/model_split/id2cls_val.pkl','rb'))

    seen_cls = np.load('/m/tiangy/embeddings/bert/model_split/seen_cls.npy')
    val_cls = np.load('/m/tiangy/embeddings/bert/model_split/val_cls.npy')
    novel_cls = np.load('/m/tiangy/embeddings/bert/model_split/novel_cls.npy')

    class_all = np.concatenate([seen_cls,val_cls,novel_cls])
    know_class = np.concatenate([seen_cls,val_cls]) 

    id2dif = [] # n x 182
    gt = []
    for cocoid in id2cls:
        if not cocoid + '_-1_0' in cls_score:
            continue
        score_now_caption = []
        for caption_id in range(10):
            if not cocoid+'_-1'+'_'+str(caption_id) in cls_score:
                break
            base = cls_score[cocoid+'_-1'+'_'+str(caption_id)]
            cls_dif = np.zeros(num_class)
            for i in range(num_class):
                now = cls_score[cocoid+'_'+str(i)+ '_' + str(caption_id)]
                
                cls_dif[i] = np.dot(now,base)/(np.linalg.norm(now)*np.linalg.norm(base))
            cls_dif = np.exp(cls_dif) / (np.exp(cls_dif).sum())
            score_now_caption.append(cls_dif)
        score_now_caption = np.stack(score_now_caption).mean(0) # 182
        id2dif.append(score_now_caption)

        gt_cls = np.zeros(num_class)
        gt_cls[id2cls[cocoid]] = 1
        gt.append(gt_cls)
    id2dif,gt = np.stack(id2dif).transpose(), np.stack(gt).transpose() # 182 x n
    print(id2dif.shape,gt.shape)

    id2dif = -id2dif  #### score increases as dif decreases

    print('mean appeared class score: ',id2dif[gt==1].mean())
    print('mean unappeared class score: ',id2dif[gt==0].mean())

    AP_all = np.zeros(num_class)
    random_all = np.zeros(num_class)
    for i in class_all:
        pre = id2dif[i]
        g = gt[i]
        AP,random = binary_mAP(pre,g)
        AP_all[i] = AP
        random_all[i] = random
        #print(all_labels[i],' ',AP)
    for i in np.argsort(AP_all):
        print(tokenizer.tokenize(all_labels[i]),AP_all[i],random_all[i])
        if AP_all[i]<random_all[i]:
            print('worse than random :',all_labels[i] )

    print('mAP results:')
    used = [  4,   9,  10,  11,  15,  18,  19,  21,  23,  26,  30,  35,  39,
    41,  42,  48,  51,  55,  58,  63,  72,  75,  77,  81,  82,  87,
    89,  90,  94, 100, 102, 107, 110, 113, 115, 117, 119, 124, 125,
    126, 133, 134, 136, 139, 141, 142, 145, 153, 154, 157, 166, 169,
    170, 172, 173, 174, 175, 176, 177, 178]
    #print("mean used class: ","%.3f"%np.nanmean(AP_all[used]),'   ',"%.3f"%np.nanmean(random_all[used]))
    print("mean seen class: ","%.3f"%np.nanmean(AP_all[know_class]),'   ',"%.3f"%np.nanmean(random_all[know_class]))
    print("mean unseen class: ","%.3f"%np.nanmean(AP_all[novel_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_cls]))
    print("mean all class: ","%.3f"%np.nanmean(AP_all),'   ',"%.3f"%np.nanmean(random_all))
    print(' ')
    print('unseen class:')
    for i in novel_cls:
        print(all_labels[i],' : ',"%.3f"%AP_all[i],'   ',"%.3f"%random_all[i])


def mAP_insert_caption_level_notmerge(file): # each distance is cal from single caption
    num_class = 182
    cls_score = pickle.load(open(file,'rb')) 
    all_labels  = np.genfromtxt('/m/tiangy/embeddings/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute


    id2cls = pickle.load(open('/m/tiangy/embeddings/bert/model_split/id2cls_val.pkl','rb'))

    seen_cls = np.load('/m/tiangy/embeddings/bert/model_split/seen_cls.npy')
    val_cls = np.load('/m/tiangy/embeddings/bert/model_split/val_cls.npy')
    novel_cls = np.load('/m/tiangy/embeddings/bert/model_split/novel_cls.npy')

    class_all = np.concatenate([seen_cls,val_cls,novel_cls])
    know_class = np.concatenate([seen_cls,val_cls]) 

    id2dif = [] # n x 182
    gt = []
    for cocoid in id2cls:
        if not cocoid + '_-1_0' in cls_score:
            continue
        score_now_caption = []
        for caption_id in range(10):
            if not cocoid+'_-1'+'_'+str(caption_id) in cls_score:
                break
            base = cls_score[cocoid+'_-1'+'_'+str(caption_id)]
            cls_dif = np.zeros(num_class)
            for i in range(num_class):
                now = cls_score[cocoid+'_'+str(i)+ '_' + str(caption_id)]
                
                cls_dif[i] = np.dot(now,base)/(np.linalg.norm(now)*np.linalg.norm(base))
            cls_dif = cls_dif / np.sqrt((cls_dif*cls_dif).sum())
            score_now_caption.append(cls_dif)
        score_now_caption = np.stack(score_now_caption).mean(0) # 182
        id2dif.append(score_now_caption)

        gt_cls = np.zeros(num_class)
        gt_cls[id2cls[cocoid]] = 1
        gt.append(gt_cls)
    id2dif,gt = np.stack(id2dif).transpose(), np.stack(gt).transpose() # 182 x n
    print(id2dif.shape,gt.shape)

    id2dif = -id2dif  #### score increases as dif decreases

    print('mean appeared class score: ',id2dif[gt==1].mean())
    print('mean unappeared class score: ',id2dif[gt==0].mean())

    AP_all = np.zeros(num_class)
    random_all = np.zeros(num_class)
    for i in class_all:
        pre = id2dif[i]
        g = gt[i]
        AP,random = binary_mAP(pre,g)
        AP_all[i] = AP
        random_all[i] = random
        #print(all_labels[i],' ',AP)
    for i in np.argsort(AP_all):
        print(tokenizer.tokenize(all_labels[i]),AP_all[i],random_all[i])
        if AP_all[i]<random_all[i]:
            print('worse than random :',all_labels[i] )

    print('mAP results:')
    used = [  4,   9,  10,  11,  15,  18,  19,  21,  23,  26,  30,  35,  39,
    41,  42,  48,  51,  55,  58,  63,  72,  75,  77,  81,  82,  87,
    89,  90,  94, 100, 102, 107, 110, 113, 115, 117, 119, 124, 125,
    126, 133, 134, 136, 139, 141, 142, 145, 153, 154, 157, 166, 169,
    170, 172, 173, 174, 175, 176, 177, 178]
    #print("mean used class: ","%.3f"%np.nanmean(AP_all[used]),'   ',"%.3f"%np.nanmean(random_all[used]))
    print("mean seen class: ","%.3f"%np.nanmean(AP_all[know_class]),'   ',"%.3f"%np.nanmean(random_all[know_class]))
    print("mean unseen class: ","%.3f"%np.nanmean(AP_all[novel_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_cls]))
    print("mean all class: ","%.3f"%np.nanmean(AP_all),'   ',"%.3f"%np.nanmean(random_all))
    print(' ')
    print('unseen class:')
    for i in novel_cls:
        print(all_labels[i],' : ',"%.3f"%AP_all[i],'   ',"%.3f"%random_all[i])

def cal_AP(file):

    num_class = 182
    cls_score = pickle.load(open(file,'rb')) #relative
    #cls_score = pickle.load(open('/m/tiangy/embeddings/bert/logs/test_pixel_num_margin/evaluation3/cls_score.pkl','rb')) #relative
    all_labels  = np.genfromtxt('/scratch/tiangy/SPNet/data/datasets/cocostuff/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute

    seen_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/seen_cls.npy')
    val_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/val_cls.npy')
    novel_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/novel_cls.npy')

    class_all = np.concatenate([seen_cls,val_cls,novel_cls])
    know_class = np.concatenate([seen_cls,val_cls]) 

    score_m = []
    gt = []
    for cocoid in cls_score:
        cls_score[cocoid] = 1/(1+np.exp(-cls_score[cocoid]))
        tmp = np.zeros(cls_score[cocoid].shape)
        tmp[class_all] = cls_score[cocoid]
        score_m.append(tmp)
        tmp_2 = np.zeros(num_class)
        #
        if not cocoid in id2cls:
            im = cv2.imread('/m/tiangy/coco_stuff_anno/val2017/'+ '{}.png'.format(str(cocoid)), cv2.IMREAD_GRAYSCALE)
            id2cls[cocoid] = np.setdiff1d(np.unique(im.flatten()),[255] )
        #
        tmp_2[id2cls[cocoid]] = 1
        gt.append(tmp_2)
    score_m = np.stack(score_m).transpose()  # 182 x 112321
    gt = np.stack(gt).transpose() # 182 x 112321
    AP_all = np.zeros(num_class)
    random_all = np.zeros(num_class)
    for i in class_all:
        if i in novel_cls:
            thresh = 0.05
        else:
            thresh = 0.5
        pre = score_m[i]
        g = gt[i]
        t =(g[pre>thresh] == 1).sum()#+(g[pre<thresh] == 0).sum()
        AP = (t/((pre>thresh).sum()+1))
        #embed()
        random = (g==1).sum() / g.size
        random = 1 - thresh - random + 2 * random * thresh
        #embed()

        #embed()
        AP_all[i] = AP
        random_all[i] = random
        #print(all_labels[i],' ',AP)
    for i in np.argsort(AP_all):
        print(tokenizer.tokenize(all_labels[i]),AP_all[i],random_all[i])
        #pass

    print('mAP results:')
    print("mean seen class: ","%.3f"%np.mean(AP_all[know_class]),'   ',"%.3f"%np.nanmean(random_all[know_class]))
    print("mean unseen class: ","%.3f"%np.nanmean(AP_all[novel_cls]),'   ',"%.3f"%np.nanmean(random_all[novel_cls]))
    print(' ')
    print('unseen class:')
    for i in novel_cls:
        print(all_labels[i],' : ',"%.3f"%AP_all[i],'   ',"%.3f"%random_all[i])


def mAP_2(file): #  split_2
    num_class = 182
    cls_score = pickle.load(open(file,'rb')) #relative
    #cls_score = pickle.load(open('/m/tiangy/embeddings/bert/logs/test_pixel_num_margin/evaluation3/cls_score.pkl','rb')) #relative
    all_labels  = np.genfromtxt('/scratch/tiangy/SPNet/data/datasets/cocostuff/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    id2cls_val = pickle.load(open('/scratch/tiangy/bert/model_split/id2cls_val.pkl','rb'))#absolute

    seen_cls = np.load('/scratch/tiangy/bert/model_split/split_2/seen_cls.npy')
    novel_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_cls.npy')

    novel_hard_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_hard.npy')
    novel_medium_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_medium.npy')
    novel_easy_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_easy.npy')

    class_all = np.concatenate([seen_cls,novel_cls])
    know_class = seen_cls

    score_m = []
    gt = []
    for cocoid in cls_score:

        tmp = np.zeros(cls_score[cocoid].shape)
        tmp[class_all] = cls_score[cocoid]
        score_m.append(tmp)
        tmp_2 = np.zeros(num_class)
        #
        if not cocoid in id2cls:
            id2cls[cocoid] = id2cls_val[cocoid]
            #im = cv2.imread('/m/tiangy/coco_stuff_anno/val2017/'+ '{}.png'.format(str(cocoid)), cv2.IMREAD_GRAYSCALE)
            #id2cls[cocoid] = np.setdiff1d(np.unique(im.flatten()),[255] )
        #
        tmp_2[id2cls[cocoid]] = 1
        gt.append(tmp_2)
    score_m = np.stack(score_m).transpose()  # 182 x 112321
    gt = np.stack(gt).transpose() # 182 x 112321
    AP_all = np.zeros(num_class)
    random_all = np.zeros(num_class)
    for i in class_all:
        pre = score_m[i]
        g = gt[i]
        AP,random = binary_mAP(pre,g)
        AP_all[i] = AP
        random_all[i] = random
        #print(all_labels[i],' ',AP)
    for i in np.argsort(AP_all):
        print(tokenizer.tokenize(all_labels[i]),AP_all[i],random_all[i])
        if AP_all[i]<random_all[i]:
            print('worse than random :',all_labels[i] )

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






def cal_score():
    cls_score = pickle.load(open('/m/tiangy/embeddings/bert/logs/test/evaluation_all/cls_score.pkl','rb'))
    all_labels  = np.genfromtxt('/scratch/tiangy/SPNet/data/datasets/cocostuff/labels_2.txt', delimiter='\t', usecols=1, dtype='str')

    id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))

    seen_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/seen_cls.npy')
    val_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/val_cls.npy')
    novel_cls = np.load('/scratch/tiangy/SPNet/data/datasets/cocostuff/split/novel_cls.npy')

    class_all = np.concatenate([seen_cls,val_cls,novel_cls])
    know_class = np.concatenate([seen_cls,val_cls])


    im_dir = '/m/tiangy/coco_stuff_anno/train2017/'

    know = []
    all_know = []
    unknow = []
    all_unknow = []
    mean = []

    novel_mean = {}
    t_out = {}
    novel_rank = {}
    all_novel_rank = {}

    novel_rank_in_unseen = {}
    all_novel_rank_in_unseen = {}
    num = 0
    for n,i in enumerate(cls_score):
        z = cls_score[i]
        z_o =  1/(1 + np.exp(-z))  
        #z_o = z
        #z_o = np.exp(z_o) / (np.exp(z_o).sum())
        z[class_all] = z_o


        g = id2cls[i]
        #t_out[i] = g

        novel_part = np.intersect1d(g,novel_cls)
        seen_part = np.intersect1d(g,know_class)
        all_part = np.intersect1d(g,class_all)

        know.append(z[seen_part].mean())
        all_know.append(z[know_class].mean())
        unknow.append(z[novel_part].mean())
        all_unknow.append(z[novel_cls].mean())
        mean.append(z[all_part].mean())

        for i in novel_part:
            if all_labels[i] in novel_mean:
                novel_mean[all_labels[i]].append(z[i])
            else:
                novel_mean[all_labels[i]] = [z[i]]

        for i in novel_part:
            if all_labels[i] in novel_rank:
                novel_rank[all_labels[i]].append( (np.argwhere(np.argsort(z)==i)[0]) )
            else:
                novel_rank[all_labels[i]] = [(np.argwhere(np.argsort(z)==i)[0]) ]

        for i in novel_cls:
            if all_labels[i] in all_novel_rank:
                all_novel_rank[all_labels[i]].append( (np.argwhere(np.argsort(z)==i)[0]) )
            else:
                all_novel_rank[all_labels[i]] = [(np.argwhere(np.argsort(z)==i)[0]) ]

        for n,i in enumerate(novel_part):
            if all_labels[i] in novel_rank_in_unseen:
                novel_rank_in_unseen[all_labels[i]].append( (np.argwhere(np.argsort(z[novel_cls])==n)[0]) )
            else:
                novel_rank_in_unseen[all_labels[i]] = [(np.argwhere(np.argsort(z[novel_cls])==n)[0]) ]

        for n,i in enumerate(novel_cls):
            if all_labels[i] in all_novel_rank_in_unseen:
                all_novel_rank_in_unseen[all_labels[i]].append( (np.argwhere(np.argsort(z[novel_cls])==n)[0]) )
            else:
                all_novel_rank_in_unseen[all_labels[i]] = [(np.argwhere(np.argsort(z[novel_cls])==n)[0]) ]

    know,all_know,unknow,all_unknow,mean = np.array(know),np.array(all_know),np.array(unknow),np.array(all_unknow),np.array(mean)

    know[np.isnan(know)] = 0
    all_know[np.isnan(all_know)] = 0
    unknow[np.isnan(unknow)] = 0
    all_unknow[np.isnan(all_unknow)] = 0
    mean[np.isnan(mean)] = 0

    for key in novel_mean:
        novel_mean[key] = np.array(novel_mean[key]).mean()
    for key in novel_rank:
        novel_rank[key] = np.array(novel_rank[key]).mean()
    for key in all_novel_rank:
        all_novel_rank[key] = np.array(all_novel_rank[key]).mean()
    for key in novel_rank_in_unseen:
        novel_rank_in_unseen[key] = np.array(novel_rank_in_unseen[key]).mean()
    for key in all_novel_rank_in_unseen:
        all_novel_rank_in_unseen[key] = np.array(all_novel_rank_in_unseen[key]).mean()


    print('softmaxed score:')
    print('appeared seen class avg:',know.mean())
    print('all seen class avg:',all_know.mean())
    print('appeared unseen class avg:',unknow.mean())
    print('all unseen class avg:',all_unknow.mean())
    print('mean',mean.mean())
    pp = pprint.PrettyPrinter(indent = 1)
    print(' ')
    print('unseen class avg:')
    pp.pprint(novel_mean)
    print('unseen class rank(appeared, all):')
    for key in novel_rank:
        print(key,"%.2f"%(182-novel_rank[key]),"%.2f"%(182-all_novel_rank[key]))
    print(' ')
    print('unseen class rank in unseen(appeared, all):')
    for key in novel_rank_in_unseen:
        print(key,"%.2f"%(15-novel_rank_in_unseen[key]),"%.2f"%(15-all_novel_rank_in_unseen[key]))
    #print(novel_mean)
    # print('know',know/num)
    # print('unknow',unknow/num)
    # print('all unknow',all_unknow/num)
    # print('mean',mean/num)
def mAP_append(file):
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
    if len(cls_score) == 2:
        cls_score=cls_score[0]
    #cls_score = pickle.load(open('/m/tiangy/embeddings/bert/logs/test_pixel_num_margin/evaluation3/cls_score.pkl','rb')) #relative
    all_labels  = np.genfromtxt('/scratch/tiangy/SPNet/data/datasets/cocostuff/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    id2cls_val = pickle.load(open('/scratch/tiangy/bert/model_split/id2cls_val.pkl','rb'))#absolute

    seen_cls = np.load('/scratch/tiangy/bert/model_split/split_2/seen_cls.npy')
    novel_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_cls.npy')

    novel_hard_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_hard.npy')
    novel_medium_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_medium.npy')
    novel_easy_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_easy.npy')

    class_all = np.concatenate([seen_cls,novel_cls])
    know_class = seen_cls
    ##
    mod = 'val'
    if mod =='val':
        id2cls_used = id2cls_val
    elif mod =='train':
        id2cls_used = id2cls

    id2score = [] # n x 182
    gt = []
    for cocoid in id2cls_used:
    #for cocoid in id2cls_val:
        #base = cls_score[cocoid+'_-1']
        score_now = np.zeros(num_class)
        gt_cls = np.zeros(num_class)
        for i in novel_cls:
            now = cls_score[cocoid+'_'+str(i)]
            score_now[i] = now
            # try:
            #   now = cls_score[cocoid+'_'+str(i)]
            #   score_now[i] = now
            # except:
            #   pass
            #cls_dif[i] = np.dot(now,base)/(np.linalg.norm(now)*np.linalg.norm(base))
        gt_cls[id2cls_used[cocoid]] = 1
        id2score.append(score_now)
        gt.append(gt_cls)
    id2score,gt = np.stack(id2score).transpose(), np.stack(gt).transpose() # 182 x n

    #id2dif = -id2dif  #### score increases as dif decreases

    # print('mean appeared class score: ',id2score[gt==1].mean())
    # print('mean unappeared class score: ',id2score[gt==0].mean())

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
    all_labels  = np.genfromtxt('/m/tiangy/embeddings/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    #id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    id2cls_val = pickle.load(open('/m/tiangy/embeddings/bert/model_split/id2cls_val.pkl','rb'))#absolute
    loc_val = pickle.load(open('/m/tiangy/embeddings/bert/data/binary_all_multi_soft/val.pkl','rb'))
    #loc_train = pickle.load(open('/scratch/tiangy/bert/data/binary_all_multi/train.pkl','rb')) #######todo
    def search_loc(cocoid,mod):
        l = loc_val if mod == 'val' else loc_train
        for i in l:
            if i[0] == cocoid:
                return i[3]

    seen_cls = np.load('/scratch/tiangy/bert/model_split/split_2/seen_cls.npy')
    novel_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_cls.npy')

    novel_hard_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_hard.npy')
    novel_medium_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_medium.npy')
    novel_easy_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_easy.npy')

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
    all_labels  = np.genfromtxt('/m/tiangy/embeddings/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    #id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    id2cls_val = pickle.load(open('/m/tiangy/embeddings/bert/model_split/id2cls_val.pkl','rb'))#absolute
    loc_val = pickle.load(open('/m/tiangy/embeddings/bert/data/binary_all_multi_soft/val.pkl','rb'))
    #loc_train = pickle.load(open('/scratch/tiangy/bert/data/binary_all_multi/train.pkl','rb')) #######todo
    def search_loc(cocoid,mod):
        l = loc_val if mod == 'val' else loc_train
        for i in l:
            if i[0] == cocoid:
                return i[3]

    seen_cls = np.load('/scratch/tiangy/bert/model_split/split_2/seen_cls.npy')
    novel_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_cls.npy')

    novel_hard_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_hard.npy')
    novel_medium_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_medium.npy')
    novel_easy_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_easy.npy')

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

def cal_IOU(file):
    #decapreted
    num_class = 182
    cls_score = pickle.load(open(file,'rb')) #relative
    location = cls_score[1]
    cls_score = cls_score[0]

    all_labels  = np.genfromtxt('/m/tiangy/embeddings/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    #id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    id2cls_val = pickle.load(open('/scratch/tiangy/bert/model_split/id2cls_val.pkl','rb'))#absolute
    loc_val = pickle.load(open('/scratch/tiangy/bert/data/binary_all_multi/val.pkl','rb'))
    #loc_train = pickle.load(open('/scratch/tiangy/bert/data/binary_all_multi/train.pkl','rb')) #######todo
    def search_loc(cocoid,mod):
        l = loc_val if mod == 'val' else loc_train
        for i in l:
            if i[0] == cocoid:
                return i[3]
    def re_scale(x):
        if x <0:
            return 0
        if x >1:
            return 1
        return x
    seen_cls = np.load('/scratch/tiangy/bert/model_split/split_2/seen_cls.npy')
    novel_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_cls.npy')

    novel_hard_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_hard.npy')
    novel_medium_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_medium.npy')
    novel_easy_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_easy.npy')

    class_all = np.concatenate([seen_cls,novel_cls])
    know_class = seen_cls
    ##
    c_map = np.zeros(256)
    c_map[know_class] = 0
    c_map[novel_cls] = 1
    def cal_iou(pred,im,cls):
        '''
        n x w
        h x w
        ''' 
        i = (pred == cls).sum()
        u1 = c_map[pred].sum()
        u2 = (im == cls).sum()

        iou = i / (u1+u2-i)
        return iou

    mod = 'val'
    if mod =='val':
        id2cls_used = id2cls_val
    elif mod =='train':
        id2cls_used = id2cls

    all_iou_base = [np.array([]) for i in range(num_class)]
    all_iou_pred = [np.array([]) for i in range(num_class)]
    all_iou_upper = [np.array([]) for i in range(num_class)]
    all_list = [all_iou_base,all_iou_pred,all_iou_upper]

    for index,cocoid in enumerate(id2cls_used):
        im = cv2.imread('/m/tiangy/coco_stuff_anno/val2017/'+ '{}.png'.format(str(cocoid)), cv2.IMREAD_GRAYSCALE)
        h = im.shape[0]
        ap_all = np.setdiff1d(np.unique(im.flatten()),255)
        ap_unseen = np.setdiff1d(np.unique(im.flatten()),np.append(know_class,255))
        for c in ap_unseen:
            pred = location[cocoid+'_'+str(c)]
            mi = re_scale(pred[0] - pred[1])
            ma = re_scale(pred[0] + pred[1])
            mi = int(round(mi*h))
            ma = int(round(ma*h))

            pred_scale = im[mi:ma]

            h_now = np.where(im == c)[0]

            upper = im[h_now.min():(h_now.max()+1)]

            iou_pred = cal_iou(pred_scale,im,c)
            iou_base = cal_iou(im,im,c)
            iou_upper = cal_iou(upper,im,c)

            # all_iou_pred[c] = np.append(all_iou_pred[c],iou_pred)
            # all_iou_base[c] = np.append(all_iou_base[c],iou_base)
            # all_iou_upper[c] = np.append(all_iou_upper[c],iou_upper)
            now_list = [iou_base,iou_pred,iou_upper]
            for i in range(3):
                all_list[i][c] = np.append(all_list[i][c],now_list[i])
        if index % 100 == 0:
            print(index)

    for i in range(3):
        all_list[i] = np.array([x.mean() for x in all_list[i]])

    name = ['base','pred','upperbound']
    print('seen class: ')
    for i in range(3):
        print(name[i]+" : ", np.nanmean(all_list[i][know_class]))
    print('unseen class: ')
    for i in range(3):
        print(name[i]+" : ", np.mean(all_list[i][novel_cls]))


def cal_IOU_soft(file,use_softmax = 1):
    #soft iou
    num_class = 182
    soft_num = 3
    soft_t = np.array(range(soft_num)) / soft_num
    soft_t = np.append(soft_t,1)

    cls_score = pickle.load(open(file,'rb')) #relative
    location = cls_score[1]
    cls_score = cls_score[0]

    all_labels  = np.genfromtxt('/m/tiangy/embeddings/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute

    id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    id2cls_val = pickle.load(open('/m/tiangy/embeddings/bert/model_split/id2cls_val.pkl','rb'))#absolute
    #loc_val = pickle.load(open('/m/tiangy/embeddings/bert/data/binary_all_multi_soft/val.pkl','rb'))
    #loc_train = pickle.load(open('/scratch/tiangy/bert/data/binary_all_multi_soft/train.pkl','rb')) #######todo

    # def search_loc(cocoid,mod):
    #     l = loc_val if mod == 'val' else loc_train
    #     for i in l:
    #         if i[0] == cocoid:
    #             return i[3]
    # # def re_scale(x):
    # #     if x <0:
    # #         return 0
    # #     if x >1:
    # #         return 1
    # #     return x
    seen_cls = np.load('/scratch/tiangy/bert/model_split/split_2/seen_cls.npy')
    s_size = seen_cls.size
    novel_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_cls.npy')
    u_size = novel_cls.size
    novel_hard_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_hard.npy')
    novel_medium_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_medium.npy')
    novel_easy_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_easy.npy')

    class_all = np.concatenate([seen_cls,novel_cls])
    know_class = seen_cls
    ##
    c_map = np.full(256,class_all.size)
    for i,x in enumerate(class_all):
        c_map[x] = i

    mod = 'val'
    if mod =='val':
        id2cls_used = id2cls_val
    elif mod =='train':
        id2cls_used = id2cls

    # all_iou_base = [np.array([]) for i in range(num_class)]
    # all_iou_pred = [np.array([]) for i in range(num_class)]
    # all_iou_upper = [np.array([]) for i in range(num_class)]
    # all_list = [all_iou_base,all_iou_pred,all_iou_upper]
    iou_base = [np.array([]) for i in range(num_class)]
    iou_pred = [np.array([]) for i in range(num_class)]
    miss_num = 0
    if use_softmax == None:
        print('Not use softmax')
    else:
        print('Using softmax scale: ', use_softmax)
    for index,cocoid in enumerate(id2cls_used):
        im = cv2.imread('/m/tiangy/coco_stuff_anno/{}2017/'.format(mod)+ '{}.png'.format(str(cocoid)), cv2.IMREAD_GRAYSCALE)
        h = im.shape[0]
        
        t_im = c_map[im]
        back_num = (t_im >= s_size).sum()
        
        
        for cls in novel_cls:
            if not cls in im:
                continue
            try:
                loc_cls = location[cocoid+'_' + str(cls)]
            except:
                miss_num = miss_num+1
                continue
            iou_base[cls] = np.append(iou_base[cls],(im ==cls).sum()/back_num)
            if not use_softmax is None:
                loc_cls = loc_cls * use_softmax
                loc_cls = np.exp(loc_cls) / (np.exp(loc_cls).sum())
            
            pixel_cls = np.where(t_im == c_map[cls])[0] / h
            pixel_back = np.where(t_im >= s_size)[0] / h
            score_cls = 0
            score_back = 0
            for i in range(soft_num):
                score_cls += ((pixel_cls<soft_t[i+1]).sum() - (pixel_cls<soft_t[i]).sum()) * loc_cls[i]
                score_back += ((pixel_back<soft_t[i+1]).sum() - (pixel_back<soft_t[i]).sum()) * loc_cls[i]
            iou_pred[cls] = np.append(iou_pred[cls] ,score_cls/score_back)
        if index % 10000 == 0:
            print(index)
    iou_base = np.array([p.mean() for p in iou_base])
    iou_pred = np.array([p.mean() for p in iou_pred])
    print("miss num: ", miss_num)
    print("mean seen class: ","%.3f"%np.nanmean(iou_pred[know_class]),'   ',"%.3f"%np.nanmean(iou_base[know_class]))
    print("mean unseen class: ","%.3f"%np.nanmean(iou_pred[novel_cls]),'   ',"%.3f"%np.nanmean(iou_base[novel_cls]))
    print(' ')
    print('unseen class:')
    for i in novel_cls:
        print(all_labels[i],' : ',"%.3f"%iou_pred[i],'   ',"%.3f"%iou_base[i])
    print(' ')
    print("mean hard unseen class: ","%.3f"%np.nanmean(iou_pred[novel_hard_cls]),'   ',"%.3f"%np.nanmean(iou_base[novel_hard_cls]))
    print("mean medium unseen class: ","%.3f"%np.nanmean(iou_pred[novel_medium_cls]),'   ',"%.3f"%np.nanmean(iou_base[novel_medium_cls]))
    print("mean easy unseen class: ","%.3f"%np.nanmean(iou_pred[novel_easy_cls]),'   ',"%.3f"%np.nanmean(iou_base[novel_easy_cls]))

def cal_IOU_soft_en(file,file2,scale = 0.5):
    #soft iou
    num_class = 182
    soft_num = 3
    soft_t = np.array(range(soft_num)) / soft_num
    soft_t = np.append(soft_t,1)

    cls_score = pickle.load(open(file,'rb')) #relative
    location = cls_score[1]
    cls_score = cls_score[0]

    cls_score2 = pickle.load(open(file2,'rb')) #relative
    location2 = cls_score2[1]
    cls_score2 = cls_score2[0]

    all_labels  = np.genfromtxt('/m/tiangy/embeddings/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute

    id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    id2cls_val = pickle.load(open('/m/tiangy/embeddings/bert/model_split/id2cls_val.pkl','rb'))#absolute
    #loc_val = pickle.load(open('/m/tiangy/embeddings/bert/data/binary_all_multi_soft/val.pkl','rb'))
    #loc_train = pickle.load(open('/scratch/tiangy/bert/data/binary_all_multi_soft/train.pkl','rb')) #######todo

    # def search_loc(cocoid,mod):
    #     l = loc_val if mod == 'val' else loc_train
    #     for i in l:
    #         if i[0] == cocoid:
    #             return i[3]
    # # def re_scale(x):
    # #     if x <0:
    # #         return 0
    # #     if x >1:
    # #         return 1
    # #     return x
    seen_cls = np.load('/scratch/tiangy/bert/model_split/split_2/seen_cls.npy')
    s_size = seen_cls.size
    novel_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_cls.npy')
    u_size = novel_cls.size
    novel_hard_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_hard.npy')
    novel_medium_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_medium.npy')
    novel_easy_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_easy.npy')

    class_all = np.concatenate([seen_cls,novel_cls])
    know_class = seen_cls
    ##
    c_map = np.full(256,class_all.size)
    for i,x in enumerate(class_all):
        c_map[x] = i

    mod = 'val'
    if mod =='val':
        id2cls_used = id2cls_val
    elif mod =='train':
        id2cls_used = id2cls

    # all_iou_base = [np.array([]) for i in range(num_class)]
    # all_iou_pred = [np.array([]) for i in range(num_class)]
    # all_iou_upper = [np.array([]) for i in range(num_class)]
    # all_list = [all_iou_base,all_iou_pred,all_iou_upper]
    iou_base = [np.array([]) for i in range(num_class)]
    iou_pred = [np.array([]) for i in range(num_class)]
    miss_num = 0
    print('Scale: ',scale)
    for index,cocoid in enumerate(id2cls_used):
        im = cv2.imread('/m/tiangy/coco_stuff_anno/{}2017/'.format(mod)+ '{}.png'.format(str(cocoid)), cv2.IMREAD_GRAYSCALE)
        h = im.shape[0]
        
        t_im = c_map[im]
        back_num = (t_im >= s_size).sum()
        
        
        for cls in novel_cls:
            if not cls in im:
                continue
            iou_base[cls] = np.append(iou_base[cls],(im ==cls).sum()/back_num)
            try:
                loc_cls = location[cocoid+'_' + str(cls)]
                loc_cls2 = location2[cocoid+'_' + str(cls)]
                
            except:
                miss_num = miss_num+1
                continue
            loc_cls *= 10
            loc_cls2 *= 10
            loc_cls = np.exp(loc_cls) / (np.exp(loc_cls).sum())
            loc_cls2 = np.exp(loc_cls2) / (np.exp(loc_cls2).sum())
            loc_cls = loc_cls*scale + loc_cls2*(1-scale)

            pixel_cls = np.where(t_im == c_map[cls])[0] / h
            pixel_back = np.where(t_im >= s_size)[0] / h
            score_cls = 0
            score_back = 0
            for i in range(soft_num):
                score_cls += ((pixel_cls<soft_t[i+1]).sum() - (pixel_cls<soft_t[i]).sum()) * loc_cls[i]
                score_back += ((pixel_back<soft_t[i+1]).sum() - (pixel_back<soft_t[i]).sum()) * loc_cls[i]
            iou_pred[cls] = np.append(iou_pred[cls] ,score_cls/score_back)
        if index % 10000 == 0:
            print(index)
    iou_base = np.array([p.mean() for p in iou_base])
    iou_pred = np.array([p.mean() for p in iou_pred])
    print("miss num: ", miss_num)
    print("mean seen class: ","%.3f"%np.nanmean(iou_pred[know_class]),'   ',"%.3f"%np.nanmean(iou_base[know_class]))
    print("mean unseen class: ","%.3f"%np.nanmean(iou_pred[novel_cls]),'   ',"%.3f"%np.nanmean(iou_base[novel_cls]))
    print(' ')
    print('unseen class:')
    for i in novel_cls:
        print(all_labels[i],' : ',"%.3f"%iou_pred[i],'   ',"%.3f"%iou_base[i])
    print(' ')
    print("mean hard unseen class: ","%.3f"%np.nanmean(iou_pred[novel_hard_cls]),'   ',"%.3f"%np.nanmean(iou_base[novel_hard_cls]))
    print("mean medium unseen class: ","%.3f"%np.nanmean(iou_pred[novel_medium_cls]),'   ',"%.3f"%np.nanmean(iou_base[novel_medium_cls]))
    print("mean easy unseen class: ","%.3f"%np.nanmean(iou_pred[novel_easy_cls]),'   ',"%.3f"%np.nanmean(iou_base[novel_easy_cls]))

def mAP_pascal(file):
    import csv
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
    num_class = 21
    cls_score = pickle.load(open(file,'rb')) #relative
    cls_score = cls_score[0]
    data = pickle.load(open('/m/tiangy/embeddings/bert/data/pascal/train.pkl','rb'))
    #cls_score = pickle.load(open('/m/tiangy/embeddings/bert/logs/test_pixel_num_margin/evaluation3/cls_score.pkl','rb')) #relative
    all_labels  = np.genfromtxt('/m/tiangy/embeddings/bert/pascal_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
    #pixel_num = pickle.load(open('/m/tiangy/embeddings/bert/data/train.pkl','rb')) #absolute

    # id2cls = pickle.load(open('/scratch/tiangy/SPNet/data/datasets/cocostuff/id2cls.pkl','rb'))#absolute
    # id2cls_val = pickle.load(open('/scratch/tiangy/bert/model_split/id2cls_val.pkl','rb'))#absolute

    seen_cls = np.concatenate([np.load('/m/tiangy/embeddings/bert/pascal_split/split/seen_cls.npy'),np.load('/m/tiangy/embeddings/bert/pascal_split/split/val_cls.npy')])
    novel_cls = np.load('/m/tiangy/embeddings/bert/pascal_split/split/novel_cls.npy')

    # novel_hard_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_hard.npy')
    # novel_medium_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_medium.npy')
    # novel_easy_cls = np.load('/scratch/tiangy/bert/model_split/split_2/novel_easy.npy')

    class_all = np.concatenate([seen_cls,novel_cls])
    know_class = seen_cls

    list_ = csv.reader(open('/m/tiangy/embeddings/bert/pascal_split/correspondence.csv'))
    pre_all = []
    gt_all= []
    for line in list_:
        score_now = np.zeros(num_class)
        gt_cls = np.zeros(num_class)
        name_ = line[1].split('/')[-1].replace('.jpg','')
        name = ''.join(line[1].split('/')[-1].replace('.jpg','').split('_'))
        if not '00'+name+'_'+'19' in cls_score:
            continue
        for cls in class_all:
            score_now[cls] = cls_score['00'+name+'_'+str(cls)]
            flag = 0
            for x in data:
                if name == x[0]:
                    gt_cls[cls] =  cls in x[2]
                    flag = 1
            if flag ==0:
                print('not found')
        pre_all.append(score_now)
        gt_all.append(gt_cls)
    pre_all,gt_all = np.stack(pre_all).transpose(), np.stack(gt_all).transpose()
    print(pre_all.shape)

    AP_all = np.zeros(num_class)
    random_all = np.zeros(num_class)
    for i in class_all:
        pre = pre_all[i]
        g = gt_all[i]
        AP,random = binary_mAP(pre,g)

        AP_all[i] = AP
        random_all[i] = random
        #print(all_labels[i],' ',AP)
    #np.save('/scratch/tiangy/bert/model_split/split_2/class_rank.npy',np.argsort(AP_all))
    for i in np.argsort(AP_all):
        print(tokenizer.tokenize(all_labels[i]),AP_all[i],random_all[i])
        if AP_all[i]<random_all[i]:
            print('worse than random :',all_labels[i] )

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
    #mAP_2(args.score_file)
    #mAP_insert_caption_level(args.score_file)
    #mAP_insert(args.score_file)
    #mAP_insert_caption_level_notmerge(args.score_file)
    #cal_AP(args.score_file)
    #mAP_pascal(args.score_file)
    #mAP_append_local(args.score_file)

    #cal_IOU_soft_en(args.score_file,'logs/test_multi_soft_div_withpixel_multilayer_consist/scale1/cls_score.pkl',scale = 0.5)
    #cal_IOU_soft(args.score_file,use_softmax = 10)
    #mAP_append(args.score_file)
    mAP_append_local(args.score_file,softmax_scale = 10)
    #mAP_append_local_en(file = args.score_file,file2 = args.score_file2,scale=0.8,softmax_scale = 10)
    


