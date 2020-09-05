import gensim
import pickle	
import io
import json
import os
import numpy as np
import cv2

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from IPython import embed








def gen_binary_multi_soft_data_with_pixel():
	soft_num = 3
	soft_t = np.array(range(soft_num)) / soft_num
	soft_t = np.append(soft_t,1)

	#mode = ['train','val']
	mode = ['val']


	for m in mode:
		output_ = []
		out_file = open('./data/binary_all_multi_soft_withpixel/{}.pkl'.format(m),'wb')
		im_dir = '/m/tiangy/coco_stuff_anno/{}2017/'.format(m)
		file = json.load(open('/m/data/mscoco/annotations/captions_{}2017.json'.format(m)))
		for i_,f in enumerate(file['annotations']):
			caption = f['caption']
			im = cv2.imread(im_dir+ '{0:012d}.png'.format(f['image_id']), cv2.IMREAD_GRAYSCALE)
			label  = np.setdiff1d(np.unique(im.flatten()),[255] )
			h = im.shape[0]
			w = im.shape[1]
			loc ={}
			pixel_loc = {}
			for cls in label:
				pixel_loc_cls = []
				pixel_ind = np.where(im==cls)
				all_pixel = pixel_ind[0] / h
				all_pixel_w = pixel_ind[1] / w
				cls_soft = np.zeros(soft_num)
				for i in range(soft_num):
					cls_soft[i] = (all_pixel<soft_t[i+1]).sum() - (all_pixel<soft_t[i]).sum()
				assert all_pixel.size == cls_soft.sum()
				loc[cls] = cls_soft

				p_size = np.array(range(all_pixel.size))
				choiced = np.random.choice(p_size,20)
				pixel_loc[cls] = np.array([all_pixel[choiced], all_pixel_w[choiced]]).transpose() # 20 x 2		

				#loc[cls] = [all_pixel.mean(),(all_pixel.max() - all_pixel.min()) / 2]

			output_.append(['{0:012d}'.format(f['image_id']),caption,label,loc,pixel_loc])
			if i_ % 1000 == 0:
				print(i_)
		pickle.dump(output_,out_file)
if __name__ == "__main__":

    gen_binary_multi_soft_data_with_pixel()