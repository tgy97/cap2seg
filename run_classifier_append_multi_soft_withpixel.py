# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import  BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification,BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from IPython import embed
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
import pickle
from tensorboardX import SummaryWriter
from IPython import embed

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, class_id, text_a, text_b=None, label=None,location = None,
        cls_a = None, cls_b = None, position_a = None, position_b = None, label_b = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.class_id = class_id
        #self.caption_id = caption_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.location = location
        self.cls_a = cls_a
        self.cls_b = cls_b
        self.position_a = position_a
        self.position_b = position_b
        self.label_b = label_b
class InputExample_pixel(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, class_id, 
        cls_a = None, cls_b = None, position_a = None, position_b = None, label_b = None):
        self.guid = guid
        self.class_id = class_id
        #self.caption_id = caption_id
        self.cls_a = cls_a
        self.cls_b = cls_b
        self.position_a = position_a
        self.position_b = position_b
        self.label_b = label_b

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, cocoid, class_id, location,
        cls_a = None, cls_b = None, position_a = None, position_b = None, label_b = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.cocoid = cocoid
        self.class_id = class_id
        #self.caption_id = caption_id
        self.location = location
        self.cls_a = cls_a
        self.cls_b = cls_b
        self.position_a = position_a
        self.position_b = position_b
        self.label_b = label_b

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

def insert_name(caption,name):
    #return name+ ' ' +caption
    return caption + ' ' + name

soft_num = 3
class COCOProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir, novel_class, label_map, label_names):
        """See base class."""
        return self._create_examples(
            pickle.load(open(os.path.join(data_dir, "val.pkl"),'rb')), "train",novel_class = novel_class, label_map = label_map, label_names = label_names)

    def get_dev_examples(self, data_dir, label_map, label_names, unseen_class = None):
        """See base class."""
        return self._create_examples(
            pickle.load(open(os.path.join(data_dir, "val.pkl"),'rb')), "dev",label_map = label_map, label_names = label_names,unseen_class = unseen_class)

    def get_dev_pixel_examples(self, data_dir, label_map, label_names, unseen_class = None):
        """See base class."""
        return self._create_examples(
            pickle.load(open(os.path.join(data_dir, "val.pkl"),'rb')), "dev_pixel",label_map = label_map, label_names = label_names,unseen_class = unseen_class)

    def get_dev_pixel_ac_examples(self, data_dir, label_map, label_names, unseen_class = None):
        """See base class."""
        return self._create_examples(
            pickle.load(open(os.path.join(data_dir, "val.pkl"),'rb')), "dev_pixel_ac",label_map = label_map, label_names = label_names,unseen_class = unseen_class)
    def get_labels(self):
        """See base class."""
        return 182

    def _create_examples(self, lines, set_type, novel_class = None, label_map = None, label_names = None, unseen_class = None, only_unseen = True):#, single_caption = True):
        """Creates examples for the training and dev sets."""
        examples = []
        novel_class_255 = np.append(novel_class,255)
        unseen_class_255 = np.append(unseen_class,255)
        dev_255 = np.array([255])
        all_cls = np.array(range(182))
        count = {}
        grid_size = soft_num
        t_position = []
        for x in range(grid_size):
            for y in range(grid_size):
                t_position.append(np.array([x/grid_size + 1/grid_size/2, y/grid_size + 1/grid_size/2]))

        for (i, line) in enumerate(lines):
            guid = line[0]
            # if single_caption:
            #     if guid in count:
            #         count[guid] = count[guid] + 1
            #     else:
            #         count[guid] = 0
            # caption_id = count[guid]
            text_a = line[1]
            label = line[2]
            pixel = line[4]
            if set_type=='train':
                label = np.setdiff1d(label,novel_class_255)
                if label.size <= 1:
                    continue
                for i in label:
                    #pb_index = np.array(range(pixel[i].size))
                    cls_a = np.random.choice(np.setdiff1d(label,i))
                    ind_a = np.random.choice(pixel[cls_a].shape[0])
                    ind_b = np.random.choice(pixel[i].shape[0])
                    examples.append(InputExample(guid=guid, class_id = i, text_a=text_a, text_b=label_names[i], label=np.array([1]), location = np.array(line[3][i]),
                        cls_a = cls_a, cls_b = i, position_a = pixel[cls_a][ind_a], position_b = pixel[i][ind_b], label_b = np.array([1])))
                neg = np.setdiff1d(all_cls,label)
                neg = np.setdiff1d(neg,novel_class_255)
                neg = np.random.choice(neg,len(label))
                #neg = np.random.choice(neg,len(label)*4)
                for i in neg:
                    cls_a = np.random.choice(label)
                    ind_a = np.random.choice(pixel[cls_a].shape[0])
                    #ind_b = np.random.choice(pixel[i].shape[0])
                    position_b = np.random.rand(2)
                    examples.append(InputExample(guid=guid, class_id = i, text_a=text_a, text_b=label_names[i], label=np.array([0]), location = np.zeros(soft_num),
                        cls_a = cls_a, cls_b = i, position_a = pixel[cls_a][ind_a], position_b = position_b, label_b = np.array([0])))
                #tmp  = np.zeros(label_map.size - novel_class.size)
                #tmp[label_map[label]] = 1
                #label = tmp
                
            elif set_type =='dev':
                #examples.append(InputExample(guid=guid, class_id = -1, caption_id = caption_id, text_a=text_a, text_b=None, label=None))
                #label = np.setdiff1d(label,dev_255)
                if only_unseen:
                    it = unseen_class
                else:
                    it = range(182)
                for i in it:
                    name = label_names[i]
                    #new_text_a = insert_name(text_a,name)      
                    #examples.append(InputExample(guid=guid, class_id = i, caption_id = caption_id, text_a = new_text_a, text_b=None, label=None))
                    examples.append(InputExample(guid=guid, class_id = i,  text_a = text_a, text_b=label_names[i], label=None))

            elif set_type == 'dev_pixel':
                if guid in count:
                    count[guid] = count[guid] + 1
                    continue
                else:
                    count[guid] = 0                

                label = np.setdiff1d(label,unseen_class_255)
                if only_unseen:
                    it = unseen_class
                    for i in label:
                        for j in it:
                            for p in range(pixel[i].shape[0]):
                                for pb in t_position:
                                    examples.append(InputExample_pixel(guid = guid, class_id = j,cls_a = i, cls_b = j, position_a = pixel[i][p], position_b = pb))
                else:
                    pass
                pass

            elif set_type == 'dev_pixel_ac':
                if guid in count:
                    count[guid] = count[guid] + 1
                    continue
                else:
                    count[guid] = 0                

                label = np.setdiff1d(label,unseen_class_255)
                if only_unseen:
                    it = np.intersect1d(line[2],unseen_class)
                    for j in it:
                        for pb in range(pixel[j].shape[0]):
                            for i in label:
                                for p in range(pixel[i].shape[0]):
                                    examples.append(InputExample_pixel(guid = guid, class_id = j,cls_a = i, cls_b = j, position_a = pixel[i][p], position_b = pixel[j][pb], label_b = np.array([1])))

                    #it_neg = np.random.choice(np.setdiff1d(unseen_class,it), it.shape[0])
                    it_neg = np.setdiff1d(unseen_class,it)
                    pixe_b_num = range(pixel[list(pixel)[0]].shape[0])
                    for j in it_neg:
                        for pb in pixe_b_num:
                            position_b = np.random.rand(2)
                            for i in label:
                                for p in range(pixel[i].shape[0]):
                                    examples.append(InputExample_pixel(guid = guid, class_id = j,cls_a = i, cls_b = j, position_a = pixel[i][p], position_b = position_b, label_b = np.array([0])))

                else:
                    pass
                pass

        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    features = []
    data_size =  len(examples)
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        #label_id = label_map[example.label]
        label_id = example.label
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("text: %s" % (example.text_a))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #logger.info("label: %s " % (str(example.label.nonzero())))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              cocoid=example.guid,
                              class_id = example.class_id,
                              location = example.location,
                              cls_a = example.cls_a, 
                              cls_b = example.cls_b, 
                              position_a = example.position_a, 
                              position_b = example.position_b, 
                              label_b = example.label_b
                              ))
                              #caption_id = example.caption_id))
        if ex_index % 100000 == 0:
            logger.info(ex_index/data_size)
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook
def cal_direct_emb(input_ids,input_mask,label_emb,model):
    # 768 x 182
    # b x seq

    emb = model.embeddings.word_embeddings(input_ids) # b x seq_len x 768
    emb = emb * (input_mask.unsqueeze(2))
    n = torch.matmul(emb,label_emb) # b x seq_len x n
    n = n.max(1)[0]
    return n
def search_max_match(input, input_mask,label_emb):
    # b x seq x 768
    # b seq
    # 768 x 182
    input = input * (input_mask.unsqueeze(2))
    n = torch.matmul(input,label_emb) # b x seq_len x n
    n = n.max(1)[0]
    return n    
def search_topk_match(input, input_mask,label_emb,k):
    # b x seq x 768
    # b seq
    # 768 x 182
    input = input * (input_mask.unsqueeze(2))
    n = torch.matmul(input,label_emb) # b x seq_len x n
    n = torch.topk(n,k = k,dim =1)[0].mean(1)
    return n    
def cur_scheduler(sorted_cls, epoch, all_epoch, policy = 'None'):
    num_class = 182
    if policy == 'None':
        masked_map = np.ones(num_class)
        base = num_class

    elif policy == 'step':
        masked_map = np.zeros(num_class)
        base_per = 0.4
        percent = base_per + (epoch/ all_epoch) * (1-base_per)
        base = int(percent* num_class)

        masked_map[sorted_cls[-base:]] = 1

    elif policy == 'step_inv':
        masked_map = np.zeros(num_class)
        base_per = 0.4
        percent = base_per + (epoch/ all_epoch) * (1-base_per)
        base = int(percent* num_class)

        masked_map[sorted_cls[0:base]] = 1

    elif policy == 'step_lastone':
        all_epoch = 3
        if epoch > all_epoch:
            epoch = all_epoch
        masked_map = np.zeros(num_class)
        base_per = 0.4
        percent = base_per + (epoch/ all_epoch) * (1-base_per)
        base = int(percent* num_class)

        masked_map[sorted_cls[-base:]] = 1
    return torch.Tensor(masked_map),base # 182
def get_loader(train_features,masked_map,args):
    masked_map = masked_map.numpy().astype(np.long)
    masked_map = masked_map.nonzero()[0]
    now_features = [f for f in train_features if f.class_id in masked_map]
    now_input_ids = torch.tensor([f.input_ids for f in now_features ], dtype=torch.long)
    now_input_mask = torch.tensor([f.input_mask for f in now_features ], dtype=torch.long)
    now_segment_ids = torch.tensor([f.segment_ids for f in now_features], dtype=torch.long)
    now_label_ids = torch.Tensor([f.label_id for f in now_features])
    now_class_ids = torch.tensor([f.class_id for f in now_features], dtype = torch.long)
    train_data = TensorDataset(now_input_ids, now_input_mask, now_segment_ids, now_label_ids, now_class_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader
def consistency_pixel_location(model_pixel,cls_a,position_a,cls_b,soft_num):
    if not hasattr(consistency_pixel_location,'target_position'):
        t_position = []
        grid_size = soft_num
        for x in range(grid_size):
            for y in range(grid_size):
                t_position.append(np.array([x/grid_size + 1/grid_size/2, y/grid_size + 1/grid_size/2]))
        consistency_pixel_location.target_position = torch.Tensor(t_position).to(cls_a.device)   ## 9x2
    t_p = consistency_pixel_location.target_position

    b = cls_a.size()[0]
    t_num,p = t_p.size()

    position_b = t_p.unsqueeze(0).expand((b,)+t_p.size()).contiguous() #### b x 9 x 2
    position_b = position_b.view(-1,p) #### b9 x 2
    cls_a = cls_a.unsqueeze(1).expand(b,t_num).contiguous().view(-1) ###b9 x 2
    cls_b = cls_b.unsqueeze(1).expand(b,t_num).contiguous().view(-1)
    position_a = position_a.unsqueeze(1).expand(b,t_num,p).contiguous().view(-1,p)### b9 x 2
    pred = model_pixel(cls_a,cls_b,position_a,position_b)
    pred = pred.view(b,soft_num,soft_num).mean(2) ####b x soft

    return pred
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,

                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--cls_dir",
                        default=None,
                        type=str,
                        required=True)

    parser.add_argument("--emb_file",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--tokenizer",
                        default='bert-base-uncased',
                        type=str)  
    parser.add_argument("--loss_type",
                        default='bce',
                        type=str) 
    parser.add_argument("--pixel_model",
                        default=None,
                        type=str)

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--bert_lowlevel",
                        default=False,
                        action='store_true',
                        help="lowlevel")
    parser.add_argument("--consistency_loss",
                        default=False,
                        action='store_true',
                        help="consistency loss")
    parser.add_argument("--consist_scale",
                        default=1,
                        type=float,
                        help="loss scale")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_pixel",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev pixel set.")
    parser.add_argument("--do_eval_pixel_ac",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev pixel set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=5,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=2311,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()
    try:
        novel_class = np.load(os.path.join(args.cls_dir,'novel_cls.npy'))
        seen_class = np.concatenate([np.load(os.path.join(args.cls_dir,'seen_cls.npy')),np.load(os.path.join(args.cls_dir,'val_cls.npy'))])
        all_label_class = np.concatenate([seen_class,novel_class])
    except:
        novel_class = np.load(os.path.join(args.cls_dir,'novel_cls.npy'))
        seen_class = np.load(os.path.join(args.cls_dir,'seen_cls.npy'))
        all_label_class = np.concatenate([seen_class,novel_class])       

    seen_class_num = seen_class.size
    all_class_num = novel_class.size + seen_class.size

    cls_map = np.zeros(all_class_num).astype(int)
    for i,c in enumerate(all_label_class):
        cls_map[c] = i
    #seen_class_fa = cls_map[seen_class]
    #novel_class_fa = cls_map[novel_class]
    logger.info("cls_map: "+str(cls_map))

    emb = pickle.load(open(args.emb_file,'rb'))
    seen_cls_emb = emb[seen_class]
    cls_emb = emb[all_label_class]
    novel_cls_emb = emb[novel_class]



    processors = {
        "coco": COCOProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_eval_pixel and not args.do_eval_pixel_ac:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        pass
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    #label_list = processor.get_labels()  ######### todo

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    train_examples = None
    num_train_steps = None if args.do_train else 100
    if args.do_train:
        all_labels  = np.genfromtxt('/scratch/tiangy/bert/model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
        train_examples = processor.get_train_examples(args.data_dir, novel_class, cls_map, all_labels)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    #model = BertForSequenceClassification.from_pretrained(args.bert_model, 1)
    from model import BertForSequenceClassification_Multi_Soft,MLP_pixel
    model_pixel = MLP_pixel(emb,emb_size = 600, inter_len = 100, device = device)
    if not args.pixel_model is None:
        state_dict = torch.load(args.pixel_model)
        model_pixel.load_state_dict(state_dict,strict = True)
        logger.info('Load pixel model from '+ args.pixel_model)

    model = BertForSequenceClassification_Multi_Soft.from_pretrained(args.bert_model,use_lowlevel = args.bert_lowlevel)
    if args.bert_lowlevel:
        logger.info("BERT Model use low level feature.")


    if args.consistency_loss:
        logger.info("Use consistency scale: "+str(args.consist_scale))
    if args.fp16:
        model.half()
        model_pixel.half()
    model.to(device)
    model_pixel.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        model_pixel = torch.nn.parallel.DistributedDataParallel(model_pixel, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model_pixel = torch.nn.DataParallel(model_pixel)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters()) + list(model_pixel.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    # for p in list(model.named_parameters()):
    #     if 'embedding' in p[0]:
    #         param_optimizer.remove(p)
    # logger.info('fix embedding')

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=num_train_steps)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr = args.learning_rate)

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples,  args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.Tensor([f.label_id for f in train_features])
        all_class_ids = torch.tensor([f.class_id for f in train_features], dtype = torch.long)
        all_location = torch.Tensor([f.location for f in train_features])
        all_cls_a = torch.tensor([f.cls_a for f in train_features], dtype = torch.long)
        all_cls_b = torch.tensor([f.cls_b for f in train_features], dtype = torch.long)
        all_position_a = torch.Tensor([f.position_a for f in train_features])
        all_position_b = torch.Tensor([f.position_b for f in train_features])
        all_label_b = torch.Tensor([f.label_b for f in train_features])
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_class_ids,all_location,
            all_cls_a, all_cls_b, all_position_a, all_position_b, all_label_b)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        model_pixel.train()
        train_emb = torch.Tensor(np.transpose(seen_cls_emb)).to(device)
        novel_emb = torch.Tensor(np.transpose(novel_cls_emb)).to(device)
        if args.loss_type == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss().to(device)
            logger.info("loss: BCE loss")
        elif args.loss_type =='focal':
            from model import FocalLossWithLogits
            criterion = FocalLossWithLogits(gamma=2,alpha=0.3).to(device)
            logger.info("loss: focal loss")
        elif args.loss_type == 'focal_wogamma':
            from model import FocalLossWithLogits_NoGamma
            criterion = FocalLossWithLogits_NoGamma().to(device)
            logger.info("loss: focal_wogamma loss")

        from model import softCrossEntropy,softCrossEntropy_div,soft_Mse
        #criterion_loc = softCrossEntropy()
        criterion_loc = softCrossEntropy_div()
        criterion_consist = soft_Mse()

        writer = SummaryWriter(args.output_dir+'/runs')
        it_num = 0
        epoch_index = 0
        used = cls_map[[  4,   9,  10,  11,  15,  18,  19,  21,  23,  26,  30,  35,  39,
                41,  42,  48,  51,  55,  58,  63,  72,  75,  77,  81,  82,  87,
                89,  90,  94, 100, 102, 107, 110, 113, 115, 117, 119, 124, 125,
                126, 133, 134, 136, 139, 141, 142, 145, 153, 154, 157, 166, 169,
                170, 172, 173, 174, 175, 176, 177, 178]]

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_index +=1
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            it = tqdm(train_dataloader, desc="Iteration")
            #mask_map, base = cur_scheduler(class_rank,epoch_index, args.num_train_epochs,policy='step_lastone') # 182
            #now_loader = get_loader(train_features,mask_map,args)
            #it = tqdm(now_loader, desc="Iteration")
            #mask_map = mask_map.to(device)
            #logger.info('visible class: ' + str(base))
            for step, batch in enumerate(it):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label, class_ab, location,cls_a,cls_b,position_a,position_b,label_b = batch
                # mask = mask_map[class_ab].view(-1).nonzero().view(-1)
                # input_ids = input_ids.index_select(0,mask)
                # input_mask = input_mask.index_select(0,mask)s
                # segment_ids = segment_ids.index_select(0,mask)
                # label = label.index_select(0,mask)

                # mask = mask_map[class_ab].view(-1,1) ##cur_1
                # logits = model(input_ids, segment_ids, input_mask)              
                # logits = logits*mask
                pixel_out = model_pixel(cls_a,cls_b,position_a,position_b)
                loss_pixel = criterion(pixel_out, label_b)
                
                logits,loc_pred = model(input_ids, segment_ids, input_mask)  
                if args.consistency_loss:
                    pixel_loc_pred = consistency_pixel_location(model_pixel,cls_a,position_a,cls_b,soft_num)
                    loss_consist = criterion_consist(pixel_loc_pred, loc_pred, label) * args.consist_scale
                else:
                    loss_consist = 0
                loc_pred = loc_pred* label

                #x1 = x[6][:,0,:]
                #x1 = x1 / (torch.sqrt((x1*x1).sum(1)).unsqueeze(1)+1e-8)
                #x2 = torch.matmul(x1,train_emb)
                
                #loss = criterion(x2[:,used],label[:,used])

                loss_cls = criterion(logits,label)
                loss_loc = criterion_loc(loc_pred, location)
                loss = loss_cls+loss_loc+loss_pixel+loss_consist

                #loss += 0.3*criterion(search_topk_match(x[0],input_mask,train_emb,3),label)
                #loss += 0.3*criterion(search_max_match(x[0],input_mask,train_emb),label)
                #loss += 0.3*criterion(search_max_match(x[1],input_mask,train_emb),label)

                #x3 = torch.matmul(x1,novel_emb)
                #loss += 0.01 * criterion(x3,torch.ones(label.shape[0],novel_class.size).to(device))

                #loss = criterion(x2[:,cls_map[115]], label[:,cls_map[115]])
                if n_gpu > 1:
                    #loss = loss.mean() # mean() to average on multi-gpu.
                    pass
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                # x[2].register_hook(save_grad('x_2'))
                # x[1].register_hook(save_grad('x_1'))
                # x[0].register_hook(save_grad('x_0'))
                # x1.register_hook(save_grad('x1'))
                # x2.register_hook(save_grad('x2'))
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    writer.add_scalar("train_loss", float(loss.item()), it_num )
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                            for param in model_pixel.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            model_pixel.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1
                it_num = it_num + 1
                if args.consistency_loss:
                    it.set_postfix(loss_cls = "%.3f" % float(loss_cls.item()) , loss_pixel = "%.3f" % float(loss_pixel.item()), loss_consist = "%.8f" % float(loss_consist.item()))
                else:
                    it.set_postfix(loss_cls = "%.3f" % float(loss_cls.item()) , loss_pixel = "%.3f" % float(loss_pixel.item()))

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(),os.path.join(args.output_dir,WEIGHTS_NAME))
        model_to_save.config.to_json_file(os.path.join(args.output_dir,CONFIG_NAME))

        model_to_save_pixel = model_pixel.module if hasattr(model_pixel, 'module') else model_pixel
        torch.save(model_to_save_pixel.state_dict(),os.path.join(args.output_dir,'pixel_model.bin'))

    if args.do_eval:
        all_labels  = np.genfromtxt('model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
        eval_examples = processor.get_dev_examples(args.data_dir,cls_map,all_labels, unseen_class = novel_class)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        out_list = [[],[],[],[]]
        out_file = open(args.output_dir+'/cls_score.pkl','wb')
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        #all_label_ids = torch.Tensor([f.label_id for f in eval_features])
        all_cocoid = torch.tensor([int(f.cocoid) for f in eval_features], dtype=torch.long)
        all_classid = torch.tensor([int(f.class_id) for f in eval_features], dtype=torch.long)
        #all_captionid = torch.tensor([int(f.caption_id) for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_cocoid,all_classid)#,all_captionid)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_emb = torch.Tensor(np.transpose(cls_emb)).to(device)
        for input_ids, input_mask, segment_ids, cocoid, class_id in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            #label_ids = label_ids.to(device)
            cocoid = cocoid.numpy() # b x 1
            class_id = class_id.numpy() # b x 1
            #caption_id = caption_id.numpy()

            with torch.no_grad():
                #x,logits = model(input_ids, segment_ids, input_mask)
                logits,loc_pred = model(input_ids, segment_ids, input_mask)
                #x1 = x[6][:,0,:]
                #logits = torch.matmul(x1,eval_emb)

                #logits = 1/ (1 + torch.exp(-logits))
                #logits += 0.3*(search_topk_match(x[0],input_mask,eval_emb,3))
                #logits += 0.3*(search_max_match(x[0],input_mask,eval_emb))
                #logits += 0.3*(search_max_match(x[1],input_mask,eval_emb))
                #logits += (0.3*cal_direct_emb(input_ids,input_mask,eval_emb,model))

            logits = logits.detach().cpu().numpy() # b x 1
            loc_pred = loc_pred.detach().cpu().numpy() # b x 2
            #logits = logits / 
            out_list[0].append(cocoid)
            out_list[1].append(class_id)
            out_list[2].append(logits)
            out_list[3].append(loc_pred)
            #out_list[3].append(caption_id)
            #embed()


            #label_ids = label_ids.to('cpu').numpy()
        ans = {}
        ans_loc = {}

        out_list[0] = np.concatenate(out_list[0])
        out_list[1] = np.concatenate(out_list[1])
        out_list[2] = np.concatenate(out_list[2])
        out_list[3] = np.concatenate(out_list[3])
        logger.info('evaluation complete')
        for i in range(out_list[0].size):
            #embed()
            #now_score = np.full(out_list[1][i].shape,-1)
            #now_score[all_label_class] = out_list[1][i]
            #id_ = '{0:012d}_{1}_{2}'.format(out_list[0][i],out_list[1][i],out_list[3][i])
            id_ = '{0:012d}_{1}'.format(out_list[0][i],out_list[1][i])
            if id_ in ans:
                ans[id_].append(out_list[2][i])
                ans_loc[id_].append(out_list[3][i])
            else:
                ans[id_] = [out_list[2][i]]
                ans_loc[id_] = [out_list[3][i]]
        for i in ans:
            ans[i] = np.stack(ans[i]).max(0)
            ans_loc[i] = np.stack(ans_loc[i]).mean(0) 
        logger.info('statistic complete')

        pickle.dump([ans,ans_loc],out_file)
        #     tmp_eval_accuracy = accuracy(logits, label_ids)

        #     eval_loss += tmp_eval_loss.mean().item()
        #     eval_accuracy += tmp_eval_accuracy

        #     nb_eval_examples += input_ids.size(0)
        #     nb_eval_steps += 1

        # eval_loss = eval_loss / nb_eval_steps
        # eval_accuracy = eval_accuracy / nb_eval_examples
        # if args.do_train:
        #     result = {'eval_loss': eval_loss,
        #               'eval_accuracy': eval_accuracy,
        #               'global_step': global_step,
        #               'loss': tr_loss/nb_tr_steps}
        # else:
        #     result = {'eval_loss': eval_loss,
        #               'eval_accuracy': eval_accuracy,
        #               'global_step': global_step}

        # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     logger.info("***** Eval results *****")
        #     for key in sorted(result.keys()):
        #         logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))
    if args.do_eval_pixel:
        all_labels  = np.genfromtxt('model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
        eval_examples = processor.get_dev_pixel_examples(args.data_dir,cls_map,all_labels, unseen_class = novel_class)
        eval_features = eval_examples
        # eval_features = convert_examples_to_features(
        #     eval_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        batch_size = 8192*8
        logger.info("  Batch size = %d", batch_size)
        out_list = [[],[],[],[]]
        out_file = open(args.output_dir+'/cls_score_pixel.pkl','wb')
        #all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        #all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        #all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        #all_label_ids = torch.Tensor([f.label_id for f in eval_features])
        all_cocoid = torch.tensor([int(f.guid) for f in eval_features], dtype=torch.long)
        all_classid = torch.tensor([int(f.class_id) for f in eval_features], dtype=torch.long)
        all_cls_a = torch.tensor([f.cls_a for f in eval_features], dtype = torch.long)
        all_cls_b = torch.tensor([f.cls_b for f in eval_features], dtype = torch.long)
        all_position_a = torch.Tensor([f.position_a for f in eval_features])
        all_position_b = torch.Tensor([f.position_b for f in eval_features])
        # eval_data = TensorDataset(all_cocoid,all_classid,all_cls_a,all_cls_b,all_position_a,all_position_b)#,all_captionid)
        # if args.local_rank == -1:
        #     eval_sampler = SequentialSampler(eval_data)
        # else:
        #     eval_sampler = DistributedSampler(eval_data)
        # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=10240)#args.eval_batch_size)

        model_pixel.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_emb = torch.Tensor(np.transpose(cls_emb)).to(device)
        count = 0
        logger.info("start evaluation")
        sample_num = len(eval_examples)
        it_num = int(sample_num/batch_size) + 1
        for batch_ind in range(it_num):
            batch_ind  = batch_ind * batch_size
            if batch_ind + batch_size > sample_num:
                batch_end = -1
            else:
                batch_end = batch_ind + batch_size
            cocoid = all_cocoid[batch_ind:batch_end]
            class_id = all_classid[batch_ind:batch_end]
            cls_a = all_cls_a[batch_ind:batch_end]
            cls_b = all_cls_b[batch_ind:batch_end]
            position_a = all_position_a[batch_ind:batch_end]
            position_b = all_position_b[batch_ind:batch_end]
        #for cocoid, class_id, cls_a, cls_b, position_a, position_b in eval_dataloader:
            if count % 100 == 0:
                print(count)

            count += 1

            # input_ids = input_ids.to(device)
            # input_mask = input_mask.to(device)
            # segment_ids = segment_ids.to(device)
            #label_ids = label_ids.to(device)
            cocoid = cocoid.numpy() # b x 1
            class_id = class_id.numpy() # b x 1
            cls_a = cls_a.to(device)
            cls_b = cls_b.to(device)
            position_a = position_a.to(device)
            position_b = position_b.to(device)
            #caption_id = caption_id.numpy()
            #embed()
            with torch.no_grad():
                #x,logits = model(input_ids, segment_ids, input_mask)
                pixel_pred = model_pixel(cls_a, cls_b, position_a, position_b)
                #x1 = x[6][:,0,:]
                #logits = torch.matmul(x1,eval_emb)

                #logits = 1/ (1 + torch.exp(-logits))
                #logits += 0.3*(search_topk_match(x[0],input_mask,eval_emb,3))
                #logits += 0.3*(search_max_match(x[0],input_mask,eval_emb))
                #logits += 0.3*(search_max_match(x[1],input_mask,eval_emb))
                #logits += (0.3*cal_direct_emb(input_ids,input_mask,eval_emb,model))

            pixel_pred = pixel_pred.detach().cpu().numpy() # b x 1
            position = position_b.detach().cpu().numpy()
            #logits = logits / 
            out_list[0].append(cocoid)
            out_list[1].append(class_id)
            out_list[2].append(pixel_pred)
            out_list[3].append(position[:,0]) # h

        ans = {}
        ans_loc = {}
        out_list[0] = np.concatenate(out_list[0])
        out_list[1] = np.concatenate(out_list[1])
        out_list[2] = np.concatenate(out_list[2])
        out_list[3] = np.concatenate(out_list[3])
        #embed()
        logger.info('evaluation complete')

        soft_t = np.array(range(soft_num)) / soft_num
        soft_t = np.append(soft_t,1)
        sta_function = 'None'
        logger.info('Result using function: ' + sta_function)
        for i in range(out_list[0].size):
            id_ = '{0:012d}_{1}'.format(out_list[0][i],out_list[1][i])
            v_index = (out_list[3][i] > soft_t).sum() - 1
            if not id_ in ans_loc:
                ans_loc[id_] = [np.zeros(soft_num),np.zeros(soft_num)]
            if sta_function == 'sigmoid':
                ans_loc[id_][0][v_index] += 1/(1 + np.exp(-out_list[2][i]))
            elif sta_function == 'None':
                ans_loc[id_][0][v_index] += out_list[2][i]
            else:
                raiseNotImplementedError("Not Implemented")
            ans_loc[id_][1][v_index] += 1
        for i in ans_loc:
            #assert ans_loc[i][1][0] == ans_loc[i][1][-1]
            ans_loc[i] = ans_loc[i][0] / ans_loc[i][1]
        logger.info('statistic complete')

        pickle.dump([ans,ans_loc],out_file)

    if args.do_eval_pixel_ac:
        all_labels  = np.genfromtxt('model_split/labels_2.txt', delimiter='\t', usecols=1, dtype='str') #abosolute
        eval_examples = processor.get_dev_pixel_ac_examples(args.data_dir,cls_map,all_labels, unseen_class = novel_class)
        eval_features = eval_examples
        # eval_features = convert_examples_to_features(
        #     eval_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        batch_size = 8192*8
        logger.info("  Batch size = %d", batch_size)
        out_list = [[],[],[],[],[]]
        out_file = open(args.output_dir+'/cls_score_pixel_ac.pkl','wb')
        #all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        #all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        #all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        #all_label_ids = torch.Tensor([f.label_id for f in eval_features])
        all_cocoid = torch.tensor([int(f.guid) for f in eval_features], dtype=torch.long)
        all_classid = torch.tensor([int(f.class_id) for f in eval_features], dtype=torch.long)
        all_cls_a = torch.tensor([f.cls_a for f in eval_features], dtype = torch.long)
        all_cls_b = torch.tensor([f.cls_b for f in eval_features], dtype = torch.long)
        all_position_a = torch.Tensor([f.position_a for f in eval_features])
        all_position_b = torch.Tensor([f.position_b for f in eval_features])
        all_label_b = torch.Tensor([f.label_b for f in eval_features])
        # eval_data = TensorDataset(all_cocoid,all_classid,all_cls_a,all_cls_b,all_position_a,all_position_b)#,all_captionid)
        # if args.local_rank == -1:
        #     eval_sampler = SequentialSampler(eval_data)
        # else:
        #     eval_sampler = DistributedSampler(eval_data)
        # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=10240)#args.eval_batch_size)

        model_pixel.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_emb = torch.Tensor(np.transpose(cls_emb)).to(device)
        count = 0
        logger.info("start evaluation")
        sample_num = len(eval_examples)
        it_num = int(sample_num/batch_size) + 1
        for batch_ind in range(it_num):
            batch_ind  = batch_ind * batch_size
            if batch_ind + batch_size > sample_num:
                batch_end = -1
            else:
                batch_end = batch_ind + batch_size
            cocoid = all_cocoid[batch_ind:batch_end]
            class_id = all_classid[batch_ind:batch_end]
            cls_a = all_cls_a[batch_ind:batch_end]
            cls_b = all_cls_b[batch_ind:batch_end]
            position_a = all_position_a[batch_ind:batch_end]
            position_b = all_position_b[batch_ind:batch_end]
            label_b = all_label_b[batch_ind:batch_end]
        #for cocoid, class_id, cls_a, cls_b, position_a, position_b in eval_dataloader:
            if count % 100 == 0:
                print(count)

            count += 1

            # input_ids = input_ids.to(device)
            # input_mask = input_mask.to(device)
            # segment_ids = segment_ids.to(device)
            #label_ids = label_ids.to(device)
            cocoid = cocoid.numpy() # b x 1
            class_id = class_id.numpy() # b x 1
            label_b = label_b.numpy()
            cls_a = cls_a.to(device)
            cls_b = cls_b.to(device)
            position_a = position_a.to(device)
            position_b = position_b.to(device)
            #caption_id = caption_id.numpy()
            #embed()
            with torch.no_grad():
                #x,logits = model(input_ids, segment_ids, input_mask)
                pixel_pred = model_pixel(cls_a, cls_b, position_a, position_b)
                #x1 = x[6][:,0,:]
                #logits = torch.matmul(x1,eval_emb)

                #logits = 1/ (1 + torch.exp(-logits))
                #logits += 0.3*(search_topk_match(x[0],input_mask,eval_emb,3))
                #logits += 0.3*(search_max_match(x[0],input_mask,eval_emb))
                #logits += 0.3*(search_max_match(x[1],input_mask,eval_emb))
                #logits += (0.3*cal_direct_emb(input_ids,input_mask,eval_emb,model))

            pixel_pred = pixel_pred.detach().cpu().numpy() # b x 1
            position = position_b.detach().cpu().numpy()
            #logits = logits / 
            out_list[0].append(cocoid)
            out_list[1].append(class_id)
            out_list[2].append(pixel_pred)
            out_list[3].append(position) 
            out_list[4].append(label_b)

        ans_pred = []
        ans_gt = []
        out_list[0] = np.concatenate(out_list[0])
        out_list[1] = np.concatenate(out_list[1])
        out_list[2] = np.concatenate(out_list[2]) # pred
        out_list[3] = np.concatenate(out_list[3]) # position
        out_list[4] = np.concatenate(out_list[4]) # gt
        #embed()
        logger.info('evaluation complete')
        def compare(a,b):
            return (np.abs(a[0]-b[0]) < 1e-6) and (np.abs(a[1]-b[1]) < 1e-6)
        ''' init'''
        position_last = out_list[3][0]
        score = []
        gt_last = out_list[4][0]
        for i in range(out_list[0].size):
            position_now = out_list[3][i]
            if compare(position_now, position_last): #equal
                pred = out_list[2][i]
                score.append(pred)

                assert out_list[4][i] == gt_last 
            else:
                ans_pred.append(np.mean(score))
                ans_gt.append(out_list[4][i])
                score = []
                position_last = position_now
            gt_last = out_list[4][i] 
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
        AP, random_ans = binary_mAP(np.array(ans_pred), np.array(ans_gt).squeeze())
        print(AP, random_ans)
        pickle.dump([AP, random_ans], out_file)

        # soft_t = np.array(range(soft_num)) / soft_num
        # soft_t = np.append(soft_t,1)
        # sta_function = 'None'
        # logger.info('Result using function: ' + sta_function)
        # for i in range(out_list[0].size):
        #     id_ = '{0:012d}_{1}'.format(out_list[0][i],out_list[1][i])
        #     v_index = (out_list[3][i] > soft_t).sum() - 1
        #     if not id_ in ans_loc:
        #         ans_loc[id_] = [np.zeros(soft_num),np.zeros(soft_num)]
        #     if sta_function == 'sigmoid':
        #         ans_loc[id_][0][v_index] += 1/(1 + np.exp(-out_list[2][i]))
        #     elif sta_function == 'None':
        #         ans_loc[id_][0][v_index] += out_list[2][i]
        #     else:
        #         raiseNotImplementedError("Not Implemented")
        #     ans_loc[id_][1][v_index] += 1
        # for i in ans_loc:
        #     #assert ans_loc[i][1][0] == ans_loc[i][1][-1]
        #     ans_loc[i] = ans_loc[i][0] / ans_loc[i][1]
        # logger.info('statistic complete')

        # pickle.dump([ans,ans_loc],out_file)

if __name__ == "__main__":
    main()