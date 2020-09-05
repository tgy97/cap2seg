from pytorch_pretrained_bert.modeling import BertModel,BertEmbeddings,BertEncoder,BertPooler,BertPreTrainedModel
import torch
from IPython import embed

import torch.nn as nn
import torch.nn.functional as F


from IPython.display import display
class FocalLossWithLogits(nn.Module):

    def __init__(self, gamma = 2, alpha = 0.1):
        super(FocalLossWithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, pred, target):
        a = 1 / (1 + torch.exp(-pred))
        t = target
        p1 = t * torch.log(a) * (1 - self.alpha) * torch.pow((1-a),self.gamma)
        p2 = (1-t) * torch.log(1-a) * self.alpha * torch.pow(a,self.gamma)
        loss = -(p1 + p2)
        return loss.mean()

class FocalLossWithLogits_NoGamma(nn.Module):

    def __init__(self, alpha = 0.1):
        super(FocalLossWithLogits_NoGamma, self).__init__()
        self.alpha = alpha


    def forward(self, pred, target):
        a = 1 / (1 + torch.exp(-pred))
        t = target
        p1 = t * torch.log(a) * (1 - self.alpha)
        p2 = (1-t) * torch.log(1-a) * self.alpha
        loss = -(p1 + p2)
        return loss.mean()

class soft_Mse(nn.Module):
    def __init__(self):
        super(soft_Mse,self).__init__()

    def forward(self, pred_loc, target_loc, label_mask):
        x1 = F.softmax(target_loc,dim = 1) * label_mask
        x2 = F.softmax(pred_loc,dim = 1)* label_mask
        return ((x1-x2)**2).mean()

class BertModelLowLevel(BertModel):

    def __init__(self, config):
        super(BertModelLowLevel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.layers = 1

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[self.layers]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[self.layers]
        return pooled_output

class BertModelLowLevel_CLA(BertModel):

    def __init__(self, config, num_labels):
        super(BertModelLowLevel_CLA, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.layers = 0

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)

        sequence_output = encoded_layers[self.layers]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.classifier(pooled_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[self.layers]
        return pooled_output


class BertForSequenceClassification_Multi(BertPreTrainedModel):

    def __init__(self, config, use_lowlevel = False, use_logits = False):
        super(BertForSequenceClassification_Multi, self).__init__(config)
        num_labels = 1
        num_location = 2 # mean, len/2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.localizer = nn.Linear(config.hidden_size, num_location)
        self.use_logits = use_logits
        self.use_lowlevel = use_lowlevel
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        x, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        #pooled_output = self.dropout(x[2][:,0])
        #pooled_output = self.dropout(pooled_output)
        #pooler_output = self.dropout(pooled_output + x[2][:,0])#+ x[6][:,0])
        if self.use_lowlevel:
            pooled_output = pooled_output + x[2][:,0]

        location = self.localizer(pooled_output)

        if self.use_logits:
            location = 1 / (1 + torch.exp(-location))

        pooler_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)


        return logits,location


class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        target = F.softmax(target, dim = 1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss

class softCrossEntropy_div(nn.Module):
    def __init__(self):
        super(softCrossEntropy_div, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        target = target/(target.sum(1).view(-1,1)+1e-6)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss
class BertForSequenceClassification_Multi_Soft(BertPreTrainedModel):
    def __init__(self, config, use_location_k = False, use_lowlevel = False):
        super(BertForSequenceClassification_Multi_Soft, self).__init__(config)
        num_labels = 1
        num_location = 3 
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.localizer = nn.Linear(config.hidden_size, num_location)
        if use_location_k:
            from pytorch_pretrained_bert.modeling import BertLayerNorm
            inter_size = 5
            self.trans = nn.Linear(config.hidden_size, inter_size)
            self.classifier = nn.Linear(inter_size + num_location, num_labels)
            self.localizer = nn.Linear(inter_size + num_location, num_location)
            self.LayerNorm = BertLayerNorm(inter_size + num_location, eps=1e-12)
        self.use_lowlevel = use_lowlevel
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, location_k = None):
        x, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        #pooled_output = self.dropout(x[2][:,0])
        #pooled_output = self.dropout(pooled_output)
        #pooler_output = self.dropout(pooled_output + x[2][:,0])#+ x[6][:,0])
        if self.use_lowlevel:
            pooled_output = pooled_output + x[2][:,0]
            #pooled_output = x[2][:,0]

        pooler_output = self.dropout(pooled_output)
        if not location_k is None:
            pooled_output = self.trans(pooled_output)

            location_k = location_k / (location_k.sum(1, keepdim = True) + 1e-12)
            pooled_output = torch.cat((pooled_output,location_k),dim=1)
            pooled_output = self.LayerNorm(pooled_output)

        logits = self.classifier(pooled_output)
        location = self.localizer(pooled_output)


        return logits,location

class MLP(nn.Module):
    def __init__(self,emb_len,soft_num,inter_len):
        super(MLP, self).__init__()
        self.trans_emb = nn.Sequential(
            nn.Linear(emb_len*2, inter_len),
            nn.ReLU())
        self.trans_loc = nn.Sequential(
            nn.Linear(soft_num, inter_len),
            nn.ReLU())
        self.trans_output = nn.Sequential(
            nn.Linear(inter_len*2, soft_num))

    def forward(self, emb_a, emb_b, location_k):
        ''' b, 200'''
        ''' b, 200'''
        ''' b, soft_num '''
        location_k = location_k / (location_k.sum(1, keepdim = True) + 1e-12)

        emb_all = torch.cat([emb_a,emb_b],axis = 1)
        inter_emb = self.trans_emb(emb_all)
        inter_loc = self.trans_loc(location_k)
        inter = torch.cat([inter_emb,inter_loc],axis = 1)
        output = self.trans_output(inter)

        return output


class MLP_pixel(nn.Module):
    def __init__(self,emb,emb_size = 600, inter_len = 100, device = None, multi_layer = True):
        '''position_no_relu positive :)'''
        super(MLP_pixel, self).__init__()
        self.emb = torch.Tensor(emb)
        if not device is None:
            self.emb = self.emb.to(device)
        position_input = 4

        if multi_layer:
            self.trans_emb = nn.Sequential(
                nn.Linear(emb_size*2, inter_len),
                nn.BatchNorm1d(inter_len),
                nn.LeakyReLU(),
                nn.Linear(inter_len, inter_len),
                nn.BatchNorm1d(inter_len),
                nn.LeakyReLU(),
                nn.Linear(inter_len, inter_len),
                nn.BatchNorm1d(inter_len),
                nn.LeakyReLU())

            self.trans_position = nn.Sequential(
                nn.Linear(position_input, inter_len),
                nn.BatchNorm1d(inter_len),
                nn.LeakyReLU(),
                nn.Linear(inter_len, inter_len),
                nn.BatchNorm1d(inter_len),
                nn.LeakyReLU(),
                nn.Linear(inter_len, inter_len),
                nn.BatchNorm1d(inter_len),
                nn.LeakyReLU())

            self.trans_output = nn.Sequential(
                nn.Linear(inter_len*2, inter_len*2),
                nn.BatchNorm1d(inter_len*2),
                nn.LeakyReLU(),
                nn.Linear(inter_len*2, inter_len*2),
                nn.BatchNorm1d(inter_len*2),
                nn.LeakyReLU(),
                nn.Linear(inter_len*2, 1))
        else:
            self.trans_emb = nn.Sequential(
                nn.Linear(emb_size*2, inter_len),
                nn.LeakyReLU())
            self.trans_position = nn.Sequential(
                nn.Linear(position_input, inter_len),
                nn.LeakyReLU())
            self.trans_output = nn.Sequential(
                nn.Linear(inter_len*2, 1))

    def forward(self, cls_a,cls_b,position_a,position_b):
        ''' b, '''
        ''' b, 2'''
        ''' b, soft_num '''
        emb_all = torch.cat([self.emb[cls_a], self.emb[cls_b]],axis = 1)
        position_all = torch.cat([position_a,position_b],axis = 1)
        emb_inter = self.trans_emb(emb_all)
        position_inter = self.trans_position(position_all)
        output = self.trans_output(torch.cat([emb_inter,position_inter],axis = 1))

        return output


if __name__ == '__main__':
    import numpy as np
    model = MLP_pixel(np.random.rand(10,600),600,100)

    cls_a = torch.randint(10,(3,))
    cls_b = torch.randint(10,(3,))
    position_a = torch.randn(3,2)
    position_b = torch.randn(3,2)
    output = model(cls_a,cls_b,position_a,position_b)
    print(model)
    print(output.shape)
    # c = softCrossEntropy_div()
    # a=torch.zeros(4,3)
    # a[0][0] = 1
    # a.requires_grad = True
    # b=torch.zeros(4,3)
    # b[0][1] = 1
    # c(a,b).backward()
    # #b[0][0] = 0
    # print(a.grad)