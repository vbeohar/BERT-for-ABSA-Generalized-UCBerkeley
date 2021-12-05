# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import os
import logging
import argparse
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tokenization import BertTokenizer
# from modeling import BertModel, BertPreTrainedModel, BertLayer, BertPooler
from modeling import BertLayer, BertPooler
from transformers import RobertaTokenizer, RobertaModel, RobertaPreTrainedModel
from optimization import BertAdam

import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
import modelconfig
from math import ceil


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
# device = torch.device("cpu")

class GRoIE(nn.Module):
    def __init__(self, count, config, num_labels):
        super(GRoIE, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pooler = BertPooler(config)
        self.pre_layers = torch.nn.ModuleList()
        self.loss_fct = torch.nn.ModuleList()
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        for i in range(count):
            self.pre_layers.append(BertLayer(config))
            self.loss_fct.append(torch.nn.CrossEntropyLoss(ignore_index=-1))

    def forward(self, layers, attention_mask, labels):
        losses = []
        logitses = []
        for i in range(self.count):
            layer = self.pre_layers[i](layers[-i-1], attention_mask)
            layer = self.pooler(layer)
            logits = self.classifier(layer)
            if labels is not None:
                loss = self.loss_fct[i](logits.view(-1, self.num_labels), labels.view(-1))
                losses.append(loss)
            logitses.append(logits)
        if labels is not None:
            total_loss = torch.sum(torch.stack(losses), dim=0)
        else:
            total_loss = torch.Tensor(0)
        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return total_loss, avg_logits

class HSUM(nn.Module):
    def __init__(self, count, config, num_labels):
        super(HSUM, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pre_layers = torch.nn.ModuleList()
        self.loss_fct = torch.nn.ModuleList()
        self.pooler = BertPooler(config)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        for i in range(count):
            self.pre_layers.append(BertLayer(config))
            self.loss_fct.append(torch.nn.CrossEntropyLoss(ignore_index=-1))

    def forward(self, layers, attention_mask, labels):
        losses = []
        logitses = []
        output = torch.zeros_like(layers[0])
        total_loss = torch.Tensor(0)
        for i in range(self.count):
            output = output + layers[-i-1]
            output = self.pre_layers[i](output, attention_mask)
            out = self.pooler(output)
            logits = self.classifier(out)
            if labels is not None:
                loss = self.loss_fct[i](logits.view(-1, self.num_labels), labels.view(-1))
                losses.append(loss)
            logitses.append(logits)
        if labels is not None:
            total_loss = torch.sum(torch.stack(losses), dim=0)
        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return total_loss, avg_logits


class BertForABSA(RobertaPreTrainedModel):
    def __init__(self, config, num_labels=3):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.roberta = RobertaModel(config)
        self.groie = GRoIE(4, config, num_labels)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # model_outputs = self.roberta(input_ids, token_type_ids=token_type_ids, 
        #                                                 attention_mask=attention_mask, 
        #                                                 output_hidden_states=True,
        #                                                 output_attentions=True)
        model_outputs = self.roberta(input_ids, attention_mask=attention_mask, 
                                                        output_hidden_states=True,
                                                        output_attentions=True)
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
        
        loss, logits = self.groie(model_outputs.hidden_states, extended_attention_mask, labels)
        if labels is not None:
            return loss
        else:
            return logits

def train(args):

    processor = data_utils.AscProcessor()
    label_list = processor.get_labels()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs
    
    train_features = data_utils.convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, "asc")
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    # logger.info("    Examples look like %s", str(train_features[0].input_ids))
    # logger.info(f"    Lables look like {train_features[0].label_id}")
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    
    # logger.info(f"  Maximum input_id is {max([max(f.input_ids) for f in train_features])}")
    # logger.info(f"  Minimum input_id is {min([min(f.input_ids) for f in train_features])}")
    
    bad_inputs = []
    for i in train_features:
        if len(i.input_ids) != 100:
            bad_inputs.append((i, i.input_ids))
    
    if len(bad_inputs) > 0:
        raise IndexError(f" Bad Inputs recieved! See the following: \n{bad_inputs}")
    
    logger.info(f"  Maximum label is {max([f.label_id for f in train_features])}")
    logger.info(f"  Minimum label is {min([f.label_id for f in train_features])}")
    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    
    #>>>>> validation
    if args.do_valid:
        valid_examples = processor.get_dev_examples(args.data_dir)
        valid_features=data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "asc")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.train_batch_size)    

        best_valid_loss=float('inf')
        valid_losses=[]
    #<<<<< end of validation declaration

    model = BertForABSA.from_pretrained("roberta-base", num_labels=len(label_list))
    # logger.info("***** Preparing Model *****")
    # logger.info("  Sending model to device...")
    model.to(device)
    # Prepare optimizer
    # logger.info("  Preparing optimizer...")
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad==True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    global_step = 0
    # logger.info("***** Training Model *****")    
    # logger.info("  Initial Training...")
    model.train()

    for epoch in range(args.num_train_epochs):
        logger.info(f"  Training Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch
            # logger.info(f"  Model looks like {model}")
            # logger.info(f"  Model embeddings like {model.roberta.embeddings.word_embeddings.__dict__}")
            # logger.info(f"  Input Ids min {torch.min(input_ids)}")
            # logger.info(f"  Input Ids max {torch.max(input_ids)}")
            # logger.info(f"  Segment Ids min {torch.min(segment_ids)}")
            # logger.info(f"  Segment Ids max {torch.max(segment_ids)}")
            # logger.info(f"  Input Mask min {torch.min(input_mask)}")
            # logger.info(f"  Input Mask max {torch.max(input_mask)}")
            # logger.info(f"  Label Ids min {torch.min(label_ids)}")
            # logger.info(f"  Label Ids max {torch.max(label_ids)}")
            optimizer.zero_grad()
            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            loss.backward()
            
            lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            global_step += 1
        print("training loss: ", loss.item(), epoch+1)
        #>>>> perform validation at the end of each epoch.
        new_dirs = os.path.join(args.output_dir, str(epoch+1))
        os.mkdir(new_dirs)
        if args.do_valid:
            model.eval()
            with torch.no_grad():
                losses=[]
                valid_size=0
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                    input_ids, segment_ids, input_mask, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    losses.append(loss.data.item()*input_ids.size(0) )
                    valid_size+=input_ids.size(0)
                valid_loss=sum(losses)/valid_size
                logger.info("validation loss: %f, epoch: %d", valid_loss, epoch+1)
                valid_losses.append(valid_loss)
                torch.save(model, os.path.join(new_dirs, "model.pt"))
                test(args, new_dirs, dev_as_test=True)
                if epoch == args.num_train_epochs-1:
                    torch.save(model, os.path.join(args.output_dir, "model.pt"))
                    test(args, args.output_dir, dev_as_test=False)
                os.remove(os.path.join(new_dirs, "model.pt"))
            if valid_loss<best_valid_loss:
                best_valid_loss=valid_loss
            model.train()
    if args.do_valid:
        with open(os.path.join(args.output_dir, "valid.json"), "w") as fw:
            json.dump({"valid_losses": valid_losses}, fw)
    else:
        logger.info(f"saving model to {args.output_dir}")
        torch.save(model, os.path.join(args.output_dir, "model.pt") )


def test(args, new_dirs=None, dev_as_test=None):  # Load a trained model that you have fine-tuned (we assume evaluate on cpu)    
    processor = data_utils.AscProcessor()
    label_list = processor.get_labels()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    if dev_as_test:
        data_dir = os.path.join(args.data_dir, 'dev_as_test')
    else:
        data_dir = args.data_dir
    eval_examples = processor.get_test_examples(data_dir)
    eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, "asc")

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model = torch.load(os.path.join(new_dirs, "model.pt"))
    model.to(device)
    model.eval()
    
    full_logits=[]
    full_label_ids=[]
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, segment_ids, input_mask, label_ids = batch
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        full_logits.extend(logits.tolist())
        full_label_ids.extend(label_ids.tolist())

    output_eval_json = os.path.join(new_dirs, "predictions.json") 
    with open(output_eval_json, "w") as fw:
        json.dump({"logits": full_logits, "label_ids": full_label_ids}, fw)
    
    



def main():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default='roberta-base', type=str)

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir containing json files.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

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
    parser.add_argument("--do_valid",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--num_train_epochs",
                        default=6,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train(args)
    if args.do_eval:
        test(args)
            
if __name__=="__main__":
    main()