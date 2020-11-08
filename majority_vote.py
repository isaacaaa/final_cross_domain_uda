
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Finetuning multi-lingual models on XNLI (Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""


import argparse
import glob
import logging
import os
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
# from sklearn.metrics import confusion_matrix
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup
)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import *
#from transformers.data.metrics import acc_and_f1
#from transformers import xnli_output_modes as output_modes
#from transformers import xnli_processors as processors
from x_data_processor import appreview_processors as processors
from x_data_processor import appreview_output_modes as output_modes
from sklearn import metrics
MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
}
def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return basic_metrics(labels, preds)
def basic_metrics(y_true, y_pred):
    return {'Accuracy': metrics.accuracy_score(y_true, y_pred),
            'Precision': metrics.precision_score(y_true, y_pred, average='macro'),
            'Recall': metrics.recall_score(y_true, y_pred, average='macro'),
            'Macro-F1': metrics.f1_score(y_true, y_pred, average='macro'),
            'Micro-F1': metrics.f1_score(y_true, y_pred, average='micro'),
            'ConfMat': confusion_matrix(y_true, y_pred)}
def confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, range(len(processors['appreview']().get_labels())))
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if task=='appreview' and not evaluate:
        processor = processors['appreview']()
        unsupprocessor = processors['unsup']()
        augprocessor = processors['aug']()
    if task=='appreview' and evaluate:
        processor = processors['appreview']()
        unsupprocessor = processors['unsup']()
    else:
        # print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        processor = processors[task]()

    output_mode = output_modes[task]

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    # print('22222222222222222222222222222222222222222222222222222222222')
    # print(label_list)
    if evaluate:

        unsupexamples1,unsupexamples2,unsupexamples3,unsupexamples4 = (
            unsupprocessor.get_train_examples(args.data_dir)
        )
        unsupexamples=unsupexamples1+unsupexamples2+unsupexamples3
        # print(examples1[0])
        # print(unsupexamples4[0])
        unsupfeatures = convert_examples_to_features(
            unsupexamples, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
        )
        unsupfeatures1 = convert_examples_to_features(
            unsupexamples1, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
        )
        unsupfeatures2 = convert_examples_to_features(
            unsupexamples2, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
        )
        unsupfeatures3 = convert_examples_to_features(
            unsupexamples3, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
        )
        unsupfeatures4 = convert_examples_to_features(
            unsupexamples4, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
        )
    else:
        examples1,examples2,examples3,examples4 = (
        processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
    )  
        features1 = convert_examples_to_features(
        examples1, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
    )
        features2 = convert_examples_to_features(
        examples2, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
    )
        features3 = convert_examples_to_features(
        examples3, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
    )
        features4 = convert_examples_to_features(
        examples4, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
    )   
        
        unsupexamples1,unsupexamples2,unsupexamples3,unsupexamples4 = (
            unsupprocessor.get_train_examples(args.data_dir)
        )
        # print(examples1[0])
        # print(unsupexamples4[0])
        unsupfeatures1 = convert_examples_to_features(
            unsupexamples1, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
        )
        unsupfeatures2 = convert_examples_to_features(
            unsupexamples2, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
        )
        unsupfeatures3 = convert_examples_to_features(
            unsupexamples3, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
        )
        unsupfeatures4 = convert_examples_to_features(
            unsupexamples4, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
        )
        augexamples1,augexamples2,augexamples3,augexamples4 = (
            augprocessor.get_train_examples(args.data_dir)
        )
        augfeatures1 = convert_examples_to_features(
            augexamples1, tokenizer, max_length=args.max_seq_length, label_list= label_list, output_mode=output_mode,
        )
        augfeatures2 = convert_examples_to_features(
            augexamples2, tokenizer, max_length=args.max_seq_length, label_list= label_list, output_mode=output_mode,
        )
        augfeatures3 = convert_examples_to_features(
            augexamples3, tokenizer, max_length=args.max_seq_length, label_list= label_list, output_mode=output_mode,
        )
        augfeatures4 = convert_examples_to_features(
            augexamples4, tokenizer, max_length=args.max_seq_length, label_list= label_list, output_mode=output_mode,
        )

    # source_features=features1+features2+features3+features4
    # source_unsup_features=unsupfeatures1+unsupfeatures2+unsupfeatures3+unsupfeatures4
    # source_aug_features=augfeatures1+augfeatures2+augfeatures3+augfeatures4
    # print(features)
    # if args.local_rank in [-1, 0]:
    #     logger.info("Saving features into cached file %s", cached_features_file)
    #     torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    # for index,features in enumerate(source_features):
        # if index ==0:
        # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    if not evaluate:
        all_input_ids1= torch.tensor([f.input_ids for f in features1], dtype=torch.long)
       
        all_attention_mask1 = torch.tensor([f.attention_mask for f in features1], dtype=torch.long)
        all_token_type_ids1 = torch.tensor([f.token_type_ids for f in features1], dtype=torch.long)

        all_input_ids2 = torch.tensor([f.input_ids for f in features2], dtype=torch.long)
        all_attention_mask2 = torch.tensor([f.attention_mask for f in features2], dtype=torch.long)
        all_token_type_ids2 = torch.tensor([f.token_type_ids for f in features2], dtype=torch.long)

        all_input_ids3 = torch.tensor([f.input_ids for f in features3], dtype=torch.long)
        all_attention_mask3 = torch.tensor([f.attention_mask for f in features3], dtype=torch.long)
        
        all_token_type_ids3 = torch.tensor([f.token_type_ids for f in features3], dtype=torch.long)

        all_input_ids4 = torch.tensor([f.input_ids for f in features4], dtype=torch.long)
        all_attention_mask4 = torch.tensor([f.attention_mask for f in features4], dtype=torch.long)
        all_token_type_ids4 = torch.tensor([f.token_type_ids for f in features4], dtype=torch.long)

        
        unsup_all_input_ids1 = torch.tensor([f.input_ids for f in unsupfeatures1], dtype=torch.long)
        unsup_all_attention_mask1 = torch.tensor([f.attention_mask for f in unsupfeatures1], dtype=torch.long)
        unsup_all_token_type_ids1 = torch.tensor([f.token_type_ids for f in unsupfeatures1], dtype=torch.long)

        unsup_all_input_ids2 = torch.tensor([f.input_ids for f in unsupfeatures2], dtype=torch.long)
        unsup_all_attention_mask2 = torch.tensor([f.attention_mask for f in unsupfeatures2], dtype=torch.long)
        unsup_all_token_type_ids2 = torch.tensor([f.token_type_ids for f in unsupfeatures2], dtype=torch.long)

        unsup_all_input_ids3 = torch.tensor([f.input_ids for f in unsupfeatures3], dtype=torch.long)
        unsup_all_attention_mask3 = torch.tensor([f.attention_mask for f in unsupfeatures3], dtype=torch.long)
        unsup_all_token_type_ids3 = torch.tensor([f.token_type_ids for f in unsupfeatures3], dtype=torch.long)

        unsup_all_input_ids4 = torch.tensor([f.input_ids for f in unsupfeatures4], dtype=torch.long)
        unsup_all_attention_mask4 = torch.tensor([f.attention_mask for f in unsupfeatures4], dtype=torch.long)
        unsup_all_token_type_ids4 = torch.tensor([f.token_type_ids for f in unsupfeatures4], dtype=torch.long)

        aug_all_input_ids1 = torch.tensor([f.input_ids for f in augfeatures1], dtype=torch.long)
        aug_all_attention_mask1 = torch.tensor([f.attention_mask for f in augfeatures1], dtype=torch.long)
        aug_all_token_type_ids1 = torch.tensor([f.token_type_ids for f in augfeatures1], dtype=torch.long)

        aug_all_input_ids2 = torch.tensor([f.input_ids for f in augfeatures2], dtype=torch.long)
        aug_all_attention_mask2 = torch.tensor([f.attention_mask for f in augfeatures2], dtype=torch.long)
        aug_all_token_type_ids2 = torch.tensor([f.token_type_ids for f in augfeatures2], dtype=torch.long)
    
        aug_all_input_ids3 = torch.tensor([f.input_ids for f in augfeatures3], dtype=torch.long)
        aug_all_attention_mask3 = torch.tensor([f.attention_mask for f in augfeatures3], dtype=torch.long)
        aug_all_token_type_ids3 = torch.tensor([f.token_type_ids for f in augfeatures3], dtype=torch.long)

        aug_all_input_ids4 = torch.tensor([f.input_ids for f in augfeatures4], dtype=torch.long)
        aug_all_attention_mask4 = torch.tensor([f.attention_mask for f in augfeatures4], dtype=torch.long)
        aug_all_token_type_ids4 = torch.tensor([f.token_type_ids for f in augfeatures4], dtype=torch.long)
    else:
        unsup_all_input_ids = torch.tensor([f.input_ids for f in unsupfeatures], dtype=torch.long)
        unsup_all_attention_mask = torch.tensor([f.attention_mask for f in unsupfeatures], dtype=torch.long)
        unsup_all_token_type_ids = torch.tensor([f.token_type_ids for f in unsupfeatures], dtype=torch.long)

        
        unsup_all_input_ids1 = torch.tensor([f.input_ids for f in unsupfeatures1], dtype=torch.long)
        unsup_all_attention_mask1 = torch.tensor([f.attention_mask for f in unsupfeatures1], dtype=torch.long)
        unsup_all_token_type_ids1 = torch.tensor([f.token_type_ids for f in unsupfeatures1], dtype=torch.long)

        unsup_all_input_ids2 = torch.tensor([f.input_ids for f in unsupfeatures2], dtype=torch.long)
        unsup_all_attention_mask2 = torch.tensor([f.attention_mask for f in unsupfeatures2], dtype=torch.long)
        unsup_all_token_type_ids2 = torch.tensor([f.token_type_ids for f in unsupfeatures2], dtype=torch.long)

        unsup_all_input_ids3 = torch.tensor([f.input_ids for f in unsupfeatures3], dtype=torch.long)
        unsup_all_attention_mask3 = torch.tensor([f.attention_mask for f in unsupfeatures3], dtype=torch.long)
        unsup_all_token_type_ids3 = torch.tensor([f.token_type_ids for f in unsupfeatures3], dtype=torch.long)

        unsup_all_input_ids4 = torch.tensor([f.input_ids for f in unsupfeatures4], dtype=torch.long)
        unsup_all_attention_mask4 = torch.tensor([f.attention_mask for f in unsupfeatures4], dtype=torch.long)
        unsup_all_token_type_ids4 = torch.tensor([f.token_type_ids for f in unsupfeatures4], dtype=torch.long)


    if output_mode == "classification":
        if not evaluate:
            all_labels1 = torch.tensor([f.label for f in features1], dtype=torch.long)
            all_labels2 = torch.tensor([f.label for f in features2], dtype=torch.long)
            all_labels3 = torch.tensor([f.label for f in features3], dtype=torch.long)
            all_labels4 = torch.tensor([f.label for f in features4], dtype=torch.long)
            unsup_all_labels1 = torch.tensor([f.label for f in unsupfeatures1], dtype=torch.long)
            unsup_all_labels2 = torch.tensor([f.label for f in unsupfeatures2], dtype=torch.long)
            unsup_all_labels3 = torch.tensor([f.label for f in unsupfeatures3], dtype=torch.long)
            unsup_all_labels4 = torch.tensor([f.label for f in unsupfeatures4], dtype=torch.long)
            aug_all_labels1 = torch.tensor([f.label for f in augfeatures1], dtype=torch.long)
            aug_all_labels2 = torch.tensor([f.label for f in augfeatures2], dtype=torch.long)
            aug_all_labels3 = torch.tensor([f.label for f in augfeatures3], dtype=torch.long)
            aug_all_labels4 = torch.tensor([f.label for f in augfeatures4], dtype=torch.long)
        else: 
            unsup_all_labels = torch.tensor([f.label for f in unsupfeatures], dtype=torch.long)
            unsup_all_labels1 = torch.tensor([f.label for f in unsupfeatures1], dtype=torch.long)
            unsup_all_labels2 = torch.tensor([f.label for f in unsupfeatures2], dtype=torch.long)
            unsup_all_labels3 = torch.tensor([f.label for f in unsupfeatures3], dtype=torch.long)
            unsup_all_labels4 = torch.tensor([f.label for f in unsupfeatures4], dtype=torch.long)


            
        # print('svnoenfi[gcpmrhpoxmpoew]chpojbpermdshpimgpcjhp')
        # print(type(all_labels))
        # print(all_labels)
        # unsup_all_labels = torch.tensor([f.label for f in unsupfeatures], dtype=torch.long)
        # aug_all_labels = torch.tensor([f.label for f in augfeatures], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for XNLI.")
    # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # print('unsupfeatures')
    # print(unsupfeatures[0])
    # print('augfeatures')
    # print(augfeatures[0])
    # print(all_labels)
    # print(all_input_ids1)
    if not evaluate:
        books_dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels1)
        dvd_dataset = TensorDataset(all_input_ids2, all_attention_mask2, all_token_type_ids2, all_labels2)
        electronics_dataset = TensorDataset(all_input_ids3, all_attention_mask3, all_token_type_ids3, all_labels3)
        kitchen_dataset = TensorDataset(all_input_ids4, all_attention_mask4, all_token_type_ids4, all_labels4)
        books_unsupdataset = TensorDataset(unsup_all_input_ids1, unsup_all_attention_mask1, unsup_all_token_type_ids1, unsup_all_labels1)
        dvd_unsupdataset = TensorDataset(unsup_all_input_ids2, unsup_all_attention_mask2, unsup_all_token_type_ids2, unsup_all_labels2)
        electronics_unsupdataset = TensorDataset(unsup_all_input_ids3, unsup_all_attention_mask3, unsup_all_token_type_ids3, unsup_all_labels3)
        kitchen_unsupdataset = TensorDataset(unsup_all_input_ids4, unsup_all_attention_mask4, unsup_all_token_type_ids4, unsup_all_labels4)

        books_augdataset = TensorDataset(aug_all_input_ids1, aug_all_attention_mask1, aug_all_token_type_ids1, aug_all_labels1)
        dvd_augdataset = TensorDataset(aug_all_input_ids2, aug_all_attention_mask2, aug_all_token_type_ids2, aug_all_labels2)
        electronics_augdataset = TensorDataset(aug_all_input_ids3, aug_all_attention_mask3, aug_all_token_type_ids3, aug_all_labels3)
        kitchen_augdataset = TensorDataset(aug_all_input_ids4, aug_all_attention_mask4, aug_all_token_type_ids4, aug_all_labels4)
        return books_dataset,dvd_dataset,electronics_dataset,kitchen_dataset,books_unsupdataset,dvd_unsupdataset,electronics_unsupdataset,kitchen_unsupdataset,books_augdataset,dvd_augdataset,electronics_augdataset,kitchen_augdataset
    else:
    
        unsupdataset = TensorDataset(unsup_all_input_ids, unsup_all_attention_mask, unsup_all_token_type_ids, unsup_all_labels)
        books_unsupdataset = TensorDataset(unsup_all_input_ids1, unsup_all_attention_mask1, unsup_all_token_type_ids1, unsup_all_labels1)
        dvd_unsupdataset = TensorDataset(unsup_all_input_ids2, unsup_all_attention_mask2, unsup_all_token_type_ids2, unsup_all_labels2)
        electronics_unsupdataset = TensorDataset(unsup_all_input_ids3, unsup_all_attention_mask3, unsup_all_token_type_ids3, unsup_all_labels3)
        kitchen_unsupdataset = TensorDataset(unsup_all_input_ids4, unsup_all_attention_mask4, unsup_all_token_type_ids4, unsup_all_labels4)
    
        return unsupdataset, books_unsupdataset,dvd_unsupdataset,electronics_unsupdataset,kitchen_unsupdataset

def evaluate(args, model,model2,model3, tokenizer4, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        unsupdataset,eval_dataset1,eval_dataset2,eval_dataset3,eval_dataset4 = load_and_cache_examples(args, eval_task, tokenizer4, evaluate=True)
        eval_dataset=eval_dataset4
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds1 = None
        preds2=None
        preds3=None
        out_label_ids1 = None
        out_label_ids2 =None
        out_label_ids3=None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                outputs1 = model(**inputs)
                outputs2=model2(**inputs)
                outputs3=model3(**inputs)
                tmp_eval_loss1, logits1 = outputs1[:2]
                tmp_eval_loss2, logits2 = outputs2[:2]
                tmp_eval_loss3, logits3 = outputs3[:2]

                eval_loss += (tmp_eval_loss1.mean().item()+tmp_eval_loss2.mean().item()+tmp_eval_loss3.mean().item())//3
            nb_eval_steps += 1
            if preds1 is None:
                preds1 = logits1.detach().cpu().numpy()
                out_label_ids1 = inputs["labels"].detach().cpu().numpy()
            else:
                preds1 = np.append(preds1, logits1.detach().cpu().numpy(), axis=0)
                out_label_ids1 = np.append(out_label_ids1, inputs["labels"].detach().cpu().numpy(), axis=0)
            if preds2 is None:
                preds2 = logits2.detach().cpu().numpy()
                out_label_ids2 = inputs["labels"].detach().cpu().numpy()
            else:
                preds2 = np.append(preds2, logits2.detach().cpu().numpy(), axis=0)
                out_label_ids2 = np.append(out_label_ids2, inputs["labels"].detach().cpu().numpy(), axis=0)
            if preds3 is None:
                preds3 = logits3.detach().cpu().numpy()
                out_label_ids3 = inputs["labels"].detach().cpu().numpy()
            else:
                preds3 = np.append(preds3, logits3.detach().cpu().numpy(), axis=0)
                out_label_ids3 = np.append(out_label_ids3, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds1 = np.argmax(preds1, axis=1)
            preds2 = np.argmax(preds2, axis=1)
            preds3 = np.argmax(preds3, axis=1)
        else:
            raise ValueError("No other `output_mode` for XNLI.")
        preds_list1=preds1.tolist()
        preds_list2=preds2.tolist()
        preds_list3=preds3.tolist()
        from collections import Counter
        
        final_pred=[]
        for index , i in enumerate(preds_list1):    
            preds=[]    
            preds.extend([i,preds_list2[index],preds_list3[index]])
            
            c = Counter(preds)
            value, count = c.most_common()[0]
            final_pred.append(value)
        print(len(final_pred))
        final_pred=np.array(final_pred)
        print(len(final_pred))
        print(final_pred)
        np.save('unlabelled_psuedo_label.npy',final_pred)
        result = compute_metrics(eval_task, preds3, out_label_ids1)
        
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            import logging
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results
def main():
   
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='sorted_data_acl/',
        type=str,
       
        help="The input data dir. Should contain the .tsv files (or other data files) for the task. FOR TARGET DOMAIN",
    )
    
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        
        help="Model type selected in the list: "+".join(MODEL_CLASSES.keys())" ,
    )
    parser.add_argument(
        "--task_name",
        default="appreview",
        type=str,
        required=False,
        help="",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='bert-base-chinese',
        type=str,
        
        help="Path to pre-trained model or shortcut name selected in the list:",
    )
    parser.add_argument(
        "--output_dir",
        default='models/output1/',
        type=str,
        
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--target_domain", default='books',type=str, help="Choose target domain,remains domains are sources domain")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_unsup_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for unsupervised training.")
    parser.add_argument("--loss_weight", default=0.5, type=float, help="loss weight between supervised loss and unsupervised loss")
    parser.add_argument("--uda_confidence_thresh", default=0.45, type=float, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--uda_softmax_temp",default=0.85,type=float,help="aaa")
    parser.add_argument("--tsa",default=True,type=bool, help="whether uda tsa or not")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--source", type=str, default="", help="For distant debugging.")

    
    args = parser.parse_args()
    args.do_train=False
    args.do_eval=True
    print(args.local_rank)
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    # args.no_cuda=True
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        print(args.no_cuda)
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        print('xwspnpiernvbpmqpoefmb[oqef[obm[oqetmb[qoemfdb;kdmkbmwrkmbpeqtmbmeqp')
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    import logging
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name]()
    if args.task_name=='appreview':
        processor = processors[args.task_name]()
        unsupprocessor = processors['unsup']()
        augprocessor = processors['aug']()
    if args.task_name=='pretrain':
        processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        #num_labels=len(label_list),
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model1 = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model2 = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # print(model)
    config.num_labels = len(label_list)
    model.num_labels = len(label_list)
    model.classifier = torch.nn.Linear(config.hidden_size, model.num_labels)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
       
        # train_dataset= load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        source_labeled_list=[]
        source_unlabeled_list=[]
        source_aug_list=[]
        books_dataset,dvd_dataset,electronics_dataset,kitchen_dataset,books_unsupdataset,dvd_unsupdataset,electronics_unsupdataset,kitchen_unsupdataset,books_augdataset,dvd_augdataset,electronics_augdataset,kitchen_augdataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        source_labeled_list.extend((books_dataset,dvd_dataset,electronics_dataset,kitchen_dataset))
        source_unlabeled_list.extend((books_unsupdataset,dvd_unsupdataset,electronics_unsupdataset,kitchen_unsupdataset))
        source_aug_list.extend((books_augdataset,dvd_augdataset,electronics_augdataset,kitchen_augdataset))
        # print(aug_dataset)
        # if evaluate==True:
            # print('what the fuck what the fuck what the fuck what the fuck what the fuck what the fuck what the fuck')
            # train_dataset= load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        # print('------------------------------------------------------------------------------------------------------------------------------------------')
        # print(train_dataset)

        # global_step, tr_loss = train(args, train_dataset,unsup_dataset,aug_dataset, model, tokenizer)
        global_step1, tr_loss1 = train_book_domain(args, source_labeled_list,source_unlabeled_list,source_aug_list, model, tokenizer)
        global_step2, tr_loss2 = train_dvd_domain(args, source_labeled_list,source_unlabeled_list,source_aug_list, model, tokenizer)
        global_step3, tr_loss3 = train_electronics_domain(args, source_labeled_list,source_unlabeled_list,source_aug_list, model, tokenizer)
        # unsup_output_list = unsuptrain(args, unsup_dataset, model, tokenizer)
        # unsup_output_list = unsuptrain(args, aug_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step1, tr_loss1)
        logger.info(" global_step = %s, average loss = %s", global_step2, tr_loss2)
        logger.info(" global_step = %s, average loss = %s", global_step3, tr_loss3)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model1 = model_class.from_pretrained(args.output_dir)
        tokenizer1 = tokenizer_class.from_pretrained(args.output_dir)
        model1.to(args.device)
     
        

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            config_class,model_class,tokenizer_class=MODEL_CLASSES["bert"]
            model1= model_class.from_pretrained("models/output1/")
            tokenizer1= tokenizer_class.from_pretrained("models/output4/")
            model1.to(device)
            config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
            model2 = model_class.from_pretrained("models/output2/")
            tokenizer2 = tokenizer_class.from_pretrained("models/output2/")
            device = torch.device("cuda")
            model2.to(device)
            # print('model2:',model)
            config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
            model3 = model_class.from_pretrained("models/output3/")
            tokenizer3 = tokenizer_class.from_pretrained("models/output3/")
            device = torch.device("cuda")
            model3.to(device)
            print(model1)
            print(model2)
            print(model3)
            # print('model3:',model)
            result = evaluate(args, model1,model2,model3, tokenizer1, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
    

