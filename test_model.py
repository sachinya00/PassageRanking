import logging
import os
import random
from tqdm import tqdm, trange
import csv

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from helpers import BertTokenizer
from model_files import PassageRanking

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class MSAIC_Example(object):
    """A single training/test example."""
    def __init__(self,
                 qid,
                 ques,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 ending_4,
                 ending_5,
                 ending_6,
                 ending_7,
                 ending_8,
                 ending_9,
                 label = None):
        self.qid = qid
        self.ques = ques
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
            ending_4,
            ending_5,
            ending_6,
            ending_7,
            ending_8,
            ending_9,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"qid: {self.qid}",
            f"ques: {self.ques}",
            f"ending_0: {self.endings[0]}",
            f"ending_1: {self.endings[1]}",
            f"ending_2: {self.endings[2]}",
            f"ending_3: {self.endings[3]}",
            f"ending_4: {self.endings[4]}",
            f"ending_5: {self.endings[5]}",
            f"ending_6: {self.endings[6]}",
            f"ending_7: {self.endings[7]}",
            f"ending_8: {self.endings[8]}",
            f"ending_9: {self.endings[9]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_MSAIC_examples(input_file, is_validating):
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    if is_validating and lines[0][-1] != 'label':
        raise ValueError(
            "For Validating, the input file must contain a label column."
        )

    examples = [
        MSAIC_Example(
            qid = int(line[0]),
            ques = line[1],
            ending_0 = line[2],
            ending_1 = line[3],
            ending_2 = line[4],
            ending_3 = line[5],
            ending_4 = line[6],
            ending_5 = line[7],
            ending_6 = line[8],
            ending_7 = line[9],
            ending_8 = line[10],
            ending_9 = line[11],
            label = int(line[12]) if is_validating else None
        ) for line in lines[1:2000] # we skip the line with the column names
    ]

    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example_index, example in enumerate(tqdm(examples)):
        context_tokens = tokenizer.tokenize(example.ques)
        
        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            context_tokens_choice = context_tokens[:]
            ending_tokens = tokenizer.tokenize(ending)
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label

        features.append(
            InputFeatures(
                example_id = example.qid,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
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

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def get_csv_from_tsv(data_file, data_dir):
    sp = -1
    data_file_path = os.path.join(data_dir,data_file)
    with open(data_file_path,'r') as fp:
        lines = fp.readlines()
        dataset = {}
        for x in lines:
            data = x.strip().split('\t')
            qid = data[0]
            ques = data[1]
            passage = data[2]
            
            if qid in dataset:
                if len(dataset[qid][1]) == 10:
                    logger.info("Duplicate Id {} found in data. Assigning unique Id {}".format(qid,str(sp)))
                    if sp in dataset:
                        dataset[sp][1].append(passage)
                        if len(dataset[sp][1]) == 10:
                            sp = sp-1
                    else:
                        dataset[sp] = []
                        dataset[sp].append([qid,ques])
                        dataset[sp].append([passage])
                else:
                    dataset[qid][1].append(passage)
            else:
                dataset[qid] = []
                dataset[qid].append([qid,ques])
                dataset[qid].append([passage])
            
        test_set = []
        keys = list(dataset.keys())
        for i,x in enumerate(tqdm(range(len(keys)))):
            test_set.append(dataset[keys[x]])
            
        test_file = os.path.join(data_dir,'test.csv')
        with open(test_file,'w') as op:
            csv_writer = csv.writer(op)
            csv_writer.writerow(['qid','ques','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
            for x in test_set:
                csv_writer.writerow([x[0][0],x[0][1],x[1][0],x[1][1],x[1][2],x[1][3],x[1][4],x[1][5],x[1][6],x[1][7],x[1][8],x[1][9]])
            op.close()

def test():
    
    options = {
        'data_name':'test.tsv',
        'data_dir':'data/',
        'output_dir':'output/',
        'model_dir':'models/',
        'vocab_path': 'vocab.txt',
        'config_path': 'bert_config.json',
        'max_seq_length':128,
        'is_validating':False,
        'eval_batch_size':32,
        'use_cuda':True,
        'seed':42,
    }
    
    os.makedirs(options['output_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and options['use_cuda'] else "cpu")
    n_gpu = 1 if torch.cuda.is_available() else 0

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(options['seed'])
    np.random.seed(options['seed'])
    torch.manual_seed(options['seed'])
    if n_gpu > 0:
        torch.cuda.manual_seed_all(options['seed'])

    logger.info("Getting Tokenizer")
    tokenizer = BertTokenizer(options['vocab_path'], do_lower_case=True)

    logger.info("Loading Pre-Trained Models")
    # Load pre-trained model
    output_model_file1 = os.path.join(options['model_dir'], "model1.bin")
    model_state_dict1 = torch.load(output_model_file1)
    model1 = PassageRanking.from_pretrained(
        config_path=options['config_path'],
        state_dict=model_state_dict1,
        num_choices=10)
    model1.to(device)

    output_model_file2 = os.path.join(options['model_dir'], "model2.bin")
    model_state_dict2 = torch.load(output_model_file2)
    model2 = PassageRanking.from_pretrained(
        config_path=options['config_path'],
        state_dict=model_state_dict2,
        num_choices=10)
    model2.to(device)

    logger.info("Preparing Data")
    get_csv_from_tsv(options['data_name'], options['data_dir'])
    #Contruct Data Loader from Data File.
    eval_examples = read_MSAIC_examples(os.path.join(options['data_dir'], 'test.csv'), is_validating = options['is_validating'])
    eval_features = convert_examples_to_features(eval_examples, tokenizer, options['max_seq_length'])
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", options['eval_batch_size'])
    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_qid = torch.tensor([f.example_id for f in eval_features], dtype=torch.long)
    #all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_qid)
    
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=options['eval_batch_size'])

    model1.eval()
    model2.eval()

    output_test_file = os.path.join(options['output_dir'], "answer.tsv")
    with open(output_test_file,'w') as fp:
        for batch in tqdm(eval_dataloader, desc="Progress"):
            input_ids, input_mask, segment_ids, qids = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits1 = model1(input_ids, segment_ids, input_mask)
                logits2 = model2(input_ids, segment_ids, input_mask)

            logits = (logits1*0.5) + (logits2*0.5)
            logits = logits.detach().cpu().numpy()
            qids = qids.detach().cpu().numpy()
            for qid,vec in zip(qids,logits):
                row = str(qid)
                for x in vec:
                    row += '\t' + str(x)
                fp.write(row + '\n')
        fp.close()
        logger.info("***** Done *****")
            
test()
