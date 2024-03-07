import torch
import random
import copy
import re
import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

DO_LOWER_CASE = {
    "bert-base-uncased": True,
    "bert-base-cased": False,
    "roberta-base": False,
    "dmis-lab/biobert-base-cased-v1.1": False,
    "google/electra-base-discriminator": True,
    "princeton-nlp/unsup-simcse-bert-base-uncased": True,
    "princeton-nlp/sup-simcse-bert-base-uncased": True,
    "openai-gpt": True,
    "facebook/bart-base": False,
    "allenai/scibert_scivocab_cased": False,
    "allenai/scibert_scivocab_uncased": True,
    "distilbert-base-cased": False,
    "distilbert-base-uncased": True,
    "nghuyong/ernie-2.0-base-en": True,
    "distilroberta-base": False,
    "distilgpt2": False,
    "distilbert-base-multilingual-cased": False,
    "albert-base-v2": True,
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": True,
    "michiyasunaga/BioLinkBERT-base": True
}


def max_length(dataset: str):
    if dataset.startswith("bioasq"):
        return 24, 168
    elif dataset == "scifact":
        return 36, 512
    elif dataset == "mrpc":
        return 36, 36
    elif dataset == "mutual":
        return 196, 36


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, do_lower_case=DO_LOWER_CASE[args.model_name_or_path])
        if args.model_name_or_path in ["openai-gpt", "distilgpt2"]:
            self.tokenizer.pad_token_id = 0
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    
    def tokenize(self, texts, text_type):
        tokens = []
        input_ids = []
        for text in tqdm(texts, desc=f'[tokenize {text_type}]', leave=True):
            tokens.append(self.tokenizer.tokenize(str(text)))
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens[-1]))
        return tokens, input_ids
    
    def padding(self, input_ids, max_len, backward=False):
        _input_ids = list(input_ids)
        for i, item in enumerate(_input_ids):
            if max_len == -1:
                _input_ids[i] = ([self.cls_token_id] if self.cls_token_id is not None else []) + item[-510:] if backward else item[:510] + ([self.sep_token_id] if self.sep_token_id is not None else [])
            else:
                _input_ids[i] = ([self.cls_token_id] if self.cls_token_id is not None else []) + item[-max_len+2:] if backward else item[:max_len-2] + ([self.sep_token_id] if self.sep_token_id is not None else [])

        max_len = max([len(s) for s in _input_ids])
        input_ids = np.array([item + [self.pad_token_id] * (max_len - len(item)) for item in _input_ids], dtype=np.int32)
        attention_mask = np.array([[1] * len(item) + [0] * (max_len-len(item)) for item in _input_ids], dtype=np.int32)
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)

        return input_ids.to(self.device), attention_mask.to(self.device)


class ReQADataset(BaseDataset):
    def __init__(self, args, split='train', data_type=None):
        super(ReQADataset, self).__init__(args)
        self.split = split
        self.data_type = data_type
        self.dataset = args.dataset
        self.data_folder = f"./data/{args.dataset}"
        self.max_question_len, self.max_answer_len = max_length(self.dataset)
        self.process(args.aggregate)

    def process(self, aggregate):
        # Load data features from cache or datas et file
        cached_dir = f"./cached_data/{self.dataset}"
        if aggregate:
            cached_dir = cached_dir + '_aggregate'
        if not os.path.exists(cached_dir):
            os.makedirs(cached_dir)
        plm_name = [s for s in self.args.model_name_or_path.split('/') if s !=''][-1]
        cached_dataset_file = os.path.join(cached_dir, f'{"train" if self.split=="dev" else self.split}_{plm_name}')
        # load processed dataset or process the original dataset
        if os.path.exists(cached_dataset_file):# and not self.args.overwrite_cache:
            logging.info("Loading dataset from cached file %s", cached_dataset_file)
            data_dict = torch.load(cached_dataset_file)
            self.question_input_ids = data_dict["questions"]
            self.answer_input_ids = data_dict["answers"]
            if self.split == 'test':
                self.ground_truths = data_dict['ground_truths']
            else:
                self.question_ids = data_dict['question_ids']
        else:
            logging.info("Creating instances from dataset file at %s", self.data_folder)
            with open(self.data_folder + f'/{"train" if self.split=="dev" else self.split}.json', 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            if self.split == 'test':
                questions = raw_data['questions']
                answers = raw_data['candidates']
                self.ground_truths = raw_data['ground_truths']
            else:
                questions = raw_data['questions']
                answers = raw_data['answers']
                self.question_ids = raw_data['question_ids']
            
            question_tokens, self.question_input_ids = self.tokenize(questions, 'question')
            answer_tokens, self.answer_input_ids = self.tokenize(answers, 'answer  ')
                
            logging.info(f"question: {questions[0]}")
            logging.info(f'question tokens: {question_tokens[0]}')
            logging.info(f'question input ids: {self.question_input_ids[0]}')
            logging.info('')
            logging.info(f"answer: {answers[0]}")
            logging.info(f'answer tokens: {answer_tokens[0]}')
            logging.info(f'answer input ids: {self.answer_input_ids[0]}')
            logging.info('')
            # save data
            if self.split == 'test':
                saved_data = {
                    'questions': self.question_input_ids,
                    'answers': self.answer_input_ids,
                    'ground_truths': self.ground_truths
                }
            else:
                saved_data = {
                    'questions': self.question_input_ids,
                    'answers': self.answer_input_ids,
                    'question_ids': self.question_ids
                }
                
            logging.info("Saving processed dataset to %s", cached_dataset_file)
            torch.save(saved_data, cached_dataset_file)
        
        
    def __len__(self):
        if self.split == "test" and self.data_type == "candidate":
            return len(self.answer_input_ids)
        else:
            return len(self.question_input_ids)

    def __getitem__(self, idx):
        if self.split == 'test':
            if self.data_type == "question":
                return (self.args, self.question_input_ids[idx], self.ground_truths[idx])
            elif self.data_type == "candidate":
                return (self.args, self.answer_input_ids[idx])
        else:
            return (self.args, 
                    self.question_input_ids[idx],
                    self.answer_input_ids[idx],
                    self.question_ids[idx])
            
    
    def collate_fn(self, raw_batch):
        args = raw_batch[-1][0]
        batch = dict()
        if self.split == 'test':
            if self.data_type == "question":
                _, input_ids, ground_truth = list(zip(*raw_batch))
                batch['ground_truth'] = ground_truth
                batch['input_ids'], batch['attention_mask'] = self.padding(input_ids, -1, backward=True if self.dataset=="mutual" else False)
            elif self.data_type == "candidate":
                _, input_ids = list(zip(*raw_batch))
                batch['input_ids'], batch['attention_mask'] = self.padding(input_ids, -1, backward=False)
        else:
            _, question_input_ids, answer_input_ids, question_ids = list(zip(*raw_batch))
            batch['src_input_ids'], batch['src_attention_mask'] = self.padding(question_input_ids, self.max_question_len, backward=True if self.dataset=="mutual" else False)
            batch['tgt_input_ids'], batch['tgt_attention_mask'] = self.padding(answer_input_ids, self.max_answer_len, backward=False)
            batch['src_ids'] = torch.LongTensor(list(question_ids)).to(self.device)
            # import pdb; pdb.set_trace()
            
        return batch



def do_sentence_breaks(uni_text):
    """Uses regexp substitution rules to insert newlines as sentence breaks.

    Args:
    uni_text: A (multi-sentence) passage of text, in Unicode.

    Returns:
    A Unicode string with internal newlines representing the inferred sentence
    breaks.
    """

    # The main split, looks for sequence of:
    #   - sentence-ending punctuation: [.?!]
    #   - optional quotes, parens, spaces: [)'" \u201D]*
    #   - whitespace: \s
    #   - optional whitespace: \s*
    #   - optional opening quotes, bracket, paren: [['"(\u201C]?
    #   - upper case letter or digit
    txt = re.sub(r'''([.?!][)'" %s]*)\s(\s*[['"(%s]?[A-Z0-9])''' % ('\u201D', '\u201C'),
               r'\1\n\2',
               uni_text)

    # Wiki-specific split, for sentence-final editorial scraps (which can stack):
    #  - ".[citation needed]", ".[note 1] ", ".[c] ", ".[n 8] "
    txt = re.sub(r'''([.?!]['"]?)((\[[a-zA-Z0-9 ?]+\])+)\s(\s*['"(]?[A-Z0-9])''',
               r'\1\2\n\4', txt)

    # Wiki-specific split, for ellipses in multi-sentence quotes:
    # "need such things [...] But"
    txt = re.sub(r'(\[\.\.\.\]\s*)\s(\[?[A-Z])', r'\1\n\2', txt)

    # Rejoin for:
    #   - social, military, religious, and professional titles
    #   - common literary abbreviations
    #   - month name abbreviations
    #   - geographical abbreviations
    #
    txt = re.sub(r'\b(Mrs?|Ms|Dr|Prof|Fr|Rev|Msgr|Sta?)\.\n', r'\1. ', txt)
    txt = re.sub(r'\b(Lt|Gen|Col|Maj|Adm|Capt|Sgt|Rep|Gov|Sen|Pres)\.\n',
               r'\1. ',
               txt)
    txt = re.sub(r'\b(e\.g|i\.?e|vs?|pp?|cf|a\.k\.a|approx|app|es[pt]|tr)\.\n',
               r'\1. ',
               txt)
    txt = re.sub(r'\b(Jan|Aug|Oct|Nov|Dec)\.\n', r'\1. ', txt)
    txt = re.sub(r'\b(Mt|Ft)\.\n', r'\1. ', txt)
    txt = re.sub(r'\b([ap]\.m)\.\n(Eastern|EST)\b', r'\1. \2', txt)

    # Rejoin for personal names with 3,2, or 1 initials preceding the last name.
    txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3 \4',
               txt)
    txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3',
               txt)
    txt = re.sub(r'\b([A-Z]\.[A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)
    txt = re.sub(r'\b([A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)

    # Resplit for common sentence starts:
    #   - The, This, That, ...
    #   - Meanwhile, However,
    #   - In, On, By, During, After, ...
    txt = re.sub(r'([.!?][\'")]*) (The|This|That|These|It) ', r'\1\n\2 ', txt)
    txt = re.sub(r'(\.) (Meanwhile|However)', r'\1\n\2', txt)
    txt = re.sub(r'(\.) (In|On|By|During|After|Under|Although|Yet|As |Several'
               r'|According to) ',
               r'\1\n\2 ',
               txt)

    # Rejoin for:
    #   - numbered parts of documents.
    #   - born, died, ruled, circa, flourished ...
    #   - et al (2005), ...
    #   - H.R. 2000
    txt = re.sub(r'\b([Aa]rt|[Nn]o|Opp?|ch|Sec|cl|Rec|Ecl|Cor|Lk|Jn|Vol)\.\n'
               r'([0-9IVX]+)\b',
               r'\1. \2',
               txt)
    txt = re.sub(r'\b([bdrc]|ca|fl)\.\n([A-Z0-9])', r'\1. \2', txt)
    txt = re.sub(r'\b(et al)\.\n(\(?[0-9]{4}\b)', r'\1. \2', txt)
    txt = re.sub(r'\b(H\.R\.)\n([0-9])', r'\1 \2', txt)

    # SQuAD-specific joins.
    txt = re.sub(r'(I Am\.\.\.)\n(Sasha Fierce|World Tour)', r'\1 \2', txt)
    txt = re.sub(r'(Warner Bros\.)\n(Records|Entertainment)', r'\1 \2', txt)
    txt = re.sub(r'(U\.S\.)\n(\(?\d\d+)', r'\1 \2', txt)
    txt = re.sub(r'\b(Rs\.)\n(\d)', r'\1 \2', txt)

    # SQuAD-specific splits.
    txt = re.sub(r'\b(Jay Z\.) ([A-Z])', r'\1\n\2', txt)
    txt = re.sub(r'\b(Washington, D\.C\.) ([A-Z])', r'\1\n\2', txt)
    txt = re.sub(r'\b(for 4\.\)) ([A-Z])', r'\1\n\2', txt)
    txt = re.sub(r'\b(Wii U\.) ([A-Z])', r'\1\n\2', txt)
    txt = re.sub(r'\. (iPod|iTunes)', r'.\n\1', txt)
    txt = re.sub(r' (\[\.\.\.\]\n)', r'\n\1', txt)
    txt = re.sub(r'(\.Sc\.)\n', r'\1 ', txt)
    txt = re.sub(r' (%s [A-Z])' % '\u2022', r'\n\1', txt)
    return txt

def infer_sentence_breaks(uni_text):
    """
    Generates (start, end) pairs demarking sentences in the text.
    """
    uni_text = re.sub(r'\n', r' ', uni_text)  # Remove pre-existing newlines.
    text_with_breaks = do_sentence_breaks(uni_text)
    starts = [m.end() for m in re.finditer(r'^\s*', text_with_breaks, re.M)]
    sentences = [s.strip() for s in text_with_breaks.split('\n')]
    assert len(starts) == len(sentences)
    for i in range(len(sentences)):
        start = starts[i]
        end = start + len(sentences[i])
        yield start, end