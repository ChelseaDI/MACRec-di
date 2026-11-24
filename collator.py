import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


def add_image_history(inputs, batch, tokenizer):
    if "image_history" in batch[0]:
        image_inputs = tokenizer(
            [sample["image_history"] for sample in batch],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        inputs["image_input_ids"] = image_inputs["input_ids"]
        inputs["image_attention_mask"] = image_inputs["attention_mask"]


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        print("self.tokenizer.model_max_length", self.tokenizer.model_max_length)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100


        if "task_flag" in batch[0] and "image_history" not in batch[0]:
            task_flags = torch.tensor([d["task_flag"] for d in batch])
            inputs['task_flag'] = task_flags
        add_image_history(inputs, batch, self.tokenizer)
        return inputs



class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.prefix_token = vars(args).get('prefix_token', '')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
            
        if isinstance(self.tokenizer, LlamaTokenizer):
            self.tokenizer.padding_side = "left"

    def __call__(self, batch):

        input_texts = [d["input_ids"] + self.prefix_token for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)
    
class TestCollatorSave(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.prefix_token = vars(args).get('prefix_token', '')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
            
        if isinstance(self.tokenizer, LlamaTokenizer):
            self.tokenizer.padding_side = "left"

    def __call__(self, batch):

        input_texts = [d["input_ids"] + self.prefix_token for d in batch]
        targets = [d["labels"] for d in batch]
        users = [d["label"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        add_image_history(inputs, batch, self.tokenizer)
        return (inputs, targets, users)
    



