# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers

from torch.utils.data import Dataset

import vlm_tuning.data.conversation as conversation_lib
from vlm_tuning.data.processor import preprocess

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"


def preprocess_multimodal(
    sources: Sequence[str],
    is_multimodal: bool = False,
    mm_use_im_start_end: bool = False,
) -> Dict:
    if not is_multimodal:
        return sources

    # i haven't seen this used anywhere
    assert False, "Not ready yet"
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_loader,
                 is_multimodal: bool = False,
                 mm_use_im_start_end: bool = False):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

        self.image_loader = image_loader
        self.mm_use_im_start_end = mm_use_im_start_end
        self.is_multimodal = is_multimodal

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image = self.image_loader(image_file)
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.is_multimodal,
                self.mm_use_im_start_end)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_loader.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        # others
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            output_hidden_states = True,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(data_path,
                                tokenizer: transformers.PreTrainedTokenizer,
                                image_loader,
                                mm_use_im_start_end) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
                                data_path=data_path,
                                tokenizer=tokenizer,
                                image_loader=image_loader,
                                mm_use_im_start_end=mm_use_im_start_end)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


if __name__ == "__main__":
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_llama_2"]

    from image_loader import ImageLoader
    from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
    data_module = make_supervised_data_module(
        data_path="/data/zhiyang_chen/datasets/LLaVA-Instruct-150K/rela1111_111_last_llava_30k.json",
        tokenizer=transformers.AutoTokenizer.from_pretrained("/data/zhiyang_chen/models/llava-llama-2-13b-chat-lightning-preview/"),
        image_loader=ImageLoader(
            image_folder="/data/zhiyang_chen/datasets/coco/train2017",
            image_processor=CLIPImageProcessor.from_pretrained("/data/zhiyang_chen/models/clip-vit-large-patch14/"),
            image_aspect_ratio="square",
        ),
        mm_use_im_start_end=False,
    )

    a = data_module["train_dataset"][0]
    import pdb;pdb.set_trace()