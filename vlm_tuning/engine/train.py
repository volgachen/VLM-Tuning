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
import argparse
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

from vlm_tuning.engine.llava_trainer import LLaVATrainer

from detectron2.config import LazyConfig, instantiate


def train(args):
    cfg = LazyConfig.load(args.config_file)
    model, tokenizer, data_args, training_args = instantiate(cfg.model)
    if args.output_dir is not None:
        training_args.output_dir = args.output_dir
        print("Setting output_dir to", training_args.output_dir)
    cfg.data_module.tokenizer=tokenizer
    cfg.data_module.mm_use_im_start_end = data_args.mm_use_im_start_end
    data_module = instantiate(cfg.data_module)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--local_rank", type=int, required=True)
    args = parser.parse_args()
    train(args)
