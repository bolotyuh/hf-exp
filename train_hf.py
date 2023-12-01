#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import evaluate
import numpy as np
import torch
import transformers
from PIL import Image
from timm.data.auto_augment import rand_augment_transform
from torch.nn import CrossEntropyLoss
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.36.0.dev0")


MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# def pil_loader(path: str):
#     with open(path, "rb") as f:
#         im = Image.open(f)
#         return im.convert("RGB")


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples], dtype=torch.long)

    return {"pixel_values": pixel_values, "labels": labels}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    train_dir: str = field(
        default=os.environ.get("SM_CHANNEL_TRAIN", ""),
        metadata={"help": "A folder containing the training data."},
    )
    validation_dir: str = field(
        default=os.environ.get("SM_CHANNEL_TEST", ""),
        metadata={"help": "A folder containing the validation data."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


    # SM_CHANNEL_TEST=/opt/ml/input/data/test
    # SM_CHANNEL_TRAIN=/opt/ml/input/data/train
    # print(data_args.train_dir)
    # print(data_args.validation_dir)

#     logging.info(f"TRAIN_DIR: {data_args.train_dir}")
#     logging.info(f"VALIDATION_DIR: {data_args.validation_dir}")
#     print(os.listdir(data_args.validation_dir))
#     logging.info(f"DIRS: {os.listdir(data_args.validation_dir)}")

#     training_args.remove_unused_columns = False
#     training_args.do_train = True
#     training_args.do_eval = True
#     training_args.fp16 = True
#     training_args.report_to = "wandb"
#     training_args.logging_strategy = "epoch"
#     training_args.evaluation_strategy = "epoch"
#     training_args.save_strategy = "epoch"
#     training_args.save_total_limit = 2
#     training_args.load_best_model_at_end = True
#     training_args.seed = 41

    # Data, model, and output directories
    # parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    #   logging_dir=f"{args.output_data_dir}/logs",
    # parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    # parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    # writes eval result to file which can be accessed later in s3 ouput
    # with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
    #     print(f"***** Eval results *****")
    #     for key, value in sorted(eval_result.items()):
    #         writer.write(f"{key} = {value}\n")

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        if image_processor.size["height"] == image_processor.size["width"]:
            size = image_processor.size["height"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])

    aa_params = dict(
        translate_const=int(size * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in image_processor.image_mean]),
    )

    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )

    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            # rand_augment_transform(config_str="rand-m9-mstd0.5", hparams=aa_params),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    train_ds = ImageFolder(data_args.train_dir, transform=_train_transforms)
    eval_ds = ImageFolder(data_args.validation_dir, transform=_val_transforms)

    logger.info(f" loaded train_dataset length is: {len(train_ds)}")
    logger.info(f" loaded eval_dataset length is: {len(eval_ds)}")

    label2id = {}
    id2label = {}
    labels = train_ds.classes

    for key, index in train_ds.class_to_idx.items():
        label2id[key] = str(index)
        id2label[str(index)] = key

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # classes_weight = {}
    # total_samples = len(train_ds)
    # classes = train_ds.classes

    # for class_name in classes:
    #     classes_weight[class_name] = 1 - (len(list(Path(data_args.train_dir, class_name).glob('*.*'))) / total_samples)

    # class_weight = torch.tensor(list(classes_weight.values()), dtype=torch.float, device=torch.device('cuda:0'))

    # class WeightedLossTrainer(Trainer):
    #     def compute_loss(self, model, inputs, return_outputs=False):
    #         labels = inputs.pop("labels")

    #         outputs = model(**inputs)

    #         logits = outputs.get('logits')
    #         loss_func = CrossEntropyLoss(weight=class_weight)
    #         loss = loss_func(logits, labels)

    #         return (loss, outputs) if return_outputs else loss

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    # # ========= Visualize batch
    # from torchvision.utils import make_grid, save_image
    # import matplotlib.pylab as plt
    # import math

    # n_images = 25
    # batch = next(iter(trainer.get_train_dataloader()))
    # img_grid = make_grid(batch['pixel_values'][:n_images,...], normalize=True, pad_value=0.5, nrow=int(math.sqrt(n_images)))
    # save_image(img_grid, f"sample_of_inputs.jpg")

    # # The above code is converting an image grid from one format to another and saving it as a JPEG
    # # file.
    # # grid = img_grid.permute(1, 2, 0).mul(255.).byte()
    # # Image.fromarray(grid.numpy()).save(f"sample_of_inputs.jpg")
    # exit()

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_ds)
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_ds)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    # kwargs = {
    #     "finetuned_from": model_args.model_name_or_path,
    #     "tasks": "image-classification",
    #     "dataset": data_args.dataset_name,
    #     "tags": ["image-classification", "vision"],
    # }
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
