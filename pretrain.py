from omegaconf import OmegaConf
import fire
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd

import os
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from accelerate import Accelerator
from datasets import Dataset
import datetime

from mem_dataset import ReplayDataset, _MemmapDataset as MemmapDataset
from utils import read_jsonl
from time import sleep
import shutil





def main(
        config_path=None
):
    if config_path is None:
        raise ValueError("config_path is required. Example: --config_path configures/pretrain_Qwen2.5-7B-ActiveIndex.yml")

    # if the number of GPUs is greater f than 1, we need to use DDP
    if torch.cuda.device_count() > 1:
        # torchrun exports these; fall back to single‑GPU defaults for debugging
        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))

        torch.cuda.set_device(local_rank)  # tell CUDA which GPU *this* rank owns

        torch.distributed.init_process_group(  # never pass -1 / -1
            backend="nccl",
            init_method="env://",  # read addr/port from env set by torchrun
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(hours=1),
        )
        # torch.distributed.init_process_group("nccl", init_method=None, timeout=datetime.timedelta(seconds=3600),
        #                                  world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None)
    conf = OmegaConf.load(config_path)

    device_index = Accelerator().process_index
    use_wandb = True if device_index == 0 else False
    project_name = conf.exp_name
    if use_wandb:
        for i in range(5):
            try:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project=project_name,
                    # track hyperparameters and run metadata
                    config={k: v for k, v in conf.items()},
                    name=conf.version,
                    group="DDP",
                )
                break
            except Exception as e:
                print(f"wandb init failed (attempt {i + 1}): {e}")
                sleep(10)
        else:
            print("wandb init failed after 5 attempts, exiting...")
            use_wandb = False

    # set seed
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)


    model_name = conf.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    use_lora = False if "fm" in conf.exp_name else True

    print(device_index)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    ) if use_lora else None
    torch_dtype = None if use_lora else torch.bfloat16
    base_model_id = conf.get("base_model_id", None)
    base_model_id = model_name if base_model_id is None else base_model_id
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map={"": device_index},
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    ) if use_lora else None

    if conf['train_data_path'].endswith(".jsonl"):
        train_data = read_jsonl(conf['train_data_path'])
        dev_data = read_jsonl(conf.get("dev_data_path", conf['train_data_path']))

        if conf.sample_num > 0:
            train_data = train_data[:conf.sample_num]
            dev_data = dev_data[:conf.sample_num]

        # shuffle data
        if conf.get("shuffle_data", True):
            train_data = train_data.sample(frac=1, random_state=conf.seed)
            dev_data = dev_data.sample(frac=1, random_state=conf.seed)

        train_dataset = Dataset.from_pandas(train_data)
        dev_dataset = Dataset.from_pandas(dev_data)

    elif conf['train_data_path'].endswith(".bin"):

        train_dataset = ReplayDataset(block_size=conf.get("max_length", 1024),
                                      bin_file=conf['train_data_path'],
                                      replay_ratio=conf.get("replay_ratio", 0.0),
                                      replay_file=conf.get("replay_data_path", None),
                                      subsample_ratio=conf.get("sample_ratio", 1.0),
                                      index_ratio=conf.get("index_ratio", 0.0),
                                      random_state=conf.get("random_state", 42),
                                      sot_token_id=conf.get("sot_token_id", 128002),
                                      eot_token_id=conf.get("eot_token_id", 128003))
        dev_dataset = MemmapDataset(block_size=conf.get("max_length", 1024),
                                    bin_file=conf.get("dev_data_path", conf['train_data_path']))



    save_dir = os.path.join(conf.save_dir,conf.exp_name, conf.version)
    fsdp_config = None
    if conf.get("fsdp", False):
        if conf.get("fsdp_config", None) is None:
            fsdp_config = {"transformer_layer_cls_to_wrap": "LlamaDecoderLayer"}
        else:
            fsdp_config = conf.get("fsdp_config", None)

    # Define Trainer
    args = SFTConfig(
        output_dir=os.path.join(save_dir),
        evaluation_strategy="steps",
        eval_steps=conf.eval_steps,
        save_steps=conf.save_steps,
        learning_rate=conf.learning_rate,
        per_device_train_batch_size=conf.per_device_train_batch_size,
        per_device_eval_batch_size=conf.get("per_device_eval_batch_size", 8),
        num_train_epochs=conf.num_train_epochs,
        max_steps=conf.get("max_steps", -1),
        seed=conf.seed,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=conf.get("logging_steps", 100),
        run_name=conf.version,
        report_to="wandb" if use_wandb else None,
        warmup_steps=conf.get("warmup_steps", 0),
        gradient_accumulation_steps=conf.get("gradient_accumulation_steps", 1),
        ddp_find_unused_parameters=False,
        lr_scheduler_type=conf.get("lr_scheduler_type", "linear"),
        max_seq_length=conf.get("max_length", 1024),
        dataset_text_field="text",
        optim=conf.get("optimizer", "adamw_torch"),
        fsdp=conf.get("fsdp", False),
        fsdp_config=fsdp_config,
        save_only_model=True if conf.get("fsdp", False) is False else False,
        # save_total_limit=1,
        # log_level="info",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        peft_config=peft_parameters,
        tokenizer=tokenizer,
        args=args,
    )

    # number of forward passes per epoch
    batches_per_epoch = len(trainer.get_train_dataloader())

    # optimizer steps per epoch
    steps_per_epoch = batches_per_epoch // args.gradient_accumulation_steps
    if batches_per_epoch % args.gradient_accumulation_steps != 0:
        steps_per_epoch += 1
    print(f"batches_per_epoch: {batches_per_epoch}, steps_per_epoch: {steps_per_epoch}")
    if conf.get("debug", False):
        raise RuntimeError(f"Debug mode: batches_per_epoch={batches_per_epoch}, steps_per_epoch={steps_per_epoch}")

    trainer.train()
    save_path = os.path.join(save_dir, "checkpoint-final")
    print(f"Saving last checkpoint of the model to {save_path}")
    if Accelerator().is_main_process:
        trainer.model.save_pretrained(save_path)
        if not conf.get("keep_ckpts", False):
            remove_cnt = 0
            for f in os.listdir(save_dir):
                if "final" not in f:
                    path = os.path.join(save_dir, f)
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    remove_cnt += 1
            print(f"Removed {remove_cnt} checkpoints")

    if use_wandb:
        wandb.finish()




if __name__ == '__main__':
    fire.Fire(main)
