from omegaconf import OmegaConf
import fire
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os
import shutil
import wandb
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
import random

from accelerate import Accelerator
from utils import make_supervised_data_module
import torch.distributed as dist
import time


def main(
        config_path=None
):
    if config_path is None:
        raise ValueError("config_path is required. Example: --config_path configures/ft_Qwen2.5-7B-ActiveIndex-Instruct.yml")
    # get local rank
    device_index = Accelerator().process_index

    conf = OmegaConf.load(config_path)

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
                time.sleep(10)
        else:
            print("wandb init failed after 5 attempts, exiting...")
            use_wandb = False

    # set seed
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)


    model_name = conf.model_name
    use_lora = False if "fm" in conf.exp_name else True
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = 'left'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    ) if use_lora else None
    torch_dtype = None if use_lora else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map={"": Accelerator().process_index},
    )
    if conf.get("adapter_id", None) is not None:
        # if "fm" not in conf.adapter_id:
        #     model_with_adapter = PeftModel.from_pretrained(
        #         model, model_id=conf.adapter_id, quantization_config=bnb_config
        #     )
        #     model = model_with_adapter.merge_and_unload()
        # else:
        model = AutoModelForCausalLM.from_pretrained(
            conf.adapter_id,
            torch_dtype=torch_dtype,
            device_map={"": Accelerator().process_index},
        )
        if not use_lora:
            for name, param in model.named_parameters():
                param.requires_grad = True
    model.config.use_cache = False
    # More info: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1
    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    ) if use_lora else None

    with open(conf['train_data_path'], 'r') as f:
        train_data = [json.loads(line) for i, line in enumerate(f) if i < conf.sample_num or conf.sample_num == -1]
    with open(conf['dev_data_path'], 'r') as f:
        dev_data = [json.loads(line) for line in f]


    if conf.sample_num > 0:
        train_data = train_data[:conf.sample_num]
        dev_data = dev_data[:conf.sample_num]
    print(f"train data size: {len(train_data)}")
    print(f"dev data size: {len(dev_data)}")

    # shuffle data
    if conf.get("shuffle_data", True):
        random.shuffle(train_data)
        random.shuffle(dev_data)

    data_module = make_supervised_data_module(tokenizer=tokenizer, train_data=train_data,
                                              eval_data=dev_data, max_length=conf.get("max_length", 1024))
    save_dir = os.path.join(conf.save_dir, conf.exp_name, conf.version)

    # calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    fsdp_config = None
    if conf.get("fsdp", False):
        if conf.get("fsdp_config", None) is None:
            fsdp_config = {"transformer_layer_cls_to_wrap": "Qwen2DecoderLayer"}
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
        seed=conf.seed,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=conf.get("logging_steps", 100),
        run_name=conf.version,
        report_to="wandb" if use_wandb else "none",
        warmup_steps=conf.get("warmup_steps", 0),
        gradient_accumulation_steps=conf.get("gradient_accumulation_steps", 1),
        ddp_find_unused_parameters=False,
        lr_scheduler_type=conf.get("lr_scheduler_type", "linear"),
        optim=conf.get("optimizer", "adamw_torch"),
        max_grad_norm=conf.get("max_grad_norm", 1.0),
        save_on_each_node=True,
        save_only_model=True if conf.get("fsdp", False) is False else False,
        save_total_limit=1,
        fsdp=conf.get("fsdp", False),
        fsdp_config=fsdp_config,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
        peft_config=peft_parameters,
        tokenizer=tokenizer,
        args=args,
    )

    trainer.train()


    if dist.is_initialized():
        trainer.accelerator.wait_for_everyone()

    if device_index == 0:
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

        save_dir = os.path.join(save_dir, "checkpoint-final")
        print(f"Saving last checkpoint of the model to {save_dir}")
        trainer.model.save_pretrained(save_dir)

    if use_wandb:
        wandb.finish()




if __name__ == '__main__':
    fire.Fire(main)
