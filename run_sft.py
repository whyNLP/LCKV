import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SftScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments
import models

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM,
    setup_chat_format
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        use_flash_attention_2=os.environ.get('LCKV_FLASH_ATTN', False),
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, add_eos_token=True, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token # do not use eos_token, see https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)

    train_dataset = raw_datasets[args.dataset_train_name]
    eval_dataset = raw_datasets[args.dataset_test_name]

    # clean the dataset so that there is no tailing ### Human: at the end of the text
    def remove_tailing(element):
        text = element['text'].split("### Human:")
        text = ''.join([text[0], "### Human:", text[1]])
        return {"text": text}
    
    train_dataset = train_dataset.map(remove_tailing, batched=False, load_from_cache_file=False)

    # This is hard coded for timdettmers/openassistant-guanaco
    # see https://huggingface.co/docs/trl/main/en/sft_trainer#using-tokenids-directly-for-responsetemplate for more details
    instruction_template_with_context = "\n### Human:"  # We added context here: "\n". This is enough for this tokenizer
    instruction_template_ids = tokenizer.encode(instruction_template_with_context, add_special_tokens=False)[2:]

    response_template_with_context = "\n### Assistant:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=response_template_ids,
        response_template=response_template_ids,  
        tokenizer=tokenizer,
        mlm=False
    )

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=args.dataset_text_field,
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            packing=args.packing,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=collator,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)