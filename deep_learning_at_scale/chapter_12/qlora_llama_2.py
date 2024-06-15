from pathlib import Path

import torch
import typer
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
__all__ = ["app"]


"""
deep-learning-at-scale chapter_12 qlora train
deep-learning-at-scale chapter_12 qlora inference
"""


@app.command()
def train(
    name: str = typer.Option("chapter_12", help="Name of the run"),
    batch_size: int = typer.Option(15, help="Batch size"),
    pretrained_model_uri: str = typer.Option(
        "openlm-research/open_llama_7b_v2", help="Source pretrained model"
    ),
    output: str = typer.Option("qlora_tuned_llama_7b_v2", help="Name of tuned model"),
):
    llama_tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_uri)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    ## Load pretrained base model along with quantisation configurations
    base_model = LlamaForCausalLM.from_pretrained(
        pretrained_model_uri,
        load_in_4bit=True,
        quantization_config=quant_config,
        device_map="auto",
    )

    ## Load 1% of book corpus training dataset
    training_data = load_dataset("bookcorpus", split="train[:1%]")

    ## Configure the Trainer and Supervised Fine Tunning Trainer
    train_params = TrainingArguments(
        output_dir=Path("results_modified"),
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        optim="adamw_hf",
        fp16=True,
        max_steps=-1,
        report_to="tensorboard",
    )
    ## Configure LoRA settings including alpha, dropout and r & the fine tuning trainer
    peft_parameters = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=8, bias="none", task_type="CAUSAL_LM"
    )
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params,
        max_seq_length=512,
    )

    # Commence the fine tuning and save the QLoRA adaptation weights
    fine_tuning.train()
    fine_tuning.model.save_pretrained(output, is_main_process=True)


@app.command()
def inference(
    base_model_uri: str = typer.Option(
        "openlm-research/open_llama_7b_v2", help="Source pretrained model"
    ),
    lora_model_fn: str = typer.Option(
        "qlora_tuned_llama_7b_v2", help="Name of tuned model"
    ),
):
    ## Load the tokenizer and base model in FP16 format
    tokenizer = LlamaTokenizer.from_pretrained(base_model_uri)
    config = PeftConfig.from_pretrained(lora_model_fn)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    ## Load the LoRA adapters
    model = PeftModel.from_pretrained(
        model,
        lora_model_fn,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    ## Conduct the generative inference
    text = "Without QLoRA fine tuning a large Deep Learning model can be daunting."
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        text_gen = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            penalty_alpha=0.6,
            top_k=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        print(
            tokenizer.batch_decode(
                text_gen.detach().cpu().numpy(), skip_special_tokens=True
            )
        )
