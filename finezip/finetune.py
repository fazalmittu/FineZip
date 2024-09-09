import pprint
from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from utils.finetune_utils import CastOutputToFloat, print_trainable_parameters
from huggingface_hub import login
from transformers import BitsAndBytesConfig

def finetune(model, save_path, dataset_path="datasets/enwik8.txt", block_size=128, epochs=10, r=8, learning_rate=1e-4, batch_size=4):
    # quant_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_4bit_compute_dtype=torch.float16  # Ensure computation type matches input type
    # )
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model, 
        device_map={"":torch.cuda.current_device()},
        # quantization_config=quant_config
    )

    loaded_model = loaded_model.half()

    tokenizer = AutoTokenizer.from_pretrained(model)

    print("Model loaded")

    for param in loaded_model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    loaded_model.gradient_checkpointing_enable()  
    loaded_model.enable_input_require_grads()

    loaded_model.lm_head = CastOutputToFloat(loaded_model.lm_head)    

    config = LoraConfig(
        r=r,
        lora_alpha=32, 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    loaded_model = get_peft_model(loaded_model, config)
    print_trainable_parameters(loaded_model)

    loaded_model.to(torch.cuda.current_device())

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    print("Dataset loaded")

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size, 
        gradient_accumulation_steps=8,
        max_steps=epochs, 
        learning_rate=learning_rate, 
        fp16=True,
        logging_steps=1,
        output_dir="output",
        warmup_steps=500,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=loaded_model, 
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator
    )

    loaded_model.config.use_cache = False  

    trainer.train()
    trainer.save_model(save_path + f"_{epochs}_r{r}") 

    del loaded_model
    del trainer
    torch.cuda.empty_cache()

    print("Finished finetuning")

if __name__ == "__main__":
    login("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") 

    finetune_list = [
        "meta-llama/Meta-Llama-3-8B"
    ]
    epoch_list = [256]

    for model in finetune_list:
        for e in epoch_list:
            finetune(
                model, 
                save_path=f"finetuned_models/10mb_qlora/16_bit/{model.replace('/', '-')}-enwik10mb", 
                dataset_path="datasets/enwik10mb.txt", 
                block_size=128, 
                epochs=e, 
                r=8,
                learning_rate=1e-4, 
                batch_size=16
            )
