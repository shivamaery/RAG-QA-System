import pymupdf
import os
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
path = current_dir + "/test_data"

def extract_text_from_all_pdfs(folder_path):
    full_text = ""

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            doc = pymupdf.open(pdf_path)
            print(f"Processing {filename}, pages: {doc.page_count}")

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                print(f"--- Page {page_num + 1} of {filename} ---")
                full_text += page_text

            doc.close()

    return full_text

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def chunk_text(text, chunk_size=256):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"][0]
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [
        {
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
            "labels": chunk.clone()  # Important: labels = input_ids
        }
        for chunk in chunks
    ]

model_name = "microsoft/phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

pdf_text = extract_text_from_all_pdfs(path)
tokenized_chunks = chunk_text(pdf_text)
dataset = Dataset.from_list(tokenized_chunks)
dataset.set_format(type="torch")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype="auto", device_map="cuda:0")

# Prepare the model for LoRA 4-bit training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir= current_dir + "/LoRA-Tuning-output",
    logging_dir= current_dir + "/LoRA-logs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=20,
    save_strategy="no",
    evaluation_strategy="no",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(current_dir + "/LoRA-Tuned-Model")
tokenizer.save_pretrained(current_dir + "/LoRA-Tuned-Model")