from peft import get_peft_model, LoraConfig, TaskType
from transformers import Blip2ForConditionalGeneration, Blip2Processor, TrainingArguments, Trainer
from datasets import load_dataset, Dataset

# Load your dataset
df = pd.read_csv("data/vqa_dataset.csv")
dataset = Dataset.from_pandas(df)

# Load model
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

# Define processor and tokenization function
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
def tokenize(example):
    image = Image.open(f"data/abo_images/{example['image_id']}").convert('RGB')
    inputs = processor(images=image, text=example['question'], return_tensors="pt")
    inputs['labels'] = processor.tokenizer(example['answer'], return_tensors="pt")['input_ids']
    return inputs

# Fine-tune with Huggingface Trainer (simplified)
training_args = TrainingArguments(
    output_dir="./lora_blip2",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset.map(tokenize),
)
trainer.train()
model.save_pretrained("models/lora_finetuned_blip2/")

