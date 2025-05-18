from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

df = pd.read_csv("data/vqa_dataset.csv")
correct = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    image = Image.open(f"data/abo_images/{row['image_id']}").convert('RGB')
    question = row['question']
    answer = row['answer'].lower().strip()

    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    pred = processor.decode(out[0], skip_special_tokens=True).lower().strip()

    if pred == answer:
        correct += 1

accuracy = correct / len(df)
print(f"Zero-shot BLIP-2 Accuracy: {accuracy:.2%}")

