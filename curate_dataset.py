import os
import csv
import requests
from PIL import Image
from tqdm import tqdm

# Example Gemini prompt wrapper
def generate_question_answer(image_path):
    prompt = "Generate a single-word answer VQA pair for this image."
    # Use Gemini or Ollama API (mocked here)
    question = "What is the color of the bag?"
    answer = "Red"
    return question, answer

def curate_vqa_dataset(image_dir, output_csv):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'question', 'answer'])
        for img in tqdm(image_files):
            q, a = generate_question_answer(os.path.join(image_dir, img))
            writer.writerow([img, q, a])

# Example usage:
# curate_vqa_dataset("data/abo_images/", "data/vqa_dataset.csv")

