import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import csv

METADATA_CSV = "./images/images.csv"           # Your metadata file
IMAGE_ROOT = "./images/small"             # Folder containing 00/, 0a/, ff/, etc.
OUTPUT_CSV = "./data/vqa_dataset.csv"     # Output dataset (auto-created)
NUM_SAMPLES = 1000                      # Set to None to use all rows

def generate_qa_pair(image: Image.Image):
    """
    Simulates visual question-answer generation.
    Replace with Gemini or other LLM in production.
    """
    return "What is the color of the object?", "Red"

def curate_dataset():
    df = pd.read_csv(METADATA_CSV)
    if NUM_SAMPLES:
        df = df.head(NUM_SAMPLES)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "question", "answer"])

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating VQA"):
            image_id = row["image_id"]
            relative_path = row["path"]
            full_path = os.path.join(IMAGE_ROOT, relative_path)

            try:
                image = Image.open(full_path).convert("RGB")
            except Exception as e:
                print(f"❌ Skipping {image_id}: {e}")
                continue

            try:
                question, answer = generate_qa_pair(image)
                question = question.strip()
                answer = answer.strip().split()[0]  # Only single-word answers
                writer.writerow([image_id, question, answer])
            except Exception as e:
                print(f"⚠️ Could not generate QA for {image_id}: {e}")

    print(f"✅ Done. VQA dataset saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    curate_dataset()
