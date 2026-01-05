
import os
from datasets import load_dataset
import random

def prepare_bicameral_data():
    print("[*] Fetching Logic Manifold (M_L) Data...")
    # GSM8K for Math Reasoning
    gsm8k = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    # MBPP for Code Logic
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train", streaming=True)

    print("[*] Fetching Creative Manifold (M_C) Data...")
    # Tiny Stories for Narrative Flow
    stories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    logic_data = []
    creative_data = []

    # Collect 500 high-quality logic samples
    print("    - Processing Logic samples...")
    gsm_iter = iter(gsm8k)
    mbpp_iter = iter(mbpp)

    for _ in range(250):
        try:
            sample = next(gsm_iter)
            # Format as CoT: Question + Answer
            logic_data.append(f"Question: {sample['question']}\nAnswer: {sample['answer']}")
        except StopIteration: break

    for _ in range(250):
        try:
            sample = next(mbpp_iter)
            logic_data.append(f"Task: {sample['prompt']}\nCode:\n{sample['code']}")
        except StopIteration: break

    # Collect 500 creative samples
    print("    - Processing Creative samples...")
    story_iter = iter(stories)
    for _ in range(500):
        try:
            sample = next(story_iter)
            creative_data.append(sample['text'])
        except StopIteration: break

    # Save to local files for the training script
    data_dir = "/home/ty/Repositories/ai_workspace/Bicameral_Manifold_GMoE/data/processed"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(os.path.join(data_dir, "logic_manifold.txt"), "w") as f:
        f.write("\n---SAMPLE_SEP---\n".join(logic_data))

    with open(os.path.join(data_dir, "creative_manifold.txt"), "w") as f:
        f.write("\n---SAMPLE_SEP---\n".join(creative_data))

    print(f"[SUCCESS] Ingested {len(logic_data)} Logic samples and {len(creative_data)} Creative samples.")

if __name__ == "__main__":
    prepare_bicameral_data()
