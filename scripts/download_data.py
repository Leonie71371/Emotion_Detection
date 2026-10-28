from pathlib import Path

from datasets import load_dataset

output = Path("../data/raw")

for language in ("afr", "deu", "ron"):
    ds = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", language)
    for split, data in ds.items():
        data.to_csv(output/ f"{language}_{split}.csv")