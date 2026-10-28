from pathlib import Path

import numpy as np
import pandas as pd


class RandomRetriever:

    DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

    def __init__(self, lang: str, data_dir: Path = DATA_DIR, seed: int = 42) -> None:
        self.lang = lang
        self.train_df = pd.read_csv(data_dir/ f"{lang}_train.csv")
        self.rng = np.random.default_rng(seed)


    def random_retrieval(self, k: int = 5):
        examples = self.train_df.sample(n = k, random_state = self.rng)
        return examples.to_dict("records")
