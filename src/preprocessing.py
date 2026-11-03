from pathlib import Path

import pandas as pd


class Preprocessing:

    def __init__(self, data_path: str = "../data/raw/") -> None:
        self.langs = ["afr", "deu", "ron"]
        self.splits = ["train", "dev", "test"]
        self.data_path = data_path
        self.data = self.read_files()

    def read_files(self):
        data = {}
        for lang in self.langs:
            for split in self.splits:
                data[lang, split] = pd.read_csv(self.data_path + f"{lang}_{split}.csv")
        return data

    def remove_nan(self):
        pass

    def remove_id(self):
        for key, df in self.data.items():
            self.data[key] = df.drop(columns = ["id"])

    def save_files(self, out_path="../data/processed/"):
        out_dir = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for (lang, split), df in self.data.items():
            df.to_csv(out_dir / f"p_{lang}_{split}.csv", index=False)