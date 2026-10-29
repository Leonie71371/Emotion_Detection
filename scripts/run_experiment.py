from pathlib import Path

import pandas as pd

from src.model import Model
from src.prompting import Prompting
from src.retrieval.random_retriever import RandomRetriever


class Main:

    DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

    def __init__(self, lang: str, data_dir: Path = DATA_DIR, seed: int = 42) -> None:
        self.lang = lang
        self.dev_df = pd.read_csv(data_dir / f"{lang}_dev.csv")
        self.seed = seed

    def main(self):
        model = Model()
        model.load_model()

        random_retriever = RandomRetriever(lang = self.lang, seed = self.seed)

        prompting = Prompting()

        dev_file = self.dev_df.sample(3, random_state=1)

        results = []

        for _, row in dev_file.iterrows():
            test_text = row["text"]
            examples = random_retriever.random_retrieval(k = 3)

            messages = prompting.build_message(test_text, examples)

            answer = model.prompt(messages = messages)

            gold = prompting.labels_to_text(row)

            results.append({"test_text": test_text, "answer": answer, "gold": gold})

            print(messages[-1]["content"])
            print("Answer:", answer)
            print("Gold:", gold, "\n")


if __name__ == "__main__":
    Main(lang = "deu", seed = 12).main()