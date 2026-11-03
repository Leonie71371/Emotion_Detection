import numpy as np
import pandas as pd


class RandomRetriever:

    def __init__(self, pool: pd.DataFrame , seed: int = 42) -> None:
        self.pool = pool
        self.rng = np.random.default_rng(seed)


    def random_retrieval(self, query: str | None = None, k: int = 5) -> list[dict]:
        return self.pool.sample(n = k, random_state = self.rng).to_dict("records")
