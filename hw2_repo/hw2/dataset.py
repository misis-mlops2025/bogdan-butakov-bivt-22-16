from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from loguru import logger
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from hw2.config import CONFIG_DIR, PROCESSED_DATA_DIR, DataConfig

app = typer.Typer()


def generate_dataset(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    x, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        random_state=cfg.random_state,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        pd.DataFrame(x),
        pd.DataFrame(y.ravel()),
        test_size=cfg.test_size,
        random_state=cfg.random_state
    )
    return x_train, x_test, y_train, y_test


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    output_path: Path = PROCESSED_DATA_DIR,
    config_path: Path = CONFIG_DIR / "data_config.yaml"
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")

    with open(config_path) as cfg_file:
        load_config = yaml.safe_load(cfg_file)
    data_cfg = DataConfig(**load_config)

    x_train, x_test, y_train, y_test = generate_dataset(data_cfg)

    x_train.to_csv(output_path / "features.csv", index=False)
    y_train.to_csv(output_path / "labels.csv", index=False)
    x_test.to_csv(output_path / "test_features.csv", index=False)
    y_test.to_csv(output_path / "test_labels.csv", index=False)

    logger.info("Getting features and labels done...")

    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
