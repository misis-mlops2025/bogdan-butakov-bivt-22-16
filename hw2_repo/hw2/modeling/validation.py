from pathlib import Path

import joblib
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import log_loss

from hw2.config import (CONFIG_DIR, MODELS_DIR, PROCESSED_DATA_DIR, DataConfig,
                        load_config)

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    config_path: Path = CONFIG_DIR / "data_config.yaml",
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv"
    # -----------------------------------------
):
    loaded_config = load_config(config_path)
    data_cfg = DataConfig(**loaded_config)

    x_test, y_test = pd.read_csv(features_path), pd.read_csv(labels_path)
    logger.info('Test features and labels loaded')

    model = joblib.load(MODELS_DIR / f"{data_cfg.model_type}.joblib")
    logger.info(f'Model {data_cfg.model_type} loaded')

    predictions = model.predict(x_test)

    loss = log_loss(y_test, predictions)
    logger.info(f"Loss: {loss}")


if __name__ == "__main__":
    app()
