from pathlib import Path

import pandas as pd
import pickle
import typer
from loguru import logger
from sklearn.metrics import log_loss

from hw2.config import (CONFIG_DIR, CONFIG_FILE_NAME, MODEL_EXTENSION, MODEL_LOG_REG, MODELS_DIR, PROCESSED_DATA_DIR, TEST_FEATURES_FILE, TEST_LABELS_FILE)

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / TEST_FEATURES_FILE,
    labels_path: Path = PROCESSED_DATA_DIR / TEST_LABELS_FILE,
    model_type: str = MODEL_LOG_REG,
    # -----------------------------------------
):

    x_test, y_test = pd.read_csv(features_path), pd.read_csv(labels_path)
    logger.info('Test features and labels loaded')

    model_path = MODELS_DIR / f"{model_type}{MODEL_EXTENSION}"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f'Model {model_type} loaded')

    predictions = model.predict(x_test)

    loss = log_loss(y_test, predictions)
    logger.info(f"Loss: {loss}")


if __name__ == "__main__":
    app()
