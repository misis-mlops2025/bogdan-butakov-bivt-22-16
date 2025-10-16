from pathlib import Path

import joblib
import pandas as pd
import typer
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from hw2.config import (CONFIG_DIR, MODEL_CONFIG_DIR, MODELS_DIR,
                        PROCESSED_DATA_DIR, DataConfig, DecisionTreeConfig,
                        LogisticRegressionConfig, RandomForestConfig,
                        get_model_config)

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    config_path: Path = CONFIG_DIR / "data_config.yaml",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    # -----------------------------------------
):
    with open(config_path) as cfg_file:
        load_config = yaml.safe_load(cfg_file)
    data_cfg = DataConfig(**load_config)

    match (data_cfg.model_type):
        case "log_reg":
            model_config_path = MODEL_CONFIG_DIR / "log_reg.yaml"
            parameters = get_model_config(model_config_path, LogisticRegressionConfig)
            model = LogisticRegression(
                penalty=parameters.penalty,
                max_iter=parameters.max_iter,
                random_state=data_cfg.random_state,
            )
        case "random_forest":
            model_config_path = MODEL_CONFIG_DIR / "random_forest.yaml"
            parameters = get_model_config(model_config_path, RandomForestConfig)
            model = RandomForestClassifier(
                n_estimators=parameters.n_estimators,
                max_depth=parameters.max_depth,
                min_samples_split=parameters.min_samples_split,
                min_samples_leaf=parameters.min_samples_leaf,
                random_state=data_cfg.random_state,
            )
        case "decision_tree":
            model_config_path = MODEL_CONFIG_DIR / "decision_tree.yaml"
            parameters = get_model_config(model_config_path, DecisionTreeConfig)
            model = DecisionTreeClassifier(
                criterion=parameters.criterion,
                max_depth=parameters.max_depth,
                min_samples_split=parameters.min_samples_split,
                min_samples_leaf=parameters.min_samples_leaf,
                random_state=data_cfg.random_state,
            )

    logger.info(f"Training {data_cfg.model_type} model...")

    x, y = pd.read_csv(features_path), pd.read_csv(labels_path)
    model.fit(x, y)
    logger.info("Model trained...")

    joblib.dump(model, MODELS_DIR / f"{data_cfg.model_type}.joblib")
    logger.info(f"Model saved to {MODELS_DIR / f'{data_cfg.model_type}.joblib'}.")

    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
