from pathlib import Path

import pandas as pd
import pickle
import typer
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from hw2.config import (CONFIG_DIR, CONFIG_FILE_NAME, FEATURES_FILE, LABELS_FILE, MODEL_CONFIG_DIR, MODEL_EXTENSION, 
                        MODEL_LOG_REG, MODEL_RANDOM_FOREST, MODEL_DECISION_TREE, MODELS_DIR, PARAMS_PATH,
                        PROCESSED_DATA_DIR, DataConfig, DecisionTreeConfig,
                        LogisticRegressionConfig, RandomForestConfig,
                        get_model_config, load_yaml)

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    config_path: Path = CONFIG_DIR / CONFIG_FILE_NAME,
    features_path: Path = PROCESSED_DATA_DIR / FEATURES_FILE,
    labels_path: Path = PROCESSED_DATA_DIR / LABELS_FILE,
    model_type: str = MODEL_LOG_REG,
    # -----------------------------------------
):
    with open(config_path) as cfg_file:
        load_config = yaml.safe_load(cfg_file)
    data_cfg = DataConfig(**load_config)

    params = load_yaml(PARAMS_PATH)
    files = params["files"]
    model_config_names = files.get("model_configs", {})

    match (model_type):
        case str(MODEL_LOG_REG):
            config_file_name = model_config_names.get(MODEL_LOG_REG, f"{MODEL_LOG_REG}.yaml")
            model_config_path = MODEL_CONFIG_DIR / config_file_name
            parameters = get_model_config(model_config_path, LogisticRegressionConfig)
            model = LogisticRegression(
                penalty=parameters.penalty,
                max_iter=parameters.max_iter,
                random_state=data_cfg.random_state,
            )
        case str(MODEL_RANDOM_FOREST):
            config_file_name = model_config_names.get(MODEL_RANDOM_FOREST, f"{MODEL_RANDOM_FOREST}.yaml")
            model_config_path = MODEL_CONFIG_DIR / config_file_name
            parameters = get_model_config(model_config_path, RandomForestConfig)
            model = RandomForestClassifier(
                n_estimators=parameters.n_estimators,
                max_depth=parameters.max_depth,
                min_samples_split=parameters.min_samples_split,
                min_samples_leaf=parameters.min_samples_leaf,
                random_state=data_cfg.random_state,
            )
        case str(MODEL_DECISION_TREE):
            config_file_name = model_config_names.get(MODEL_DECISION_TREE, f"{MODEL_DECISION_TREE}.yaml")
            model_config_path = MODEL_CONFIG_DIR / config_file_name
            parameters = get_model_config(model_config_path, DecisionTreeConfig)
            model = DecisionTreeClassifier(
                criterion=parameters.criterion,
                max_depth=parameters.max_depth,
                min_samples_split=parameters.min_samples_split,
                min_samples_leaf=parameters.min_samples_leaf,
                random_state=data_cfg.random_state,
            )
        case _:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Training {model_type} model...")

    x, y = pd.read_csv(features_path), pd.read_csv(labels_path)
    model.fit(x, y)
    logger.info("Model trained...")

    model_path = MODELS_DIR / f"{model_type}{MODEL_EXTENSION}"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}.")

    # -----------------------------------------


if __name__ == "__main__":
    app()
