from pathlib import Path
from typing import Literal, TypeVar

import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, field_validator

MODEL_LOG_REG = 'log_reg'
MODEL_RANDOM_FOREST = 'random_forest'
MODEL_DECISION_TREE = 'decision_tree'

FEATURES_FILE = 'features.csv'
LABELS_FILE = 'labels.csv'
TEST_FEATURES_FILE = 'test_features.csv'
TEST_LABELS_FILE = 'test_labels.csv'
CONFIG_FILE_NAME = 'data_config.yaml'
MODEL_EXTENSION = '.pkl'

PATH_DEFAULTS = {
    "config_dir": "config",
    "model_config_dir": "config/model_config",
    "data_dir": "data",
    "processed_dir": "data/processed",
    "models_dir": "models",
    "reports_dir": "reports",
    "figures_dir": "reports/figures",
}

# Load environment variables from .env file if it exists
load_dotenv()


class DataConfig(BaseModel):
    n_samples: int = Field(1000, ge=10)
    n_features: int = Field(20, ge=1)
    n_informative: int = Field(5, ge=0, description='cnt informative features')
    n_redundant: int = Field(0, ge=0, description='cnt redundant features')
    random_state: int = Field(42, ge=0, description='seed')
    test_size: float = 0.2


class TrainConfig(BaseModel):
    test_size: float = Field(0.2, ge=0.0, le=0.5,
                             description="test sample percentage")
    n_jobs: int = Field(1, description="parallelism")


class ModelConfig(BaseModel):
    model_type: str
    random_state: int | None = Field(42, ge=0)


class LogisticRegressionConfig(ModelConfig):
    model_type: str = MODEL_LOG_REG
    c: float = Field(1.0, gt=0, description='Inverse of regularization. Smaller -> stronger reg.')
    penalty: str = 'l2'
    max_iter: int = Field(1000, ge=1)

    @field_validator('c')
    @classmethod
    def validate_c(cls, val):
        if val <= 0:
            raise ValueError('c must be > 0')
        return val


class RandomForestConfig(ModelConfig):
    model_type: str = MODEL_RANDOM_FOREST
    n_estimators: int = Field(100, ge=1, description='cnt trees in forest')
    max_depth: int | None = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=1)
    min_samples_leaf: int = Field(1, ge=1)


class DecisionTreeConfig(ModelConfig):
    model_type: str = MODEL_DECISION_TREE
    criterion: str = 'gini'
    max_depth: int | None = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=1)
    min_samples_leaf: int = Field(1, ge=1)


def load_config(cfg_path: Path):
    with open(cfg_path) as config_file:
        return yaml.safe_load(config_file)
    

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


T = TypeVar("T", bound="ModelConfig")


def get_model_config(model_cfg_path: Path, model_type: T) -> T:
    loaded_config = load_config(model_cfg_path)
    return model_type(**loaded_config)


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
PARAMS_PATH = PROJ_ROOT / "params.yaml"
params = load_yaml(PARAMS_PATH)
paths = params.get("paths", {})

CONFIG_DIR = PROJ_ROOT / paths.get("config_dir", PATH_DEFAULTS["config_dir"])
MODEL_CONFIG_DIR = PROJ_ROOT / paths.get("model_config_dir", PATH_DEFAULTS["model_config_dir"])
DATA_DIR = PROJ_ROOT / paths.get("data_dir", PATH_DEFAULTS["data_dir"])
PROCESSED_DATA_DIR = PROJ_ROOT / paths.get("processed_dir", PATH_DEFAULTS["processed_dir"])
MODELS_DIR = PROJ_ROOT / paths.get("models_dir", PATH_DEFAULTS["models_dir"])
REPORTS_DIR = PROJ_ROOT / paths.get("reports_dir", PATH_DEFAULTS["reports_dir"])
FIGURES_DIR = PROJ_ROOT / paths.get("figures_dir", PATH_DEFAULTS["figures_dir"])

logger.info(f"Loaded paths from params.yaml: {paths}")
