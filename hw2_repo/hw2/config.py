from pathlib import Path
from typing import Literal, TypeVar

import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file if it exists
load_dotenv()


class DataConfig(BaseModel):
    n_samples: int = Field(1000, ge=10)
    n_features: int = Field(20, ge=1)
    n_informative: int = Field(5, ge=0, description='cnt informative features')
    n_redundant: int = Field(0, ge=0, description='cnt redundant features')
    random_state: int = Field(42, ge=0, description='seed')
    test_size: float = 0.2
    model_type: str


class TrainConfig(BaseModel):
    test_size: float = Field(0.2, ge=0.0, le=0.5,
                             description="test sample percentage")
    n_jobs: int = Field(1, description="parallelism")


class ModelConfig(BaseModel):
    model_type: str
    random_state: int | None = Field(42, ge=0)


class LogisticRegressionConfig(ModelConfig):
    model_type: Literal['logistic'] = 'logistic'
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
    model_type: Literal['random_forest'] = 'random_forest'
    n_estimators: int = Field(100, ge=1, description='cnt trees in forest')
    max_depth: int | None = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=1)
    min_samples_leaf: int = Field(1, ge=1)


class DecisionTreeConfig(ModelConfig):
    model_type: Literal['decision_tree'] = 'decision_tree'
    criterion: str = 'gini'
    max_depth: int | None = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=1)
    min_samples_leaf: int = Field(1, ge=1)


T = TypeVar("T", bound="ModelConfig")


def get_model_config(model_cfg_path: Path, model_type: T) -> T:
    with open(model_cfg_path) as config_file:
        loaded_config = yaml.safe_load(config_file)
    return model_type(**loaded_config)


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

CONFIG_DIR = PROJ_ROOT / "config"
MODEL_CONFIG_DIR = CONFIG_DIR / "model_config"
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
