import yaml

from pathlib import Path
from hw2.config import get_model_config


def test_get_model_config(tmp_path: Path):
    data = {"max_depth": 3, "n_estimators": 10, "random_state": 42}
    yaml_path = tmp_path / "config.yaml"
    yaml.dump(data, open(yaml_path, "w"))

    class DummyCfg:
        def __init__(self, max_depth, n_estimators, random_state):
            self.max_depth = max_depth
            self.n_estimators = n_estimators
            self.random_state = random_state

    cfg = get_model_config(yaml_path, DummyCfg)
    assert isinstance(cfg, DummyCfg)
    assert cfg.max_depth == 3 and cfg.random_state == 42
