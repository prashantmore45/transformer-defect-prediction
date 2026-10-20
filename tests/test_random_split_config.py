"""The random-split config must differ from the M7 config only in run_name and cache_dir."""

from sdp.config import PROJECT_ROOT
from sdp.training.config import DEFAULT_TRAIN_CONFIG_PATH, TrainConfig

RANDOM_SPLIT_CONFIG = PROJECT_ROOT / "configs" / "train_random_split.yaml"


def test_random_split_config_differs_only_in_run_name_and_cache_dir() -> None:
    base = TrainConfig.from_yaml(DEFAULT_TRAIN_CONFIG_PATH).to_dict()
    other = TrainConfig.from_yaml(RANDOM_SPLIT_CONFIG).to_dict()
    assert {key for key in base if base[key] != other[key]} == {"run_name", "cache_dir"}


def test_random_split_config_points_at_the_random_split_cache() -> None:
    other = TrainConfig.from_yaml(RANDOM_SPLIT_CONFIG)
    assert other.cache_dir.replace("\\", "/").endswith("tokenized_9class_random")
    assert other.run_name == "codebert-9class-random-split-v1"
