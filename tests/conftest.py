import pytest

from creditrisk.config import Config
from creditrisk.data import load_raw, prepare


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config()


@pytest.fixture(scope="session")
def dataset(cfg):
    return prepare(load_raw(cfg.data_path), cfg)
