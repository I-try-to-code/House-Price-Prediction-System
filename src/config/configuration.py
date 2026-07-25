from pathlib import Path

from src.utils.common import read_yaml

from src.entity.config_entity import DataIngestionConfig

class Configuration:
    def __init__(self):
        self.config = read_yaml(Path("config/config.yaml"))
        Path(self.config["artifacts_root"]).mkdir(parents=True, exist_ok=True)

    def get_data_ingestion_config(self):
        config = self.config["data_ingestion"]
        return DataIngestionConfig(
            raw_data_path=config["raw_data_path"],
            train_data_path=config["train_data_path"],
            test_data_path=config["test_data_path"],
            source_data_path=config["source_data_path"]
        )

    def get_schema(self):
        return self.config["COLUMNS"]