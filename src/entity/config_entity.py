from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
	train_data_path: str
	test_data_path: str
	raw_data_path: str 
	source_data_path: str
	

