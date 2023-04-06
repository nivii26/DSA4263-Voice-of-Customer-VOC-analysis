from yaml import safe_load
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = ROOT_DIR / "models" / "tm"
CUR_DIR = Path(__file__).parent


def fetch_config_from_yaml(cfg_file="config.yml"):
    """Parse YAML containing the package configuration."""
    cfg_path = CUR_DIR / cfg_file
    if cfg_path.exists():
        with open(str(cfg_path), "r", encoding="utf-8") as conf_file:
            parsed_config = safe_load(conf_file)
            return parsed_config

    raise OSError(f"Did not find config file at path: {cfg_path}")


def parse_yaml():
    """Constrain values from yaml"""
    config = fetch_config_from_yaml()
    if len(config["topic_map"]) != config["num_topics"]:
        raise ValueError("Length of topic map should be the same as num_topics value")
    if config["model_name"] not in ["lda", "nmf", "lsa"]:
        raise ValueError("model_name should be either lda, nmf or lsa")
    if config["preprocess_type"] not in ["bow", "tfidf"]:
        raise ValueError("preprocess_type should either be bow or tfidf")
    return config


CONFIG = parse_yaml()

# if __name__ == "__main__":
#     print(CONFIG)
