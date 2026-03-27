import yaml


def carregar_config() -> dict:
    """Lê config.yaml da raiz do projeto."""
    with open('config.yaml') as f:
        return yaml.safe_load(f)
