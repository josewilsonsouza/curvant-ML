import yaml

def carregar_config() -> dict:
    """Lê config.yaml da raiz do projeto."""
    with open('config.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)
