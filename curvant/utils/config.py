import yaml

def carregar_config() -> dict:
    """Lê config.yaml da raiz do projeto."""
    with open('config.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


def carregar_features_config() -> dict:
    """Lê features.yaml da raiz do projeto (governança de features por flag)."""
    try:
        with open('features.yaml', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "features.yaml não encontrado na raiz do projeto. Ele define quais "
            "features cada flag usa (whitelist por flag). Veja docs/TARGETS_E_FEATURES.md."
        )
