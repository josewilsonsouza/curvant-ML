import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega os dados do trajeto a partir de um arquivo CSV ou Parquet ou Excel.
    Supomos que seja um arquivo concatenado de todos os trajetos.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Formato de arquivo não suportado.")

def contar_curvas(dfs_curves: pd.DataFrame) -> dict[str, int]:
    """Retorna dicionário {id_route: número de curvas detectadas}."""
    contagem = {}
    for traj in dfs_curves['id_route'].unique():
        df = dfs_curves.query(f'id_route == "{traj}"')
        curva = df['curva'].tolist()
        n, em_sequencia = 0, False
        for val in curva:
            if val and not em_sequencia:
                n += 1
                em_sequencia = True
            elif not val:
                em_sequencia = False
        contagem[traj] = n
    return contagem