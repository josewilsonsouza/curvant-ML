import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.curve_detection import detectar_curvas


def plot_trajeto_com_curvatura(
    df: pd.DataFrame,
    sigma: float = 2,
    limiar: float = 30,
    limite_raio: float = 50,
    save_fig: bool = False,
    name_file: str | None = None,
) -> None:
    """
    Plota o trajeto completo destacando curvas detectadas e raios de curvatura
    menores que limite_raio (em metros).
    """
    df_u = detectar_curvas(df, sigma=sigma, limiar=limiar)[1]

    x = df_u['x']
    y = df_u['y']
    dxdt = np.gradient(x)
    dydt = np.gradient(y)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color='blue', linewidth=2, label='Trajeto Completo')
    ax.scatter(
        x[df_u['curva']], y[df_u['curva']],
        color='red', s=30, label='Curvas Detectadas',
    )

    raio_arr = df_u['raio_curvatura'].values
    sinal_arr = df_u['sinal_curvatura'].values
    x_arr = x.values
    y_arr = y.values

    for i in range(len(x_arr)):
        raio = raio_arr[i]
        if raio < limite_raio:
            nx, ny = -dydt[i], dxdt[i]
            norma = np.sqrt(nx**2 + ny**2)
            if norma != 0:
                nx, ny = nx / norma * sinal_arr[i], ny / norma * sinal_arr[i]
            ax.plot(
                [x_arr[i], x_arr[i] + nx * raio],
                [y_arr[i], y_arr[i] + ny * raio],
                color='green', linewidth=1,
            )

    ax.set_title(f'Trajeto com Curvas e Raios de Curvatura (limite < {limite_raio} m)', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid()

    if save_fig and name_file:
        plt.savefig(f'{name_file}.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plotar_trajeto_conducao(df: pd.DataFrame, tipo: str = 'conducao') -> None:
    """
    Plota o mapa do trajeto (X × Y) com pontos perigosos em vermelho
    e o perfil de velocidade ao lado.
    """
    pts_perigosos = (
        df[df[tipo] == 'Perigosa'] if tipo == 'conducao' else df[df[tipo]]
    )

    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    ax[0].plot(-df['x'], -df['y'], label='Trajeto', color='blue', alpha=0.5)
    ax[1].plot(-df['x'], df['vehicle_speed'], label='vel (km/h)')

    if not pts_perigosos.empty:
        ax[0].scatter(
            -pts_perigosos['x'], -pts_perigosos['y'],
            color='red', label=f'Condução Perigosa ({tipo})', s=50, marker='.',
        )

    name_traj = df['id_route'].iloc[0]
    ax[0].set_title(f'Condução Perigosa Destacada\n{name_traj}')
    ax[0].set_xlabel('Posição X')
    ax[0].set_ylabel('Posição Y')
    ax[1].set_xlabel('Posição X')
    ax[1].set_ylabel('Velocidade (km/h)')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].legend()
    ax[1].grid(True)
    plt.show()


def plotar_curva_treinamento(history) -> None:
    """Plota acurácia de treino e validação ao longo das épocas (para modelos Keras)."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Treinamento', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validação', color='orange')
    plt.title('Acurácia por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.show()
