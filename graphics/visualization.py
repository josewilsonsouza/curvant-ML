import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import folium

from src.curve_detection import detectar_curvas


def plot_trajeto_com_curvatura(df, sigma=2, limite_raio=100, min_pontos=3,
                                save_fig=False, name_file=None):
    """
    Plota o trajeto completo com as curvas detectadas e os raios de curvatura.
    """
    df_final = detectar_curvas(df, sigma=sigma, limite_raio=limite_raio, min_pontos=min_pontos)

    x = df_final['x'].values
    y = df_final['y'].values
    pontos_curva   = df_final['curva'].values
    raio_curvatura = df_final['raio_curvatura'].values
    sinal_curvatura = df_final['sinal_curvatura'].values
    dxdt = np.gradient(x)
    dydt = np.gradient(y)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color='blue', linewidth=2, label='Trajeto Completo')
    ax.scatter(x[pontos_curva], y[pontos_curva], color='red', s=30, label='Curvas Detectadas')

    for i in range(len(x)):
        if raio_curvatura[i] < limite_raio:
            nx, ny = -dydt[i], dxdt[i]
            norma = np.sqrt(nx**2 + ny**2)
            if norma != 0:
                nx /= norma
                ny /= norma
            nx *= sinal_curvatura[i]
            ny *= sinal_curvatura[i]
            ax.plot([x[i], x[i] + nx * raio_curvatura[i]],
                    [y[i], y[i] + ny * raio_curvatura[i]],
                    color='green', linewidth=1)

    ax.set_title(f'Trajeto com Curvas e Raios de Curvatura (limite < {limite_raio} m)', fontsize=14)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend()
    ax.grid()

    if save_fig and name_file:
        plt.savefig(f'{name_file}.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plot_trip_folium(df, sigma=2, limite_raio=50, name_traj=None):
    """
    Plota o trajeto completo com curvas detectadas em um mapa Folium.
    """
    df_final = detectar_curvas(df, sigma=sigma, limite_raio=limite_raio, name_traj=name_traj)

    if df_final.empty or df_final[['lat', 'lon']].isnull().all().any() or len(df_final) < 2:
        print(f"Não há dados suficientes para plotar o trajeto {name_traj}.")
        return None

    first_valid_point = df_final.dropna(subset=['lat', 'lon']).iloc[0]
    m = folium.Map(location=[first_valid_point['lat'], first_valid_point['lon']], zoom_start=15)

    folium.PolyLine(
        locations=df_final[['lat', 'lon']].values.tolist(),
        color='blue', weight=3, opacity=0.7,
        tooltip=f"Trajeto: {df_final['id_route'].iloc[0]}",
    ).add_to(m)

    for _, row in df_final[df_final['curva']].dropna(subset=['lat', 'lon']).iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3, color='red', fill=True, fill_color='red', fill_opacity=0.7,
            popup=f"Curvatura: {row['curvatura']:.2e}<br>Raio: {row['raio_curvatura']:.2f} m",
        ).add_to(m)

    m.fit_bounds([
        [df_final['lat'].min(), df_final['lon'].min()],
        [df_final['lat'].max(), df_final['lon'].max()],
    ])
    return m


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
