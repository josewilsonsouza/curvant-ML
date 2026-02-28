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
    Plota o trajeto completo com curvas detectadas e raios de curvatura em um mapa Folium.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados do trajeto com 'lat' e 'lon'.
        sigma (int): Parâmetro sigma para o filtro gaussiano.
        limite_raio (int): Limite do raio de curvatura para plotagem das linhas de raio.
        name_traj (str, optional): Nome do trajeto específico a ser plotado. Se None, usa o df completo.

    Returns:
        folium.Map: Objeto Folium Map com o trajeto plotado.
    """

    df_final = detectar_curvas(df, sigma=sigma, limite_raio=limite_raio, name_traj=name_traj)

    if df_final.empty or df_final[['lat', 'lon']].isnull().all().any() or len(df_final) < 2:
        print(f"Não há dados suficientes ou válidos para plotar o trajeto {name_traj if name_traj else 'completo'}.")
        return None

    # Pegar o primeiro ponto válido para centrar o mapa inicialmente
    first_valid_point = df_final.dropna(subset=['lat', 'lon']).iloc[0]
    map_center = [first_valid_point['lat'], first_valid_point['lon']]
    m = folium.Map(location=map_center, zoom_start=15)

    # Adicionar o trajeto completo como uma linha
    folium.PolyLine(locations=df_final[['lat', 'lon']].values.tolist(), color='blue', weight=3, opacity=0.7,
                    tooltip=f"Trajeto: {df_final['id_route'].iloc[0]}").add_to(m)

    # Filtrar pontos de curva detectados
    curve_points = df_final[df_final['curva'] == True].dropna(subset=['lat', 'lon'])

    for idx, row in curve_points.iterrows():
        # Adicionar marcadores para pontos de curva
        popup_text = f"Curva<br>Curvatura: {row['curvatura']:.2e}<br>Raio: {row['raio_curvatura']:.2f} m"
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

        # Adicionar linha indicando o raio de curvatura para raios pequenos
        if row['raio_curvatura'] < limite_raio and not pd.isna(row['raio_curvatura']) and row['raio_curvatura'] != np.inf:
            # Para Folium, é mais fácil simplesmente marcar o ponto da curva
            # e, se quisermos visualizar o raio, teríamos que calcular o ponto final do vetor do raio.
            # Isso exige a direção do movimento no ponto da curva.
            # Para simplificar, focaremos nos CircleMarkers para a visualização inicial.
            pass # Deixando a complexidade da linha do raio de fora por enquanto para manter a simplicidade.

    # Ajustar o zoom para cobrir todo o trajeto
    min_lat, max_lat = df_final['lat'].min(), df_final['lat'].max()
    min_lon, max_lon = df_final['lon'].min(), df_final['lon'].max()
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

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
