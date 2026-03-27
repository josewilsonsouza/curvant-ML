import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.curve_detection import detectar_curvas

df = pd.read_parquet('data/eletro_rjdf_serra_clean.parquet')

# Compara limite_raio=50 vs 200 no mesmo trajeto
traj = 'tabela_final-19-spin-trajeto-t1_p2'
dt = df.query(f'id_route == "{traj}"')

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

DNIT_COR = {
    'muito_fechada': '#d32f2f',
    'fechada':       '#f57c00',
    'media':         '#fbc02d',
    'aberta':        '#388e3c',
    'suave':         '#b0bec5',
}

for col, limite in enumerate([50, 200]):
    dc = detectar_curvas(dt, sigma=2, limite_raio=limite)

    n_curva = dc['curva'].sum()
    dnit_dist = dc[dc['curva']]['classe_dnit'].value_counts().to_dict()

    ax = axes[0, col]
    ax.plot(dc['x'], dc['y'], color='#90caf9', linewidth=1.2, zorder=1, label='Reto')

    for classe, cor in DNIT_COR.items():
        pts = dc[dc['classe_dnit'] == classe]
        if not pts.empty:
            ax.scatter(pts['x'], pts['y'], color=cor, s=15, label=classe, zorder=2)

    ax.set_title(f'limite_raio = {limite} m\n{n_curva}/{len(dc)} pontos em curva ({n_curva/len(dc)*100:.1f}%)')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1, col]
    raio_clip = dc['raio_curvatura'].clip(upper=600)
    ax2.fill_between(dc.index, raio_clip, alpha=0.15, color='gray')
    ax2.plot(dc.index, raio_clip, color='gray', linewidth=0.6)

    for thresh, label, cor in [(50, 'muito_fechada ≤50m', '#d32f2f'),
                                (100, 'fechada ≤100m',    '#f57c00'),
                                (200, 'media ≤200m',      '#fbc02d'),
                                (500, 'aberta ≤500m',     '#388e3c')]:
        ax2.axhline(thresh, linestyle='--', linewidth=1, color=cor, label=label)

    ax2.axhline(limite, linestyle='-', linewidth=2, color='black', label=f'limite_raio={limite}m (detecção)')
    ax2.set_ylim(0, 620)
    ax2.set_xlabel('Índice'); ax2.set_ylabel('Raio (m)')
    ax2.set_title(f'Perfil de raio de curvatura — limite_raio={limite}m')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.3)

    print(f"\nlimite_raio={limite}m — DNIT: {dnit_dist}")

plt.suptitle(f'Detecção de curvas: {traj}', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('results/curvas_dnit_comparacao.png', dpi=120, bbox_inches='tight')
print("\nSalvo: results/curvas_dnit_comparacao.png")
