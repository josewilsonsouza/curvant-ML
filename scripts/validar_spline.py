"""
CurvantML — Validação/calibração da spline suavizadora (Camada 3).

Compara a detecção de curvas com a spline INTERPOLADORA legada (sigma_gps=0)
contra a SUAVIZADORA (sigma_gps>0), nos mesmos trajetos, para checar dois critérios:

  (i)  os raios impossíveis (overshoot, raio < 10 m) caem para ~0;
  (ii) o raio do "bulk" (curvas reais, 30–800 m) permanece correlacionado (>0.95)
       com o método atual — ou seja, não achatou curvas legítimas.

Uso:
    python scripts/validar_spline.py
    python scripts/validar_spline.py --n 120 --candidatos 1 2 3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from src.curve_detection import detectar_curvas

_CANDIDATOS = [
    'data/eletro_rjdf_serra_rjmgba_janeiro_clean.parquet',
    'data/eletro_rjdf_serra_clean.parquet',
    'data/eletro_rjdf_serra_rjmgba_janeiro.parquet',
    'data/eletro_rjdf_serra.parquet',
]


def _raios(df_traj, sigma_gps):
    """Raio por ponto para um trajeto, alinhado (mesmo drop_duplicates interno)."""
    try:
        out = detectar_curvas(df_traj, sigma=2, limite_raio=150, sigma_gps=sigma_gps)
        return out['raio_curvatura'].replace([np.inf, -np.inf], np.nan).values, out['curva'].values
    except Exception:
        return None, None


def main(args):
    data_path = next((p for p in _CANDIDATOS if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError("Nenhum dataset encontrado em data/")
    print(f"Dados: {data_path}")
    df = pd.read_parquet(data_path)

    rotas = df['id_route'].unique()
    rng = np.random.default_rng(42)
    if args.n and args.n < len(rotas):
        rotas = rng.choice(rotas, size=args.n, replace=False)
    print(f"Trajetos avaliados: {len(rotas)}\n")

    def frac_imp(r):
        r = r[~np.isnan(r)]
        return 100 * np.mean(r < 10) if len(r) else np.nan

    # raios legados (sigma_gps=0), uma vez
    base = {}
    for rid in rotas:
        dt = df[df['id_route'] == rid]
        r0, c0 = _raios(dt, 0.0)
        if r0 is not None:
            base[rid] = (r0, c0)

    r_old_all = np.concatenate([v[0] for v in base.values()])
    print(f"{'sigma_gps':>9} | {'raio<10m old->new':>20} | {'corr bulk(30-800m)':>18} | {'concord. curva':>14}")
    print("-" * 72)
    print(f"{'0 (atual)':>9} | {frac_imp(r_old_all):>8.2f}%{'':>11} | {'—':>18} | {'—':>14}")

    for sg in args.candidatos:
        r_new_parts, corr_x, corr_y, flag_agree, flag_n = [], [], [], 0, 0
        for rid, (r0, c0) in base.items():
            dt = df[df['id_route'] == rid]
            rn, cn = _raios(dt, float(sg))
            if rn is None or len(rn) != len(r0):
                continue
            r_new_parts.append(rn)
            # bulk: pontos onde AMBOS estão em 30-800 m (curvas reais, não outliers nem retas)
            m = (~np.isnan(r0)) & (~np.isnan(rn)) & (r0 > 30) & (r0 < 800) & (rn > 30) & (rn < 800)
            corr_x.append(r0[m]); corr_y.append(rn[m])
            flag_agree += int(np.sum(c0 == cn)); flag_n += len(c0)
        r_new_all = np.concatenate(r_new_parts)
        cx = np.concatenate(corr_x); cy = np.concatenate(corr_y)
        corr = np.corrcoef(cx, cy)[0, 1] if len(cx) > 2 else np.nan
        print(f"{sg:>9.1f} | {frac_imp(r_old_all):>7.2f}% -> {frac_imp(r_new_all):>5.2f}% | "
              f"{corr:>18.3f} | {100*flag_agree/flag_n:>13.1f}%")

    print("\nCritério OK se: raio<10m -> ~0  E  corr bulk > 0.95")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120, help='nº de trajetos amostrados (0 = todos)')
    ap.add_argument('--candidatos', type=float, nargs='+', default=[1.0, 2.0, 3.0])
    main(ap.parse_args())
