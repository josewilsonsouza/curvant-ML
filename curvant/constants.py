try:
    from curvant.utils.config import carregar_config
    _cfg  = carregar_config() or {}
    _phys = _cfg.get('physics', {}) or {}
    _curv = _cfg.get('curve_detection', {}) or {}
except Exception:
    _phys = {}
    _curv = {}

G: float = 9.81                                        # gravidade (m/s²)
MU: float = float(_phys.get('mu', 0.6))                # coeficiente de atrito
ISL_BAIXO: float = float(_phys.get('isl_baixo', 0.5))  # ISL < ISL_BAIXO  -> 'baixo'
ISL_ALTO: float = float(_phys.get('isl_alto', 0.8))    # ISL >= ISL_ALTO -> 'alto'

# Piso do raio (m) só para o rótulo de ISL; isola o ISL do ruído de B-spline sem tocar no
# ctp_accel do risco. Ver features._alvos_isl e config curve_detection.raio_min_isl.
RAIO_MIN_ISL: float = float(_curv.get('raio_min_isl', 20.0))
