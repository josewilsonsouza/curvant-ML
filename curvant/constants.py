"""
Constantes físicas do ISL — fonte única do projeto.

g é constante universal (fixa). μ (atrito) e os limiares de classe vêm de
config.yaml (seção `physics`), então podem ser ajustados sem editar código.
Todos os módulos importam daqui em vez de redefinir 9.81 / 0.6 / 0.5 / 0.8.

  ISL = v² / (R · g · μ)
  classe = 'baixo' se ISL < ISL_BAIXO | 'alto' se ISL >= ISL_ALTO | senão 'medio'
"""

try:
    from curvant.utils.config import carregar_config
    _phys = (carregar_config() or {}).get('physics', {}) or {}
except Exception:
    _phys = {}

G: float = 9.81                                        # gravidade (m/s²)
MU: float = float(_phys.get('mu', 0.6))                # coeficiente de atrito
ISL_BAIXO: float = float(_phys.get('isl_baixo', 0.5))  # ISL < ISL_BAIXO  -> 'baixo'
ISL_ALTO: float = float(_phys.get('isl_alto', 0.8))    # ISL >= ISL_ALTO -> 'alto'
