from curvant.constants import ISL_BAIXO, ISL_ALTO


def classificar_isl(isl: float) -> str:
    """Faixa a que um valor de ISL pertence."""
    if isl < ISL_BAIXO:
        return 'baixo'
    if isl < ISL_ALTO:
        return 'medio'
    return 'alto'
