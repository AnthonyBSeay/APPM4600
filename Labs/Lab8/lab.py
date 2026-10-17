def line(x0, x1, y0, y1, alpha):
    """
    Constructs the line through (x0, f(x0)) and (x1, f(x1))
    and evaluates it at alpha.

    """

    slope = (y1 - y0) / (x1 - x0)

    return y0 + slope * (alpha - x0)