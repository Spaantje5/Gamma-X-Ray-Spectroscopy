import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot(data, measurement_type, material):
    plt.figure()
    plt.plot(data[measurement_type][material]["channels"], data[measurement_type][material]["spectrum"])
    plt.title(f"{measurement_type} / {material}")
    plt.xlabel("Channel"); plt.ylabel("Counts")
    plt.show()

FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2)) #conversion from sigma to full width at half height

def gauss(x, A, mu, sigma, c):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def fit_peak(channels, spectrum, x_lo, x_hi):
    m = (channels >= x_lo) & (channels <= x_hi)
    x, y = channels[m], spectrum[m]
    if len(x) < 5:
        return None
    p0 = [y.max() - y.min(), x[np.argmax(y)], (x_hi - x_lo) / 4, y.min()]
    try:
        popt, pcov = curve_fit(gauss, x, y, p0=p0)
    except RuntimeError:
        return None
    perr = np.sqrt(np.diag(pcov))
    return {
        "mu": popt[1],
        "mu_err": perr[1],
        "fwhm": FWHM_FACTOR * abs(popt[2]),
        "fwhm_err": FWHM_FACTOR * perr[2],
        "popt": popt,
        "range": (x_lo, x_hi)
    }

def gauss2(x, A1, m1, s1, A2, m2, s2, c):
    return (A1 * np.exp(-(x - m1)**2 / (2 * s1**2))
            + A2 * np.exp(-(x - m2)**2 / (2 * s2**2)) + c)

def two_peaks(x, A1, m1, s1, r, m2, c, b):
    s2 = s1 * np.sqrt(m2 / m1)                       # width scales like sqrt(E)
    return (A1 * np.exp(-(x - m1)**2 / (2 * s1**2))
            + r * A1 * np.exp(-(x - m2)**2 / (2 * s2**2))
            + c + b * (x - m1))

def fit_two(ch, sp, r1, r2):
    """r1 = (lo, hi) around the main peak, r2 = (lo, hi) around the shoulder.
    Returns (fit1, fit2, params) or (None, None, None)."""
    K = 2 * np.sqrt(2 * np.log(2))
    lo, hi = min(r1[0], r2[0]), max(r1[1], r2[1])
    m = (ch >= lo) & (ch <= hi)
    x, y = ch[m], sp[m]
    k = (x >= r1[0]) & (x <= r1[1])
    if len(x) < 10 or k.sum() < 3:
        return None, None, None
    j = np.argmax(y[k])
    c0 = np.percentile(y, 5)
    p0 = [y[k][j] - c0, x[k][j], (r1[1] - r1[0]) / 4, 0.2, 0.5 * (r2[0] + r2[1]), c0, 0]
    lower = [0, r1[0], (hi - lo) / 300, 0, r2[0], 0, -np.inf]
    upper = [np.inf, r1[1], hi - lo, 1.0, r2[1], np.inf, np.inf]
    try:
        p, cov = curve_fit(two_peaks, x, y, p0=p0, bounds=(lower, upper),
                           sigma=np.sqrt(np.maximum(y, 1)))
    except (RuntimeError, ValueError):
        return None, None, None
    e = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(e[:5])):
        return None, None, None

    A1, m1, s1, r, m2, c, b = p
    s2 = s1 * np.sqrt(m2 / m1)
    f1 = {"mu": m1, "mu_err": e[1], "fwhm": K * s1, "fwhm_err": K * e[2],
          "popt": np.array([A1, m1, s1, c]), "range": r1}
    f2 = {"mu": m2, "mu_err": e[4], "fwhm": K * s2,
          "fwhm_err": K * s2 * np.hypot(e[2] / s1, 0.5 * e[4] / m2),   # rough, ignores correlations
          "popt": np.array([r * A1, m2, s2, c + b * (m2 - m1)]), "range": r2}
    return f1, f2, p