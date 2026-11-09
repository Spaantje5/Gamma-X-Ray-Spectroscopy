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