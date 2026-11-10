#IMPORTS

import csv
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from pathlib import Path
from fitting_functions import *

#CSV FIELDS

FIELDS = ["folder", "measurement", "peak", "line_lo", "line_hi",
          "mu", "mu_err", "fwhm", "fwhm_err", "A", "sigma", "c"]
CSV_PATH = "fit_results.csv"

#FREE UP LEFT AND RIGHT ARROWS
 
for k in ("keymap.back", "keymap.forward"):
    plt.rcParams[k] = [x for x in plt.rcParams[k] if x not in ("left", "right")]

#LOADING AND SAVING

def load_results(path=CSV_PATH):
    if not Path(path).exists():
        return None
    results = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            d = results.setdefault(row["folder"], {})
            r = d.setdefault(row["measurement"],
                            {"peak1": None, "peak2": None, "lines": []})
            lo, hi = float(row["line_lo"]), float(row["line_hi"])
            r["lines"].append((lo, hi))
            if row["mu"]: #empty string = no successful fit
                r[f"peak{row['peak']}"] = {
                    "mu": float(row["mu"]),
                    "mu_err": float(row["mu_err"]),
                    "fwhm": float(row["fwhm"]),
                    "fwhm_err": float(row["fwhm_err"]),
                    "popt": np.array([float(row[k]) for k in ("A", "mu", "sigma", "c")]),
                    "range": (lo, hi)
                }        
    return results

def save_results(results, path=CSV_PATH):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for name, d in results.items():
            for key, r in d.items():
                for i, (lo, hi) in enumerate(r["lines"]):
                    row = {"folder": name, "measurement": key, "peak": i + 1,
                        "line_lo": lo, "line_hi": hi}
                    f = r[f"peak{i + 1}"]
                    if f is not None: #failed fits keep just the lines
                        A, mu, sigma, c = f["popt"]
                        row.update(mu=mu, mu_err=f["mu_err"], fwhm=f["fwhm"],
                                fwhm_err=f["fwhm_err"], A=A, sigma=sigma, c=c)
                    w.writerow(row)

#INTERACTIVE FIT

def interactive_fit_all(data, results=None, measurementtype=None, only=None):
    plt.close("all")
    if measurementtype is not None:
        data = {measurementtype: data[measurementtype]}
    if results is None:
        results = load_results() or {}
    
    items = only or [(n, k) for n, d in data.items() for k in d]
    
    S = {"done": False} #state for the current measurement
    drag = [None] #line currently being dragged
    updating = [False] #True while code edits the text boxes

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(bottom=0.2)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(bottom=0.2)
 
    def pos(pair):
        return sorted(l.get_xdata()[0] for l in pair)
 
    def sync_boxes():
        lo, hi = pos(S["lines"][S["active"]])
        updating[0] = True
        lo_box.set_val(f"{lo:.1f}")
        hi_box.set_val(f"{hi:.1f}")
        updating[0] = False
 
    def refresh(msg=""):
        for idx in (0, 1):
            visible = idx == 0 or S["blue"]
            for l in S["lines"][idx] + [S["fitlines"][idx]]:
                l.set_visible(visible)
                l.set_alpha(1.0 if idx == S["active"] else 0.35)
        text = "   ".join(f"{c}: {lo:.1f}–{hi:.1f}"
                          for idx, c in enumerate(("red", "blue"))
                          if idx == 0 or S["blue"]
                          for lo, hi in [pos(S["lines"][idx])])
        name, key = items[S["i"]]
        ax.set_title(f"{name}/{key} [{S['i'] + 1}/{len(items)}]  {text}\n{msg}\n"
                     "→ blue pair, ← red pair, e = done", fontsize=8)
        sync_boxes()
        fig.canvas.draw_idle()
 
    def draw_fit(idx):
        lo, hi = pos(S["lines"][idx])
        fit = S["fits"][idx] = fit_peak(S["e"], S["sp"], lo, hi)
        if fit is None:
            S["fitlines"][idx].set_data([], [])
            return "Fit failed / too few points"
        xs = np.linspace(lo, hi, 300)
        S["fitlines"][idx].set_data(xs, gauss(xs, *fit["popt"]))
        return f"mu = {fit['mu']:.2f}, FWHM = {fit['fwhm']:.2f}"
 
    def load(i):
        name, key = items[i]
        e, sp = data[name][key]["energy"], data[name][key]["spectrum"]
        ax.clear()
        ax.plot(e, sp, lw=0.8)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")
        old = results.get(name, {}).get(key)    # earlier result, or None
        lines, fitlines = [], []
        for idx, (c, (a, b)) in enumerate(zip("rb", [(40, 50), (60, 70)])):
            lo, hi = np.percentile(e, [a, b])
            if old is not None and idx < len(old["lines"]):
                lo, hi = old["lines"][idx]      # put the lines back where they were
            lines.append([ax.axvline(lo, color=c), ax.axvline(hi, color=c)])
            fitlines.append(ax.plot([], [], "k", lw=1.5)[0])
        n_old = len(old["lines"]) if old is not None else 0
        S.update(i=i, e=e, sp=sp, active=0, blue=n_old > 1,
                 fits=[None, None], lines=lines, fitlines=fitlines)
        msg = ""
        for idx in range(n_old):
            msg = draw_fit(idx)
        refresh(msg)
 
    #MOUSE INPUTS
 
    def on_press(e):
        if e.inaxes is ax:
            for l in S["lines"][S["active"]]:
                if abs(e.x - ax.transData.transform((l.get_xdata()[0], 0))[0]) < 8:
                    drag[0] = l
                    return
 
    def on_motion(e):
        if drag[0] is not None and e.xdata is not None:
            drag[0].set_xdata([e.xdata, e.xdata])
            fig.canvas.draw_idle()
 
    def on_release(e):
        if drag[0] is None:
            return
        drag[0] = None
        refresh(draw_fit(S["active"]))
 
    #KEYBOARD INPUT
 
    def on_key(e):
        try:
            if e.key in ("left", "right"):
                S["active"] = int(e.key == "right")
                msg = ""
                if S["active"] == 1 and not S["blue"]:
                    S["blue"] = True
                    msg = draw_fit(1)           # fit the blue pair as soon as it appears
                refresh(msg)
            elif e.key == "e":
                name, key = items[S["i"]]
                n = 1 + S["blue"]
                results.setdefault(name, {})[key] = {
                    "peak1": S["fits"][0],
                    "peak2": S["fits"][1] if S["blue"] else None,
                    "lines": [tuple(pos(p)) for p in S["lines"][:n]],
                }
                save_results(results)
                if S["i"] + 1 < len(items):
                    load(S["i"] + 1)
                else:
                    S["done"] = True
                    fig.canvas.mpl_disconnect(cid)
                    ax.set_title("All measurements done. Run show_all_fits(data, results)")
                    fig.canvas.draw_idle()
        except Exception:   # matplotlib swallows callback errors, so print them
            traceback.print_exc()
            ax.set_title("ERROR, see console")
            fig.canvas.draw_idle()
 
    #TEXT BOXES
 
    def on_submit(_text):
        if updating[0]:
            return
        try:
            a, b = sorted((float(lo_box.text), float(hi_box.text)))
        except ValueError:
            sync_boxes()
            return
        l0, l1 = S["lines"][S["active"]]
        l0.set_xdata([a, a])
        l1.set_xdata([b, b])
        refresh(draw_fit(S["active"]))
 
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    cid = fig.canvas.mpl_connect("key_press_event", on_key)
 
    lo_box = TextBox(fig.add_axes([0.2, 0.05, 0.15, 0.06]), "Low ")
    hi_box = TextBox(fig.add_axes([0.55, 0.05, 0.15, 0.06]), "High ")
    lo_box.on_submit(on_submit)
    hi_box.on_submit(on_submit)
 
    load(0)
    plt.show()
    return fig, results
 
#SUMMARY
 
def show_all_fits(data, results, ncols=3):
    items = [(n, k) for n, d in results.items() for k in d if k in data.get(n, {})]
    if not items:
        print("No results to show")
        return
    nrows = -(-len(items) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
 
    for ax, (name, key) in zip(axes.flat, items):
        e, sp = data[name][key]["energy"], data[name][key]["spectrum"]
        ax.plot(e, sp, lw=0.8, color="gray")
        fits = [(results[name][key][p], c) for p, c in (("peak1", "r"), ("peak2", "b"))]
        fits = [(f, c) for f, c in fits if f is not None]
 
        for f, c in fits:
            xs = np.linspace(*f["range"], 300)
            ax.plot(xs, gauss(xs, *f["popt"]), color=c, lw=1.5)
 
        if fits:    # zoom to the fitted region
            lo = min(f["range"][0] for f, _ in fits)
            hi = max(f["range"][1] for f, _ in fits)
            pad = 0.5 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(0, sp[(e >= lo - pad) & (e <= hi + pad)].max() * 1.1)
 
        labels = [f"mu={f['mu']:.2f}, FWHM={f['fwhm']:.2f}" for f, _ in fits] or ["no fit"]
        ax.set_title(f"{name}/{key}\n" + "\n".join(labels), fontsize=8)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")
 
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()