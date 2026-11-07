import matplotlib.pyplot as plt
from fitting_functions import *

#Left/right arrows are matplotlib's default back/forward shortcuts, so free them up
for k in ("keymap.back", "keymap.forward"):
    plt.rcParams[k] = [x for x in plt.rcParams[k] if x not in ("left", "right")]

def interactive_fit_all(data, measurementtype):
    items = [(n, k) for n, d in data.items() for k in d]
    if results is None:
        results = {n: {} for n in data}
    S = {} #state for the current measurement
    drag = [None] #line currently being dragged
    fig, ax = plt.subplots(figsize=(9, 5))

    def spectrum():
        name, key = items[S["i"]]
        return data[name][key]["channels"], data[name][key]["spectrum"]

    def pos(pair):
        return sorted(l.get_xdata()[0] for l in pair)

    def refresh(msg=""):
        """Update visibility, dimming and title (which shows the line positions)."""
        for idx in (0, 1):
            visible = idx == 0 or S["blue"]
            for l in S["lines"][idx] + [S["fitlines"][idx]]:
                l.set_visible(visible)
                l.set_alpha(1.0 if idx == S["active"] else 0.35)
        text = "   ".join(f"{c}: {lo:.0f}–{hi:.0f}"
                          for idx, c in enumerate(("red", "blue"))
                          if idx == 0 or S["blue"]
                          for lo, hi in [pos(S["lines"][idx])])
        name, key = items[S["i"]]
        ax.set_title(f"{name}/{key} [{S['i']+1}/{len(items)}]  {text}\n{msg}\n"
                     "→ blue pair, ← red pair, Enter = done", fontsize=8)
        fig.canvas.draw_idle()

    def load(i):
        S.update(i=i, active=0, blue=False, fits=[None, None], lines=[], fitlines=[])
        ch, sp = spectrum()
        ax.clear()
        ax.plot(ch, sp, lw=0.8)
        name, key = items[i]
        old = results.get(name, {}).get(key) #earlier result, or None
        for idx, (c, (a, b)) in enumerate(zip("rb", [(40, 50), (60, 70)])):
            lo, hi = np.percentile(ch, [a, b])
            had = old is not None and idx < len(old["lines"])
            if had:
                lo, hi = old["lines"][idx] #put the lines back where they were
            S["lines"].append([ax.axvline(lo, color=c), ax.axvline(hi, color=c)])
            fl, = ax.plot([], [], "k", lw=1.5)
            S["fitlines"].append(fl)
            if had and old[f"peak{idx + 1}"] is not None:
                S["fits"][idx] = old[f"peak{idx + 1}"]
                xs = np.linspace(lo, hi, 300)
                fl.set_data(xs, gauss(xs, *S["fits"][idx]["popt"]))
        S["blue"] = old is not None and len(old["lines"]) > 1
        refresh()

    def on_press(e):
        if e.inaxes is ax:
            for l in S["lines"][S["active"]]:
                if abs(e.x - ax.transData.transform((l.get_xdata()[0], 0))[0]) < 8:
                    drag[0] = l
                    return

    def on_motion(e):
        if drag[0] is not None and e.xdata is not None:
            drag[0].set_xdata([e.xdata, e.xdata])
            refresh()

    def on_release(e):
        if drag[0] is None:
            return
        drag[0] = None
        a = S["active"]
        lo, hi = pos(S["lines"][a])
        fit = S["fits"][a] = fit_peak(*spectrum(), lo, hi)
        if fit is None:
            S["fitlines"][a].set_data([], [])
            refresh("Fit failed / too few points")
        else:
            xs = np.linspace(lo, hi, 300)
            S["fitlines"][a].set_data(xs, gauss(xs, *fit["popt"]))
            refresh(f"mu = {fit['mu']:.2f}, FWHM = {fit['fwhm']:.2f}")

    def on_key(e):
        if e.key in ("left", "right"):
            S["active"] = int(e.key == "right")
            S["blue"] = S["blue"] or S["active"] == 1
            refresh()
        elif e.key == "enter":
            name, key = items[S["i"]]
            results[name][key] = {
                "peak1": S["fits"][0], "peak2": S["fits"][1],
                "lines": [tuple(pos(p)) for p in S["lines"][:1 + S["blue"]]],
            }
            if S["i"] + 1 < len(items):
                load(S["i"] + 1)
            else:
                fig.canvas.mpl_disconnect(cid)
                ax.set_title("All measurements done")
                fig.canvas.draw_idle()
                show_all_fits(data, results)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    cid = fig.canvas.mpl_connect("key_press_event", on_key)

    load(0)
    plt.show()
    return fig, results

def show_all_fits(data, results, ncols=3):
    items = [(n, k) for n, d in results.items() for k in d]
    if not items:
        print("No results to show")
        return
    nrows = -(-len(items) // ncols) #rounds up, so no math.ceil needed
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for ax, (name, key) in zip(axes.flat, items):
        ch, sp = data[name][key]["channels"], data[name][key]["spectrum"]
        ax.plot(ch, sp, lw=0.8, color="gray")
        fits = [(results[name][key][p], c) for p, c in (("peak1", "r"), ("peak2", "b"))]
        fits = [(f, c) for f, c in fits if f is not None]

        for f, c in fits:
            xs = np.linspace(*f["range"], 300)
            ax.plot(xs, gauss(xs, *f["popt"]), color=c, lw=1.5)

        if fits:   # zoom to the fitted region
            lo = min(f["range"][0] for f, _ in fits)
            hi = max(f["range"][1] for f, _ in fits)
            pad = 0.5 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(0, sp[(ch >= lo - pad) & (ch <= hi + pad)].max() * 1.1)

        labels = [f"mu={f['mu']:.2f}, FWHM={f['fwhm']:.2f}" for f, _ in fits] or ["no fit"]
        ax.set_title(f"{name}/{key}\n" + "\n".join(labels), fontsize=8)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts")

    for ax in axes.flat[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()

import csv

FIELDS = ["folder", "measurement", "peak", "line_lo", "line_hi",
          "mu", "mu_err", "fwhm", "fwhm_err", "A", "sigma", "c"]

def save_results(results, path="fit_results.csv"):
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

def load_results(path="fit_results.csv"):
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
                    "mu": float(row["mu"]), "mu_err": float(row["mu_err"]),
                    "fwhm": float(row["fwhm"]), "fwhm_err": float(row["fwhm_err"]),
                    "popt": np.array([float(row[k]) for k in ("A", "mu", "sigma", "c")]),
                    "range": (lo, hi),
                }
    return results