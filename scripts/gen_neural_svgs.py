"""Generate animated SVGs for the profile README from real training runs.

Two panels, each in a dark and light variant:
  assets/neural-field-{dark,light}.svg  -- a coordinate MLP (Fourier features
    -> tanh MLP) fitting a spiral interference field; the grid animates
    through training checkpoints, so you watch the field converge.
  assets/mlp-forward-{dark,light}.svg   -- a 4-6-6-3 tanh classifier trained
    on Gaussian blobs; edge pulses and neuron brightness replay the forward
    pass for three test samples using the real weights and activations.

Pure numpy, seed 0, no fabricated numbers: every color/opacity in the SVGs
is computed from the actual runs.
"""

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- colormaps
# small viridis-like ramp (colorblind-safe), anchors -> linear interpolation
_ANCHORS = np.array([
    [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]
], dtype=float)


def cmap(v):
    """v in [0,1] -> hex color via the viridis-like ramp."""
    v = float(np.clip(v, 0.0, 1.0)) * (len(_ANCHORS) - 1)
    i = min(int(v), len(_ANCHORS) - 2)
    f = v - i
    c = _ANCHORS[i] * (1 - f) + _ANCHORS[i + 1] * f
    return "#{:02x}{:02x}{:02x}".format(*(int(round(x)) for x in c))


# ---------------------------------------------------------------- panel 1
def train_neural_field():
    """Fit f(x,y) = sin(6r^2 - 3theta) with Fourier features + tanh MLP."""
    n = 24
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1)              # (576, 2)
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    th = np.arctan2(pts[:, 1], pts[:, 0])
    target = np.sin(6 * r ** 2 * np.pi - 3 * th)           # spiral arms

    B = rng.normal(0, 2.0, (2, 16))                        # Fourier features
    feats = np.concatenate([np.sin(pts @ B * np.pi), np.cos(pts @ B * np.pi)], 1)

    d_in, d_h = feats.shape[1], 64
    W1 = rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_h)); b1 = np.zeros(d_h)
    W2 = rng.normal(0, 1 / np.sqrt(d_h), (d_h, 1)); b2 = np.zeros(1)

    checkpoints, losses, epochs_kept = [], [], []
    keep = [0, 20, 60, 150, 400, 1200]
    m = {k: np.zeros_like(v) for k, v in
         dict(W1=W1, b1=b1, W2=W2, b2=b2).items()}
    v_ = {k: np.zeros_like(val) for k, val in
          dict(W1=W1, b1=b1, W2=W2, b2=b2).items()}
    lr, beta1, beta2, eps = 3e-3, 0.9, 0.999, 1e-8

    for it in range(keep[-1] + 1):
        h = np.tanh(feats @ W1 + b1)
        pred = (h @ W2 + b2).ravel()
        err = pred - target
        loss = float(np.mean(err ** 2))
        if it in keep:
            checkpoints.append(pred.copy())
            losses.append(loss)
            epochs_kept.append(it)
        # backward
        g_pred = (2 * err / len(err))[:, None]
        gW2 = h.T @ g_pred; gb2 = g_pred.sum(0)
        g_h = g_pred @ W2.T * (1 - h ** 2)
        gW1 = feats.T @ g_h; gb1 = g_h.sum(0)
        for k, g in dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2).items():
            m[k] = beta1 * m[k] + (1 - beta1) * g
            v_[k] = beta2 * v_[k] + (1 - beta2) * g ** 2
            mh = m[k] / (1 - beta1 ** (it + 1))
            vh = v_[k] / (1 - beta2 ** (it + 1))
            upd = lr * mh / (np.sqrt(vh) + eps)
            if k == "W1": W1 -= upd
            elif k == "b1": b1 -= upd
            elif k == "W2": W2 -= upd
            elif k == "b2": b2 -= upd

    return n, np.array(checkpoints), losses, epochs_kept, target


def field_svg(dark):
    n, ckpts, losses, epochs, target = FIELD
    cell, pad_l, pad_t = 13, 14, 40
    gw = n * cell
    width, height = pad_l * 2 + gw, pad_t + gw + 30
    fg = "#c9d1d9" if dark else "#24292f"
    sub = "#8b949e" if dark else "#57606a"
    # normalize all frames to the target's range for honest comparison
    lo, hi = target.min(), target.max()
    frames = [(f - lo) / (hi - lo) for f in ckpts]

    # dwell on the converged frame, then loop
    nf = len(frames)
    kt = [0.14 * i for i in range(nf)] + [0.92, 1.0]
    key_times = ";".join(f"{t:.3f}" for t in kt)
    dur = "12s"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">',
        f'<text x="{pad_l}" y="18" fill="{fg}" font-size="13" font-weight="bold">neural field: f(x,y) learning sin(6&#960;r&#178; &#8722; 3&#952;)</text>',
        f'<text x="{pad_l}" y="32" fill="{sub}" font-size="10">Fourier features &#8594; tanh MLP (64) &#183; Adam &#183; real checkpoints, seed 0</text>',
    ]
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            vals = [cmap(f[idx]) for f in frames]
            vals = vals + [vals[-1], vals[0]]           # hold, then wrap
            x, y = pad_l + j * cell, pad_t + i * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}">'
                f'<animate attributeName="fill" values="{";".join(vals)}" keyTimes="{key_times}" dur="{dur}" repeatCount="indefinite"/></rect>'
            )
    # epoch / loss captions that switch with the frames
    for i, (ep, ls) in enumerate(zip(epochs, losses)):
        op = ["0"] * nf + ["0", "0"]
        op[i] = "1"
        if i == nf - 1:
            op[nf] = "1"                                 # keep last caption during dwell
        parts.append(
            f'<text x="{pad_l}" y="{pad_t + gw + 20}" fill="{sub}" font-size="11" opacity="0">'
            f'epoch {ep:>4} &#183; mse {ls:.4f}'
            f'<animate attributeName="opacity" values="{";".join(op)}" keyTimes="{key_times}" dur="{dur}" repeatCount="indefinite"/></text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- panel 2
def train_mlp():
    """4-6-6-3 tanh classifier on 3 Gaussian blobs in R^4."""
    centers = rng.normal(0, 1.6, (3, 4))
    Xs, ys = [], []
    for c in range(3):
        Xs.append(centers[c] + rng.normal(0, 0.55, (60, 4)))
        ys += [c] * 60
    X = np.vstack(Xs); y = np.array(ys)
    Yh = np.eye(3)[y]

    sizes = [4, 6, 6, 3]
    Ws = [rng.normal(0, 1 / np.sqrt(a), (a, b)) for a, b in zip(sizes, sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]

    def forward(A):
        acts = [A]
        for i, (W, b) in enumerate(zip(Ws, bs)):
            Z = acts[-1] @ W + b
            acts.append(np.tanh(Z) if i < len(Ws) - 1 else Z)
        e = np.exp(acts[-1] - acts[-1].max(1, keepdims=True))
        acts[-1] = e / e.sum(1, keepdims=True)
        return acts

    lr = 0.08
    for _ in range(600):
        acts = forward(X)
        delta = (acts[-1] - Yh) / len(X)
        for i in range(len(Ws) - 1, -1, -1):
            gW = acts[i].T @ delta; gb = delta.sum(0)
            if i > 0:
                delta = delta @ Ws[i].T * (1 - acts[i] ** 2)
            Ws[i] -= lr * gW; bs[i] -= lr * gb

    acc = float((forward(X)[-1].argmax(1) == y).mean())
    samples = np.stack([centers[c] + rng.normal(0, 0.55, 4) for c in range(3)])
    sample_acts = [forward(s[None])[0:] for s in samples]
    sample_acts = [[a[0] for a in forward(s[None])] for s in samples]
    return sizes, Ws, sample_acts, acc


def mlp_svg(dark):
    sizes, Ws, sample_acts, acc = MLP
    width, height = 640, 300
    lx = [70, 240, 410, 570]
    fg = "#c9d1d9" if dark else "#24292f"
    sub = "#8b949e" if dark else "#57606a"
    pos, neg = "#58a6ff", "#f0883e"
    glow = "#e3b341" if dark else "#bf8700"
    n_phase, phase, dur = 3, 4.0, "12s"

    def ys(k):
        m = sizes[k]
        return [150 + (i - (m - 1) / 2) * 34 for i in range(m)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">',
        f'<text x="20" y="22" fill="{fg}" font-size="13" font-weight="bold">forward pass: 4&#8722;6&#8722;6&#8722;3 tanh &#8594; softmax</text>',
        f'<text x="20" y="36" fill="{sub}" font-size="10">real trained weights &#183; edge width &#8733; |w| &#183; neuron glow = activation &#183; train acc {acc:.2%} &#183; seed 0</text>',
        f'<text x="{lx[0]}" y="284" fill="{sub}" font-size="10" text-anchor="middle">x &#8712; &#8477;&#8308;</text>',
        f'<text x="{lx[1]}" y="284" fill="{sub}" font-size="10" text-anchor="middle">tanh</text>',
        f'<text x="{lx[2]}" y="284" fill="{sub}" font-size="10" text-anchor="middle">tanh</text>',
        f'<text x="{lx[3]}" y="284" fill="{sub}" font-size="10" text-anchor="middle">softmax</text>',
    ]

    # edges + travelling pulses (one pulse window per layer per phase)
    for k, W in enumerate(Ws):
        y0s, y1s = ys(k), ys(k + 1)
        wmax = np.abs(W).max()
        for i, y0 in enumerate(y0s):
            for j, y1 in enumerate(y1s):
                w = W[i, j]
                sw = 0.4 + 2.2 * abs(w) / wmax
                col = pos if w >= 0 else neg
                parts.append(
                    f'<line x1="{lx[k]}" y1="{y0:.1f}" x2="{lx[k+1]}" y2="{y1:.1f}" stroke="{col}" stroke-width="{sw:.2f}" opacity="0.16"/>'
                )
        # pulse overlay: dash sliding along each edge during its window
        seg = np.hypot(lx[k + 1] - lx[k], 0) + 40
        for i, y0 in enumerate(y0s):
            for j, y1 in enumerate(y1s):
                w = Ws[k][i, j]
                if abs(w) / wmax < 0.25:
                    continue                            # only strong edges pulse
                col = pos if w >= 0 else neg
                L = float(np.hypot(lx[k + 1] - lx[k], y1 - y0))
                t0s, t1s = [], []
                offs, ops, kts = [], [], []
                # windows: phase p starts at p*phase; layer k pulses in [k*0.8, k*0.8+0.7]
                kts, offs, ops = ["0"], [f"{L:.0f}"], ["0"]
                for p in range(n_phase):
                    a = (p * phase + k * 0.8) / 12.0
                    b = (p * phase + k * 0.8 + 0.7) / 12.0
                    kts += [f"{a:.4f}", f"{a+1e-4:.4f}", f"{b:.4f}", f"{b+1e-4:.4f}"]
                    offs += [f"{L:.0f}", f"{L:.0f}", "0", f"{L:.0f}"]
                    ops += ["0", "1", "1", "0"]
                kts.append("1"); offs.append(f"{L:.0f}"); ops.append("0")
                parts.append(
                    f'<line x1="{lx[k]}" y1="{y0:.1f}" x2="{lx[k+1]}" y2="{y1:.1f}" stroke="{col}" stroke-width="2" '
                    f'stroke-dasharray="7 {L:.0f}" opacity="0">'
                    f'<animate attributeName="stroke-dashoffset" values="{";".join(offs)}" keyTimes="{";".join(kts)}" dur="{dur}" repeatCount="indefinite" calcMode="linear"/>'
                    f'<animate attributeName="opacity" values="{";".join(ops)}" keyTimes="{";".join(kts)}" dur="{dur}" repeatCount="indefinite" calcMode="discrete"/></line>'
                )

    # neurons: glow tracks the real activation of the current sample
    for k in range(4):
        for i, yy in enumerate(ys(k)):
            # per-phase activation value in [0,1]
            vals = []
            for p in range(n_phase):
                a = sample_acts[p][k][i]
                if k == 0:
                    a = (a + 3) / 6                      # inputs: squash to [0,1]
                elif k < 3:
                    a = (a + 1) / 2                      # tanh range
                vals.append(float(np.clip(a, 0, 1)))
            kts, ops = ["0"], [f"{0.15 + 0.85 * vals[-1]:.2f}"]
            for p in range(n_phase):
                arrive = (p * phase + k * 0.8 + 0.55) / 12.0
                kts.append(f"{arrive:.4f}")
                ops.append(f"{0.15 + 0.85 * vals[p]:.2f}")
            kts.append("1"); ops.append(ops[-1])
            parts.append(
                f'<circle cx="{lx[k]}" cy="{yy:.1f}" r="9" fill="{glow}" stroke="{sub}" stroke-width="1" opacity="0.2">'
                f'<animate attributeName="opacity" values="{";".join(ops)}" keyTimes="{";".join(kts)}" dur="{dur}" repeatCount="indefinite" calcMode="discrete"/></circle>'
            )

    # argmax ring on the output layer per phase
    for p in range(n_phase):
        probs = sample_acts[p][3]
        win = int(np.argmax(probs))
        yy = ys(3)[win]
        a = (p * phase + 3 * 0.8 + 0.6) / 12.0
        b = ((p + 1) * phase - 0.15) / 12.0
        kts = ["0", f"{a:.4f}", f"{b:.4f}", "1"]
        ops = ["0", "1", "0", "0"]
        parts.append(
            f'<circle cx="{lx[3]}" cy="{yy:.1f}" r="14" fill="none" stroke="{glow}" stroke-width="2" opacity="0">'
            f'<animate attributeName="opacity" values="{";".join(ops)}" keyTimes="{";".join(kts)}" dur="{dur}" repeatCount="indefinite" calcMode="discrete"/></circle>'
        )
        parts.append(
            f'<text x="{lx[3] + 24}" y="{yy + 4:.1f}" fill="{fg}" font-size="10" opacity="0">p={probs[win]:.2f}'
            f'<animate attributeName="opacity" values="{";".join(ops)}" keyTimes="{";".join(kts)}" dur="{dur}" repeatCount="indefinite" calcMode="discrete"/></text>'
        )
    parts.append("</svg>")
    return "".join(parts)


if __name__ == "__main__":
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "assets"
    FIELD = train_neural_field()
    MLP = train_mlp()
    globals()["FIELD"], globals()["MLP"] = FIELD, MLP
    for dark in (True, False):
        suffix = "dark" if dark else "light"
        (root / f"neural-field-{suffix}.svg").write_text(field_svg(dark), encoding="utf-8")
        (root / f"mlp-forward-{suffix}.svg").write_text(mlp_svg(dark), encoding="utf-8")
        print(f"wrote neural-field-{suffix}.svg, mlp-forward-{suffix}.svg")
