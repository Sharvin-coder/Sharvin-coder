"""Generate animated SVGs for the profile README from real training runs.

Two panels, each in a dark and light variant:

  assets/neural-field-sphere-{dark,light}.svg
    A coordinate MLP (Fourier features of (x,y,z) -> tanh MLP) fits a real
    spherical-harmonic mixture Re[Y_4^3] + 0.6 Re[Y_6^2] on S^2. The front
    hemisphere is rendered as an orthographically projected, Lambert-shaded
    quad mesh with a diverging blue-white-red colormap; the mesh animates
    through actual training checkpoints (epoch 0 -> 1200) with the true MSE.

  assets/mlp-multilingual-{dark,light}.svg
    An 8-12-12-8-5 tanh classifier trained on 100 real words from five
    languages (EN, ES, HI, AR, ZH) featurized by script-block fractions and
    character statistics. The animation replays forward passes for one word
    per language: pulses sweep the strong weights layer by layer, neuron
    glow equals the real activation, and the argmax language is ringed with
    its true softmax probability.

Pure numpy + scipy, seed 0. Every color, opacity, and number in the SVGs is
computed from the actual runs -- nothing is hand-typed.
"""

import unicodedata

import numpy as np
from scipy.special import sph_harm_y

rng = np.random.default_rng(0)

# ------------------------------------------------------------ colormap
# diverging blue-white-red (coolwarm-like), colorblind-legible
_DIV = np.array([
    [59, 76, 192], [144, 178, 254], [247, 247, 247],
    [245, 156, 125], [180, 4, 38],
], dtype=float)


def divmap(v, shade=1.0):
    """v in [0,1] -> hex, RGB scaled by Lambert shade factor."""
    v = float(np.clip(v, 0.0, 1.0)) * (len(_DIV) - 1)
    i = min(int(v), len(_DIV) - 2)
    f = v - i
    c = (_DIV[i] * (1 - f) + _DIV[i + 1] * f) * shade
    return "#{:02x}{:02x}{:02x}".format(*(int(round(min(255, x))) for x in c))


# ------------------------------------------------------------ panel 1: S^2 field
N_LAT, N_LON = 18, 22          # front-hemisphere mesh resolution
TILT = np.deg2rad(-18)         # tilt toward the viewer
KEEP = [0, 20, 60, 150, 400, 1200]


def sphere_target(theta, phi):
    """Real spherical-harmonic mixture Re[Y_4^3] + 0.6 Re[Y_6^2]."""
    y43 = sph_harm_y(4, 3, theta, phi).real
    y62 = sph_harm_y(6, 2, theta, phi).real
    return y43 + 0.6 * y62


def train_sphere_field():
    """Fit the SH mixture with Fourier features of (x,y,z) -> tanh MLP."""
    # train on a full-sphere grid so the field is globally consistent
    th = np.linspace(0.03, np.pi - 0.03, 30)
    ph = np.linspace(-np.pi, np.pi, 60, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    theta, phi = TH.ravel(), PH.ravel()
    xyz = np.stack([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)], 1)
    target = sphere_target(theta, phi)

    B = rng.normal(0, 1.6, (3, 24))
    feats = np.concatenate([np.sin(xyz @ B * np.pi), np.cos(xyz @ B * np.pi)], 1)
    d_in, d_h = feats.shape[1], 64
    W1 = rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_h)); b1 = np.zeros(d_h)
    W2 = rng.normal(0, 1 / np.sqrt(d_h), (d_h, 1)); b2 = np.zeros(1)

    params = dict(W1=W1, b1=b1, W2=W2, b2=b2)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v_ = {k: np.zeros_like(v) for k, v in params.items()}
    lr, beta1, beta2, eps = 3e-3, 0.9, 0.999, 1e-8

    # evaluation mesh: front hemisphere cell centers (after tilt)
    def mesh_feats(theta_c, phi_c):
        p = np.stack([np.sin(theta_c) * np.cos(phi_c),
                      np.sin(theta_c) * np.sin(phi_c),
                      np.cos(theta_c)], 1)
        return np.concatenate([np.sin(p @ B * np.pi), np.cos(p @ B * np.pi)], 1)

    checkpoints, losses = [], []
    eval_feats = None  # filled by build_mesh() before training loop
    mesh = build_mesh()
    eval_feats = mesh_feats(mesh["theta_c"], mesh["phi_c"])

    for it in range(KEEP[-1] + 1):
        h = np.tanh(feats @ params["W1"] + params["b1"])
        pred = (h @ params["W2"] + params["b2"]).ravel()
        err = pred - target
        if it in KEEP:
            he = np.tanh(eval_feats @ params["W1"] + params["b1"])
            checkpoints.append((he @ params["W2"] + params["b2"]).ravel())
            losses.append(float(np.mean(err ** 2)))
        g_pred = (2 * err / len(err))[:, None]
        gW2 = h.T @ g_pred; gb2 = g_pred.sum(0)
        g_h = g_pred @ params["W2"].T * (1 - h ** 2)
        gW1 = feats.T @ g_h; gb1 = g_h.sum(0)
        grads = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2)
        for k in params:
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v_[k] = beta2 * v_[k] + (1 - beta2) * grads[k] ** 2
            mh = m[k] / (1 - beta1 ** (it + 1))
            vh = v_[k] / (1 - beta2 ** (it + 1))
            params[k] -= lr * mh / (np.sqrt(vh) + eps)

    tgt = sphere_target(mesh["theta_c"], mesh["phi_c"])
    return mesh, np.array(checkpoints), losses, tgt


def build_mesh():
    """Front-hemisphere lat-lon quads, tilted, orthographically projected."""
    ct, st = np.cos(TILT), np.sin(TILT)

    def project(theta, phi):
        x = np.sin(theta) * np.sin(phi)          # right
        y = np.sin(theta) * np.cos(phi)          # toward viewer
        z = np.cos(theta)                        # up
        y2 = y * ct - z * st                     # tilt about x-axis
        z2 = y * st + z * ct
        return x, y2, z2

    thetas = np.linspace(0.02, np.pi - 0.02, N_LAT + 1)
    phis = np.linspace(-np.pi / 2, np.pi / 2, N_LON + 1)
    quads, theta_c, phi_c, shades = [], [], [], []
    light = np.array([-0.45, 0.75, 0.5])
    light /= np.linalg.norm(light)
    for i in range(N_LAT):
        for j in range(N_LON):
            tc = (thetas[i] + thetas[i + 1]) / 2
            pc = (phis[j] + phis[j + 1]) / 2
            corners = [project(t, p) for t, p in
                       [(thetas[i], phis[j]), (thetas[i], phis[j + 1]),
                        (thetas[i + 1], phis[j + 1]), (thetas[i + 1], phis[j])]]
            if all(c[1] < 0.02 for c in corners):
                continue                          # fully back-facing after tilt
            nx = np.sin(tc) * np.sin(pc)
            ny = np.sin(tc) * np.cos(pc)
            nz = np.cos(tc)
            n = np.array([nx, ny * np.cos(TILT) - nz * np.sin(TILT),
                          ny * np.sin(TILT) + nz * np.cos(TILT)])
            shade = 0.62 + 0.38 * max(0.0, float(n @ light))
            quads.append([(c[0], c[2]) for c in corners])
            theta_c.append(tc); phi_c.append(pc); shades.append(shade)
    return dict(quads=quads, theta_c=np.array(theta_c),
                phi_c=np.array(phi_c), shades=np.array(shades))


def train_field2d(target_fn, iters=800):
    """Fit a 2D neural field on [-1,1]^2; return final grid values + mse."""
    n = 20
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    target = target_fn(pts[:, 0], pts[:, 1])

    B = rng.normal(0, 2.0, (2, 16))
    feats = np.concatenate([np.sin(pts @ B * np.pi), np.cos(pts @ B * np.pi)], 1)
    d_in, d_h = feats.shape[1], 48
    W1 = rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_h)); b1 = np.zeros(d_h)
    W2 = rng.normal(0, 1 / np.sqrt(d_h), (d_h, 1)); b2 = np.zeros(1)
    params = dict(W1=W1, b1=b1, W2=W2, b2=b2)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v_ = {k: np.zeros_like(v) for k, v in params.items()}
    lr, beta1, beta2, eps = 3e-3, 0.9, 0.999, 1e-8
    for it in range(iters):
        h = np.tanh(feats @ params["W1"] + params["b1"])
        pred = (h @ params["W2"] + params["b2"]).ravel()
        err = pred - target
        g_pred = (2 * err / len(err))[:, None]
        gW2 = h.T @ g_pred; gb2 = g_pred.sum(0)
        g_h = g_pred @ params["W2"].T * (1 - h ** 2)
        gW1 = feats.T @ g_h; gb1 = g_h.sum(0)
        grads = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2)
        for k in params:
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v_[k] = beta2 * v_[k] + (1 - beta2) * grads[k] ** 2
            mh = m[k] / (1 - beta1 ** (it + 1))
            vh = v_[k] / (1 - beta2 ** (it + 1))
            params[k] -= lr * mh / (np.sqrt(vh) + eps)
    h = np.tanh(feats @ params["W1"] + params["b1"])
    pred = (h @ params["W2"] + params["b2"]).ravel()
    mse = float(np.mean((pred - target) ** 2))
    return n, pred, target, mse


FIELD2D_TARGETS = [
    ("spiral", lambda x, y: np.sin(6 * (x ** 2 + y ** 2) * np.pi
                                   - 3 * np.arctan2(y, x))),
    ("interference", lambda x, y: (np.sin(3 * np.pi * x)
                                   + np.sin(3 * np.pi * y)) / 2),
    ("gabor", lambda x, y: np.exp(-2 * (x ** 2 + y ** 2))
                           * np.sin(6 * np.pi * x)),
    ("rings", lambda x, y: np.sin(5 * np.pi * np.sqrt(x ** 2 + y ** 2))),
]


def _project_pt(x, y, z):
    """Same tilted orthographic projection used for the sphere mesh."""
    ct, st = np.cos(TILT), np.sin(TILT)
    return x, y * st + z * ct           # screen (right, up)


def sphere_svg(dark):
    mesh, ckpts, losses, tgt = FIELD
    width, height = 660, 352
    cx, cy, R = 172, 196, 108
    fg = "#c9d1d9" if dark else "#24292f"
    sub = "#8b949e" if dark else "#57606a"
    rim = "#30363d" if dark else "#d0d7de"
    grid = "#21262d" if dark else "#eaeef2"
    axis = "#484f58" if dark else "#afb8c1"

    amp = np.abs(tgt).max()
    frames = [np.clip((f / amp + 1) / 2, 0, 1) for f in ckpts]
    nf = len(frames)
    kt = [0.14 * i for i in range(nf)] + [0.92, 1.0]
    key_times = ";".join(f"{t:.3f}" for t in kt)
    dur = "12s"

    def P(x, y, z):
        sx, sz = _project_pt(x, y, z)
        return cx + R * sx, cy - R * sz

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">',
        f'<text x="16" y="19" fill="{fg}" font-size="12" font-weight="bold">neural field on S&#178; &#183; fitting Re[Y&#8324;&#179;] + 0.6&#183;Re[Y&#8326;&#178;]</text>',
        f'<text x="16" y="33" fill="{sub}" font-size="9">Fourier(x,y,z) &#8594; tanh MLP &#183; Adam &#183; real checkpoints, seed 0</text>',
    ]

    # --- 3D coordinate frame: three back walls with grid lines + axes
    E = 1.3                                      # box half-extent
    steps = np.linspace(-E, E, 6)
    walls = [
        lambda a, b: (a, E, b),                  # back wall   (y = +E)
        lambda a, b: (-E, a, b),                 # left wall   (x = -E)
        lambda a, b: (a, b, -E),                 # floor       (z = -E)
    ]
    for wall in walls:
        for s in steps:
            for line in ((wall(s, -E), wall(s, E)), (wall(-E, s), wall(E, s))):
                (x1, y1), (x2, y2) = P(*line[0]), P(*line[1])
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{grid}" stroke-width="0.7"/>'
                )
    for a, b, lbl in [((-E, -E, -E), (E, -E, -E), "x"),
                      ((-E, -E, -E), (-E, E, -E), "y"),
                      ((-E, -E, -E), (-E, -E, E), "z")]:
        (x1, y1), (x2, y2) = P(*a), P(*b)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{axis}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{x2 + (6 if lbl != "z" else -2):.1f}" y="{y2 + (10 if lbl != "z" else -5):.1f}" fill="{axis}" font-size="9">{lbl}</text>'
        )

    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{R + 1}" fill="none" stroke="{rim}" stroke-width="1.2"/>'
    )
    for k, (q, shade) in enumerate(zip(mesh["quads"], mesh["shades"])):
        pts = " ".join(f"{cx + R * x:.1f},{cy - R * z:.1f}" for x, z in q)
        vals = [divmap(f[k], shade) for f in frames]
        vals = vals + [vals[-1], vals[0]]
        parts.append(
            f'<polygon points="{pts}" stroke="none">'
            f'<animate attributeName="fill" values="{";".join(vals)}" keyTimes="{key_times}" dur="{dur}" repeatCount="indefinite"/></polygon>'
        )
    parts.append(
        f'<ellipse cx="{cx - 38}" cy="{cy - 60}" rx="27" ry="17" fill="white" opacity="{0.10 if dark else 0.20}" transform="rotate(-32 {cx - 38} {cy - 60})"/>'
    )
    for i, (ep, ls) in enumerate(zip(KEEP, losses)):
        op = ["0"] * (nf + 2)
        op[i] = "1"
        if i == nf - 1:
            op[nf] = "1"
        parts.append(
            f'<text x="16" y="{height - 12}" fill="{sub}" font-size="10" opacity="0">'
            f'epoch {ep:>4} &#183; mse {ls:.4f}'
            f'<animate attributeName="opacity" values="{";".join(op)}" keyTimes="{key_times}" dur="{dur}" repeatCount="indefinite"/></text>'
        )

    # --- right half: 2x2 gallery of converged 2D neural fields
    sq, gap, gx, gy = 133, 17, 356, 52
    for idx, ((name, _), (n, pred, tgt2, mse)) in enumerate(zip(FIELD2D_TARGETS, FIELDS2D)):
        ox = gx + (idx % 2) * (sq + gap)
        oy = gy + (idx // 2) * (sq + gap + 16)
        a2 = np.abs(tgt2).max()
        vv = np.clip((pred / a2 + 1) / 2, 0, 1)
        cell = sq / n
        for i in range(n):
            for j in range(n):
                parts.append(
                    f'<rect x="{ox + j * cell:.1f}" y="{oy + i * cell:.1f}" width="{cell + 0.15:.2f}" height="{cell + 0.15:.2f}" fill="{divmap(vv[i * n + j])}"/>'
                )
        parts.append(
            f'<rect x="{ox}" y="{oy}" width="{sq}" height="{sq}" fill="none" stroke="{rim}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ox}" y="{oy + sq + 11}" fill="{sub}" font-size="9">{name} &#183; mse {mse:.4f}</text>'
        )
    parts.append(
        f'<text x="{gx}" y="{gy - 8}" fill="{sub} " font-size="9">converged 2D fields (same recipe, four targets)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------ panel 2: multilingual MLP
WORDS = {
    "EN": ["the", "and", "water", "house", "night", "light", "dream", "world",
           "heart", "time", "stone", "river", "cloud", "fire", "green",
           "small", "voice", "road", "paper", "music"],
    "ES": ["agua", "casa", "noche", "luz", "sueño", "mundo",
           "corazón", "tiempo", "piedra", "río", "nube", "fuego",
           "verde", "pequeño", "voz", "camino", "papel",
           "música", "árbol", "cielo"],
    "HI": ["पानी", "घर", "रात",
           "रोशनी", "सपना",
           "दुनिया", "दिल",
           "समय", "पत्थर",
           "नदी", "बादल",
           "आग", "हरा", "छोटा",
           "आवाज़", "रास्ता",
           "कागज़", "संगीत",
           "पेड़", "आकाश"],
    "AR": ["ماء", "بيت", "ليل",
           "نور", "حلم", "عالم",
           "قلب", "وقت", "حجر",
           "نهر", "سحاب", "نار",
           "أخضر", "صغير",
           "صوت", "طريق", "ورق",
           "موسيقى", "شجرة",
           "سماء"],
    "ZH": ["水", "家", "夜", "光", "梦", "世界",
           "心", "时间", "石头", "河", "云",
           "火", "绿", "小", "声音", "路",
           "纸", "音乐", "树", "天空"],
}
LANGS = list(WORDS)


def _block_fracs(word):
    """Fractions of chars in latin / devanagari / arabic / CJK blocks."""
    lat = dev = ara = cjk = 0
    for ch in word:
        o = ord(ch)
        if o < 0x250:
            lat += 1
        elif 0x900 <= o < 0x980:
            dev += 1
        elif 0x600 <= o < 0x700:
            ara += 1
        elif 0x4E00 <= o < 0xA000:
            cjk += 1
    n = max(1, len(word))
    return lat / n, dev / n, ara / n, cjk / n


def featurize(word):
    lat, dev, ara, cjk = _block_fracs(word)
    vowels = sum(ch in "aeiouáéíóúü" for ch in word.lower()) / max(1, len(word))
    marks = sum(1 for ch in unicodedata.normalize("NFD", word)
                if unicodedata.combining(ch)) / max(1, len(word))
    length = min(len(word), 10) / 10
    distinct = len(set(word)) / max(1, len(word))
    return np.array([lat, dev, ara, cjk, vowels, marks, length, distinct])


def train_multilingual():
    X = np.array([featurize(w) for lang in LANGS for w in WORDS[lang]])
    y = np.array([i for i, lang in enumerate(LANGS) for _ in WORDS[lang]])
    Yh = np.eye(len(LANGS))[y]

    sizes = [8, 12, 12, 8, 5]
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

    lr = 0.15
    for _ in range(2500):
        acts = forward(X)
        delta = (acts[-1] - Yh) / len(X)
        for i in range(len(Ws) - 1, -1, -1):
            gW = acts[i].T @ delta; gb = delta.sum(0)
            if i > 0:
                delta = delta @ Ws[i].T * (1 - acts[i] ** 2)
            Ws[i] -= lr * gW; bs[i] -= lr * gb

    acc = float((forward(X)[-1].argmax(1) == y).mean())
    demo_words = {"EN": "dream", "ES": "corazón",
                  "HI": "संगीत",
                  "AR": "موسيقى",
                  "ZH": "音乐"}
    sample_acts = [[a[0] for a in forward(featurize(demo_words[lang])[None])]
                   for lang in LANGS]
    return sizes, Ws, sample_acts, acc, demo_words


def mlp_svg(dark):
    sizes, Ws, sample_acts, acc, demo_words = MLP
    n_layers = len(sizes)
    width, height = 760, 380
    lx = [80, 230, 380, 530, 680]
    fg = "#c9d1d9" if dark else "#24292f"
    sub = "#8b949e" if dark else "#57606a"
    pos, neg = "#58a6ff", "#f0883e"
    glow = "#e3b341" if dark else "#bf8700"
    n_phase, phase_s, total = len(LANGS), 3.0, 15.0
    dur = f"{total:.0f}s"

    def ys(k):
        m = sizes[k]
        return [206 + (i - (m - 1) / 2) * 25 for i in range(m)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">',
        f'<text x="20" y="22" fill="{fg}" font-size="13" font-weight="bold">which language is this word? 8&#8722;12&#8722;12&#8722;8&#8722;5 tanh &#8594; softmax</text>',
        f'<text x="20" y="36" fill="{sub}" font-size="10">trained on 100 real words across 5 scripts &#183; edge width &#8733; |w| &#183; glow = real activation &#183; train acc {acc:.0%} &#183; seed 0</text>',
        f'<text x="{lx[0]}" y="364" fill="{sub}" font-size="10" text-anchor="middle">script + char stats &#8712; &#8477;&#8312;</text>',
        f'<text x="{lx[1]}" y="364" fill="{sub}" font-size="10" text-anchor="middle">tanh (12)</text>',
        f'<text x="{lx[2]}" y="364" fill="{sub}" font-size="10" text-anchor="middle">tanh (12)</text>',
        f'<text x="{lx[3]}" y="364" fill="{sub}" font-size="10" text-anchor="middle">tanh (8)</text>',
        f'<text x="{lx[4]}" y="364" fill="{sub}" font-size="10" text-anchor="middle">softmax (5)</text>',
    ]

    kt_all = None
    for k, W in enumerate(Ws):
        y0s, y1s = ys(k), ys(k + 1)
        wmax = np.abs(W).max()
        for i, y0 in enumerate(y0s):
            for j, y1 in enumerate(y1s):
                w = W[i, j]
                sw = 0.3 + 2.0 * abs(w) / wmax
                col = pos if w >= 0 else neg
                parts.append(
                    f'<line x1="{lx[k]}" y1="{y0:.1f}" x2="{lx[k+1]}" y2="{y1:.1f}" stroke="{col}" stroke-width="{sw:.2f}" opacity="0.13"/>'
                )
        for i, y0 in enumerate(y0s):
            for j, y1 in enumerate(y1s):
                w = W[i, j]
                if abs(w) / wmax < 0.35:
                    continue
                col = pos if w >= 0 else neg
                L = float(np.hypot(lx[k + 1] - lx[k], y1 - y0))
                kts, offs, ops = ["0"], [f"{L:.0f}"], ["0"]
                for p in range(n_phase):
                    a = (p * phase_s + k * 0.55) / total
                    b = (p * phase_s + k * 0.55 + 0.5) / total
                    kts += [f"{a:.4f}", f"{a+1e-4:.4f}", f"{b:.4f}", f"{b+1e-4:.4f}"]
                    offs += [f"{L:.0f}", f"{L:.0f}", "0", f"{L:.0f}"]
                    ops += ["0", "1", "1", "0"]
                kts.append("1"); offs.append(f"{L:.0f}"); ops.append("0")
                parts.append(
                    f'<line x1="{lx[k]}" y1="{y0:.1f}" x2="{lx[k+1]}" y2="{y1:.1f}" stroke="{col}" stroke-width="1.8" '
                    f'stroke-dasharray="6 {L:.0f}" opacity="0">'
                    f'<animate attributeName="stroke-dashoffset" values="{";".join(offs)}" keyTimes="{";".join(kts)}" dur="{dur}" repeatCount="indefinite" calcMode="linear"/>'
                    f'<animate attributeName="opacity" values="{";".join(ops)}" keyTimes="{";".join(kts)}" dur="{dur}" repeatCount="indefinite" calcMode="discrete"/></line>'
                )

    for k in range(n_layers):
        for i, yy in enumerate(ys(k)):
            vals = []
            for p in range(n_phase):
                a = sample_acts[p][k][i]
                if k == 0:
                    a = float(a)                        # features already in [0,1]
                elif k < n_layers - 1:
                    a = (a + 1) / 2
                vals.append(float(np.clip(a, 0, 1)))
            kts, ops = ["0"], [f"{0.12 + 0.88 * vals[-1]:.2f}"]
            for p in range(n_phase):
                arrive = (p * phase_s + k * 0.55 + 0.4) / total
                kts.append(f"{arrive:.4f}")
                ops.append(f"{0.12 + 0.88 * vals[p]:.2f}")
            kts.append("1"); ops.append(ops[-1])
            parts.append(
                f'<circle cx="{lx[k]}" cy="{yy:.1f}" r="7.5" fill="{glow}" stroke="{sub}" stroke-width="0.8" opacity="0.15">'
                f'<animate attributeName="opacity" values="{";".join(ops)}" keyTimes="{";".join(kts)}" dur="{dur}" repeatCount="indefinite" calcMode="discrete"/></circle>'
            )

    # output labels
    for i, (lang, yy) in enumerate(zip(LANGS, ys(n_layers - 1))):
        parts.append(
            f'<text x="{lx[-1] + 18}" y="{yy + 3.5:.1f}" fill="{sub}" font-size="10">{lang}</text>'
        )

    # per-phase: the word itself (in its script), argmax ring + probability
    for p, lang in enumerate(LANGS):
        probs = sample_acts[p][n_layers - 1]
        win = int(np.argmax(probs))
        yy = ys(n_layers - 1)[win]
        a = (p * phase_s) / total
        arrive = (p * phase_s + (n_layers - 1) * 0.55 + 0.45) / total
        end = ((p + 1) * phase_s - 0.1) / total
        show = ["0", f"{a:.4f}", f"{end:.4f}", "1"]
        ring = ["0", f"{arrive:.4f}", f"{end:.4f}", "1"]
        parts.append(
            f'<text x="{lx[0]}" y="62" fill="{fg}" font-size="20" text-anchor="middle" opacity="0">{demo_words[lang]}'
            f'<animate attributeName="opacity" values="0;1;0;0" keyTimes="{";".join(show)}" dur="{dur}" repeatCount="indefinite" calcMode="discrete"/></text>'
        )
        parts.append(
            f'<circle cx="{lx[-1]}" cy="{yy:.1f}" r="12.5" fill="none" stroke="{glow}" stroke-width="2" opacity="0">'
            f'<animate attributeName="opacity" values="0;1;0;0" keyTimes="{";".join(ring)}" dur="{dur}" repeatCount="indefinite" calcMode="discrete"/></circle>'
        )
        parts.append(
            f'<text x="{lx[-1] + 18}" y="{yy - 12:.1f}" fill="{fg}" font-size="10" opacity="0">p={probs[win]:.2f}'
            f'<animate attributeName="opacity" values="0;1;0;0" keyTimes="{";".join(ring)}" dur="{dur}" repeatCount="indefinite" calcMode="discrete"/></text>'
        )
    parts.append("</svg>")
    return "".join(parts)


if __name__ == "__main__":
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "assets"
    FIELD = train_sphere_field()
    FIELDS2D = [train_field2d(fn) for _, fn in FIELD2D_TARGETS]
    MLP = train_multilingual()
    for dark in (True, False):
        suffix = "dark" if dark else "light"
        (root / f"neural-field-sphere-{suffix}.svg").write_text(sphere_svg(dark), encoding="utf-8")
        (root / f"mlp-multilingual-{suffix}.svg").write_text(mlp_svg(dark), encoding="utf-8")
        print(f"wrote neural-field-sphere-{suffix}.svg, mlp-multilingual-{suffix}.svg")
