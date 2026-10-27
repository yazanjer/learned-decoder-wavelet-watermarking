"""Figure: architecture of the watermarking network (encoder with fixed wavelet, channel, decoder with learned
multi-level analysis transform; capacity control as inset). Schematic only; every size stated in it is taken from
Section 3 of the manuscript. Output: manuscript/figures/fig_architecture.{pdf,png} (300 dpi), double-column width."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 6.5,
                     "pdf.fonttype": 42, "mathtext.fontset": "dejavuserif"})
FIXED, LEARN, DATA, CHAN = "#e6e6e6", "#c9d6e3", "white", "#efe3cf"
EDGE = "#404040"

fig, ax = plt.subplots(figsize=(7.16, 3.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
FS = 5.2


def box(x, y, w, h, text, fc, lw=0.6, fs=FS, style="round,pad=0.2,rounding_size=0.7"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=EDGE, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.2)
    return (x, y, w, h)


def arrow(p, q, text=None, off=(0, 0.6)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5, lw=0.6, color=EDGE,
                                 shrinkA=0, shrinkB=0))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=FS)


def R(b, dy=0.0):
    return (b[0] + b[2] + 0.4, b[1] + b[3] / 2 + dy)


def L(b, dy=0.0):
    return (b[0] - 0.4, b[1] + b[3] / 2 + dy)


# ---------------- encoder (top row)
ax.text(0.5, 48, "Encoder", fontsize=6.8, fontweight="bold", va="center")
x = box(0.5, 34, 8, 8, "Cover $x$\n$3\\times128$\n$\\times128$", DATA)
a0 = box(12, 34, 11, 8, "Fixed analysis\n$A_0$: CDF 9/7\nor Haar, 2 levels", FIXED)
arrow(R(x), L(a0))
m = box(12, 22.5, 11, 8, "Message $m$\n(64 bits)\n$\\to M(m)$,\n$16\\times32\\times32$", DATA)
enc = box(27.3, 27, 15, 12, "Encoder network\ninput $[B_L(x), M(m)]$\n3 conv-BN-ReLU (64)\n$\\to\\Delta$, $12\\times32\\times32$", LEARN)
arrow(R(a0), L(enc, 3), "$B_L(x)$", off=(-0.3, 0.9))
arrow(R(m), L(enc, -3))
s0 = box(46, 34, 11, 8, "Fixed synthesis\n$S_0=A_0^{-1}$ of\n$A_0(x)\\oplus\\Delta$", FIXED)
arrow(R(enc, 3), L(s0))
alpha = box(46, 22.5, 11, 8, "Strength map\n$\\alpha(x)\\in(0,1)$\n(3-layer CNN)", LEARN)
res = box(61, 27, 13.5, 12, "Residual\n$r=\\alpha\\odot(S_0(\\cdot)-x)$\nprojected to 40 dB:\n$x_w=x+\\rho(r)\\,r$", LEARN)
arrow(R(s0), L(res, 3))
arrow(R(alpha), L(res, -3))
ch = box(78.5, 27, 21, 12, "Channel $\\mathcal{N}$ (training)\nreal JPEG Q50 or Q30-90,\nblur, noise, dropout,\nidentity; straight-through", CHAN)
arrow(R(res), L(ch))
ax.text((res[0] + res[2] + ch[0]) / 2, 31.2, "$x_w$\n(8-bit)", ha="center", va="top", fontsize=FS)

# ---------------- decoder (bottom row)
ax.text(0.5, 17.5, "Decoder", fontsize=6.8, fontweight="bold", va="center")
y = box(83, 2.5, 16.5, 9, "Received $y$\nmapped to $[-1,1]$", DATA)
arrow((91.25, ch[1] - 0.4), (91.25, y[1] + y[3] + 0.4))
ath = box(58, 0.5, 21, 13, "Learned analysis $A_\\theta$\ninitialised at $A_0$, invertible\nV2: residual lifting, 24 ops,\n21,192 parameters\nV3: affine coupling (Haar),\n20,960 parameters", LEARN)
arrow(L(y), R(ath))
z = box(41, 2.5, 13, 9, "Deepest bands\n$z=B_L^{\\theta}(2y-1)$\n$12\\times32\\times32$", DATA)
arrow(L(ath), R(z))
body = box(20, 2, 17, 10, "Decoder body\n4 conv-BN-ReLU (64),\nconv to 1, pool $8\\times8$,\nlinear to 64 logits", LEARN)
arrow(L(z), R(body))
out = box(0.5, 2.5, 15.5, 9, "$\\hat m_k=\\mathbb{1}[D(y)_k>0]$\nloss: BCE$(D(y),m)$", DATA)
arrow(L(body), R(out))

# inset: capacity control
ax.add_patch(FancyBboxPatch((20, 15.5), 59, 5.5, boxstyle="round,pad=0.2,rounding_size=0.5", fc="white",
                            ec="#808080", lw=0.5, ls="--"))
ax.text(49.5, 18.25, "Capacity control V1+: $A_\\theta=A_0$ (CDF 9/7, fixed) and a front end on the deepest bands,\n"
                     "$z=B_L(2y-1)+F_\\psi(B_L(2y-1))$, 21,278 parameters, no invertible structure",
        ha="center", va="center", fontsize=FS, linespacing=1.25)

# legend
for i, (fc, t) in enumerate([(FIXED, "fixed"), (LEARN, "trained"), (CHAN, "distortion")]):
    ax.add_patch(FancyBboxPatch((66 + i * 11.5, 47), 2.4, 1.8, boxstyle="round,pad=0.1,rounding_size=0.3",
                                fc=fc, ec=EDGE, lw=0.5))
    ax.text(69.2 + i * 11.5, 47.9, t, va="center", fontsize=5.6)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
fig.savefig(os.path.join(out_dir, "fig_architecture.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(out_dir, "fig_architecture.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
print("ok")
