#!/usr/bin/env python3
"""Usage: tools/make_scoreboard_svg.py assets/scoreboard.svg. Draws the README scoreboard: first run -> now per size class (log scale) and one Moon-Beam token's 144 ms."""
import math, sys

BEST = [  # class, model, first run, now, note
    ("0.8B", "Qwen3.5-0.8B", 7.5, 16.8, "bench"),
    ("2B", "Qwen3.5-2B", 0.1, 7.7, "chat"),
    ("4B", "Ternary-Bonsai-4B", 2.2, 4.3, "chat"),
    ("8B dense", "Bonsai-8B (1-bit)", 1.21, 5.2, "chat"),
    ("8B MoE", "Moon-Beam-8B-A1B", 1.87, 7.2, "chat"),
    ("9B", "Qwen3.5-9B", 0.16, 0.27, "chat"),
    ("16B MoE", "Ling-mini-2.0", 4.1, 5.1, "chat"),
]
ANATOMY = [  # label, ms, colour slot
    ("Q4_K non-expert weights", 50, 1),
    ("flash waits", 32, 2),
    ("Q2_K experts", 26, 3),
    ("DeltaNet, attn", 18, 4),
    ("other", 18, 5),
]

W, L, R = 760, 168, 120
T = 88
rowH = 40
lo, hi = math.log10(0.08), math.log10(30)
x = lambda v: L + (math.log10(v) - lo) / (hi - lo) * (W - L - R)
chartBottom = T + len(BEST) * rowH
aT = chartBottom + 64           # anatomy section top
H = aT + 112

o = []
a = o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">')
a('''<style>
  .bg{fill:#ffffff} .ink{fill:#1f2328} .mut{fill:#59636e} .grid{stroke:#d8dee4} .bar{stroke:#2a78d6}
  .start{fill:#ffffff;stroke:#8c959f} .now{fill:#2a78d6} .gain{fill:#1a7f37}
  .s1{fill:#2a78d6} .s2{fill:#eb6834} .s3{fill:#1baf7a} .s4{fill:#eda100} .s5{fill:#e87ba4}
  @media (prefers-color-scheme: dark){
    .bg{fill:#0d1117} .ink{fill:#e6edf3} .mut{fill:#9198a1} .grid{stroke:#30363d} .bar{stroke:#3987e5}
    .start{fill:#0d1117;stroke:#9198a1} .now{fill:#3987e5} .gain{fill:#3fb950}
    .s1{fill:#3987e5} .s2{fill:#d95926} .s3{fill:#199e70} .s4{fill:#c98500} .s5{fill:#d55181}
  }
  text{font-size:12px}
</style>''')
a(f'<rect class="bg" width="{W}" height="{H}" rx="8"/>')
a(f'<text class="ink" x="16" y="26" style="font-size:16px;font-weight:600">Best model per size class: first run on the TV box → now</text>')
a(f'<text class="mut" x="16" y="44">generation speed, tokens per second (log scale); ○ first run, ● now; gain on the right</text>')
for t in (0.1, 0.3, 1, 3, 10, 30):
    a(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T-8}" y2="{chartBottom}" stroke-width="1"/>')
    a(f'<text class="mut" x="{x(t):.1f}" y="{T-14}" text-anchor="middle">{t}</text>')
for i, (cls, model, s, n, note) in enumerate(BEST):
    cy = T + i * rowH + rowH / 2
    a(f'<text class="ink" x="16" y="{cy-3:.1f}" style="font-size:13px;font-weight:600">{cls}</text>')
    a(f'<text class="mut" x="16" y="{cy+12:.1f}" style="font-size:11px">{model}</text>')
    a(f'<line class="bar" x1="{x(s):.1f}" x2="{x(n):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke-width="3" stroke-linecap="round"/>')
    a(f'<circle class="start" cx="{x(s):.1f}" cy="{cy:.1f}" r="5" stroke-width="2"/>')
    a(f'<circle class="now" cx="{x(n):.1f}" cy="{cy:.1f}" r="6"/>')
    a(f'<text class="mut" x="{x(s)-10:.1f}" y="{cy+4:.1f}" text-anchor="end">{s:g}</text>')
    a(f'<text class="ink" x="{x(n)+11:.1f}" y="{cy+4:.1f}" style="font-weight:600">{n:g} <tspan class="mut" style="font-weight:400;font-size:11px">{note}</tspan></text>')
    g = n / s
    a(f'<text class="gain" x="{W-16}" y="{cy+4:.1f}" text-anchor="end" style="font-weight:600">×{g:.1f}</text>' if g < 10 else
      f'<text class="gain" x="{W-16}" y="{cy+4:.1f}" text-anchor="end" style="font-weight:600">×{g:.0f}</text>')

# anatomy of one Moon-Beam token
a(f'<text class="ink" x="16" y="{aT-24}" style="font-size:14px;font-weight:600">Where one Moon-Beam-8B-A1B token\'s 144 ms go (≈7 tok/s, E104)</text>')
total = sum(m for _, m, _ in ANATOMY)
bx, bw = 16, W - 32
cx = bx
for i, (label, ms, slot) in enumerate(ANATOMY):
    w = ms / total * bw
    a(f'<rect class="s{slot}" x="{cx:.1f}" y="{aT-8}" width="{max(w-2,1):.1f}" height="26" rx="{4 if i in (0, len(ANATOMY)-1) else 0}"/>')
    a(f'<text class="ink" x="{cx+2:.1f}" y="{aT+38}" style="font-weight:600">{ms} ms</text>')
    a(f'<text class="mut" x="{cx+2:.1f}" y="{aT+54}" style="font-size:11px">{label}</text>')
    cx += w
a(f'<text class="mut" x="16" y="{aT+84}" style="font-size:11px">Flash waits: 4.7 expert misses per token × ~6.8 ms. The non-expert weights are RAM-bound; the experts are flash-bound.</text>')
a('</svg>')
open(sys.argv[1], 'w').write('\n'.join(o) + '\n')
