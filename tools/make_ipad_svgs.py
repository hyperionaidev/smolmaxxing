#!/usr/bin/env python3
"""Usage: tools/make_ipad_svgs.py assets/. Draws the iPad Air diagrams for the README, in the style of make_scoreboard_svg.py:
  scoreboard-ipad.svg  first run -> now per size class (log scale) and one Moon-Beam token on the iPad (E149)
  devices.svg          best speed per size class, TV box vs iPad Air
  tradeoff-ipad.svg    speed against perplexity cost for the two MoE models on the iPad (E154, E159-E160)
Every number comes from EXPERIMENT.md section 21 (TV numbers from the scoreboard, sections 11-20)."""
import math, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else 'assets'
STYLE = '''<style>
  .bg{fill:#ffffff} .ink{fill:#1f2328} .mut{fill:#59636e} .grid{stroke:#d8dee4} .bar{stroke:#2a78d6} .band{fill:#2a78d6;fill-opacity:.07}
  .start{fill:#ffffff;stroke:#8c959f} .now{fill:#2a78d6} .alt{fill:#ffffff;stroke:#2a78d6} .gain{fill:#1a7f37} .bad{fill:#cf222e}
  .s1{fill:#2a78d6} .s2{fill:#eb6834} .s3{fill:#1baf7a} .s4{fill:#eda100} .s5{fill:#e87ba4}
  .k1{stroke:#2a78d6} .k2{stroke:#eb6834}
  @media (prefers-color-scheme: dark){
    .bg{fill:#0d1117} .ink{fill:#e6edf3} .mut{fill:#9198a1} .grid{stroke:#30363d} .bar{stroke:#3987e5} .band{fill:#3987e5;fill-opacity:.12}
    .start{fill:#0d1117;stroke:#9198a1} .now{fill:#3987e5} .alt{fill:#0d1117;stroke:#3987e5} .gain{fill:#3fb950} .bad{fill:#f85149}
    .s1{fill:#3987e5} .s2{fill:#d95926} .s3{fill:#199e70} .s4{fill:#c98500} .s5{fill:#d55181}
    .k1{stroke:#3987e5} .k2{stroke:#d95926}
  }
  text{font-size:12px}
</style>'''
FONT = 'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif"'


def svg(W, H, body):
    return '\n'.join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" {FONT}>', STYLE,
                      f'<rect class="bg" width="{W}" height="{H}" rx="8"/>'] + body + ['</svg>']) + '\n'


def logx(lo, hi, L, R, W):
    a, b = math.log10(lo), math.log10(hi)
    return lambda v: L + (math.log10(v) - a) / (b - a) * (W - L - R)


# ---------------------------------------------------------------- 1. scoreboard
# class, model, first run (None = stock llama.cpp can't load it), first label, now, now label, faster setting (value, label), gain text
BEST = [
    ("1.7B", "Bonsai-1.7B (1-bit)", 2.75, "2.75", 23.7, "23.7-25.8", None, "×8.6"),
    ("4B", "Bonsai-4B (1-bit)", None, None, 9.4, "9.4-11.4", None, "fits 1 GB"),
    ("8B dense", "Bonsai-8B (1-bit)", None, None, 0.08, "0.08", None, "flash-bound"),
    ("8B MoE", "Moon-Beam-8B-A1B", 1.59, "1.6", 3.5, "3.5", (5.0, "5.0 at 4 experts"), "×2.2"),
    ("16B MoE", "Ling-mini-2.0", None, None, 2.56, "2.6", (4.5, "4.5 bias 0.3"), "new"),
]
ANATOMY = [("waiting on flash", 130, 2), ("Q4_K non-expert weights", 109, 1), ("Q2_K experts", 60, 3), ("router", 14, 4), ("delta-net, other", 11, 5)]


def scoreboard():
    W, L, R, T, rowH = 760, 168, 132, 96, 42
    x = logx(0.05, 40, L, R, W)
    bottom = T + len(BEST) * rowH
    aT = bottom + 70
    H = aT + 112
    o = []; a = o.append
    a(f'<text class="ink" x="16" y="26" style="font-size:16px;font-weight:600">Best model per size class: first run on the iPad Air → now</text>')
    a(f'<text class="mut" x="16" y="44">generation speed, tokens per second (log scale); ○ first run, ● now at full quality, ◇ a faster setting that costs quality</text>')
    a(f'<text class="mut" x="16" y="60">✗ = stock llama.cpp can\'t load it: iOS 12 lets a process map at most ~1.7 GB</text>')
    for t in (0.1, 0.3, 1, 3, 10, 30):
        a(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T-8}" y2="{bottom}" stroke-width="1"/>')
        a(f'<text class="mut" x="{x(t):.1f}" y="{T-14}" text-anchor="middle">{t}</text>')
    for i, (cls, model, s, sl, n, nl, alt, gain) in enumerate(BEST):
        cy = T + i * rowH + rowH / 2
        a(f'<text class="ink" x="16" y="{cy-3:.1f}" style="font-size:13px;font-weight:600">{cls}</text>')
        a(f'<text class="mut" x="16" y="{cy+12:.1f}" style="font-size:11px">{model}</text>')
        if s is not None:
            a(f'<line class="bar" x1="{x(s):.1f}" x2="{x(n):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke-width="3" stroke-linecap="round"/>')
            a(f'<circle class="start" cx="{x(s):.1f}" cy="{cy:.1f}" r="5" stroke-width="2"/>')
            a(f'<text class="mut" x="{x(s)-10:.1f}" y="{cy+4:.1f}" text-anchor="end">{sl}</text>')
        if cls.endswith("MoE"):
            a(f'<text class="bad" x="{x(0.05)+4:.1f}" y="{cy+4:.1f}" style="font-weight:600">✗</text>')
        if alt:
            av, al = alt
            a(f'<line class="bar" x1="{x(n):.1f}" x2="{x(av):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke-width="1.5" stroke-dasharray="3 3"/>')
            a(f'<rect class="alt" x="{x(av)-5:.1f}" y="{cy-5:.1f}" width="10" height="10" transform="rotate(45 {x(av):.1f} {cy:.1f})" stroke-width="2"/>')
            a(f'<text class="mut" x="{x(av)+11:.1f}" y="{cy+4:.1f}" style="font-size:11px">{al}</text>')
            a(f'<text class="ink" x="{x(n):.1f}" y="{cy-10:.1f}" text-anchor="middle" style="font-weight:600">{nl}</text>')
        else:
            a(f'<text class="ink" x="{x(n)+11:.1f}" y="{cy+4:.1f}" style="font-weight:600">{nl}</text>')
        a(f'<circle class="now" cx="{x(n):.1f}" cy="{cy:.1f}" r="6"/>')
        a(f'<text class="gain" x="{W-16}" y="{cy+4:.1f}" text-anchor="end" style="font-weight:600">{gain}</text>')
    a(f'<text class="ink" x="16" y="{aT-26}" style="font-size:14px;font-weight:600">Where one Moon-Beam-8B-A1B token goes on the iPad (about 0.3 s at 3.35 tok/s, E149)</text>')
    total = sum(m for _, m, _ in ANATOMY); bx, bw = 16, W - 32; cx = bx
    for i, (label, ms, slot) in enumerate(ANATOMY):
        w = ms / total * bw
        a(f'<rect class="s{slot}" x="{cx:.1f}" y="{aT-10}" width="{max(w-2,1):.1f}" height="26" rx="{4 if i in (0, len(ANATOMY)-1) else 0}"/>')
        if w > 60:
            a(f'<text class="ink" x="{cx+2:.1f}" y="{aT+36}" style="font-weight:600">~{ms} ms</text>')
            a(f'<text class="mut" x="{cx+2:.1f}" y="{aT+52}" style="font-size:11px">{label}</text>')
        cx += w
    a(f'<text class="mut" x="16" y="{aT+76}" style="font-size:11px">Yellow: the f32 router, ~14 ms. Pink: gated delta-net and the rest, ~11 ms. About 53 MB of experts are read per token at ~266 MB/s:</text>')
    a(f'<text class="mut" x="16" y="{aT+92}" style="font-size:11px">the flash and the 2-core CPU are both near their limits, and a bigger expert pool panicked the kernel.</text>')
    return svg(W, H, o)


# ---------------------------------------------------------------- 2. devices
# class, TV (model, tok/s), iPad (model, tok/s, note)
DEV = [
    ("2B class", ("Qwen3.5-2B", 7.7), ("Bonsai-1.7B, 1-bit", 23.7, "")),
    ("4B", ("Ternary-Bonsai-4B", 4.3), ("Bonsai-4B, 1-bit", 9.4, "")),
    ("8B dense", ("Bonsai-8B, 1-bit", 5.2), ("Bonsai-8B, 1-bit", 0.08, "doesn't fit, read from flash")),
    ("8B MoE", ("Moon-Beam-8B-A1B", 7.2), ("Moon-Beam-8B-A1B", 3.5, "")),
    ("16B MoE", ("Ling-mini-2.0, bias 0.3", 5.1), ("Ling-mini-2.0, bias 0.3", 4.5, "2.6 unbiased")),
]


def devices():
    W, L, R, T, rowH = 760, 120, 210, 112, 58
    x = logx(0.05, 40, L, R, W)
    bottom = T + len(DEV) * rowH
    H = bottom + 100
    o = []; a = o.append
    a(f'<text class="ink" x="16" y="26" style="font-size:16px;font-weight:600">Two 1 GB devices: best generation speed per size class</text>')
    a(f'<text class="mut" x="16" y="44">tokens per second in chat (log scale); the same ideas on both: expert pager, cache bias, pruned heads, 1-bit kernels</text>')
    a(f'<rect class="s1" x="16" y="58" width="12" height="12" rx="2"/><text class="ink" x="34" y="68">Xiaomi TV Box S (3rd Gen): 4× Cortex-A55 2.5 GHz, dotprod, ~1 GB usable, Android 14</text>')
    a(f'<rect class="s2" x="16" y="76" width="12" height="12" rx="2"/><text class="ink" x="34" y="86">iPad Air (2013): 2× Apple A7 1.4 GHz, no dotprod, 1 GB, iOS 12.5.7</text>')
    for t in (0.1, 0.3, 1, 3, 10, 30):
        a(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T-6}" y2="{bottom}" stroke-width="1"/>')
        a(f'<text class="mut" x="{x(t):.1f}" y="{bottom+16}" text-anchor="middle">{t}</text>')
    for i, (cls, tv, ip) in enumerate(DEV):
        y0 = T + i * rowH
        a(f'<text class="ink" x="16" y="{y0+26:.1f}" style="font-size:13px;font-weight:600">{cls}</text>')
        for j, (m, v, *note) in enumerate((tv, ip)):
            yy = y0 + 8 + j * 20
            a(f'<rect class="s{j+1}" x="{L}" y="{yy}" width="{max(x(v)-L,2):.1f}" height="16" rx="3"/>')
            extra = f' · {note[0]}' if note and note[0] else ''
            a(f'<text class="ink" x="{x(v)+6:.1f}" y="{yy+12}" style="font-weight:600">{v:g} <tspan class="mut" style="font-weight:400;font-size:11px">{m}{extra}</tspan></text>')
    notes = ["The 2B and 4B classes use different models: the iPad's 1-bit ones are faster than the TV's Q4_K and ternary ones,",
             "but lower quality (1.158 vs 0.909 and 1.073 vs ≈0.965 bits/byte). The TV wins where its 4 cores and dotprod kernels matter,",
             "and only its headless mode fits the dense 8B in RAM. Both run the 16B model at 4.5-5 tok/s with the same cache bias."]
    for k, line in enumerate(notes):
        a(f'<text class="mut" x="16" y="{bottom+40+16*k}" style="font-size:11px">{line}</text>')
    return svg(W, H, o)


# ---------------------------------------------------------------- 3. trade-off
MB = [  # experts, bias, tok/s, perplexity cost % vs top-8 unbiased (Mac, 3,072 tokens, E154)
    (8, 0.03, 3.53, 0.0), (8, 0.05, 3.71, 1.2), (7, 0.05, 4.03, 0.4), (7, 0.03, 3.8, 1.9), (6, 0.05, 4.32, 3.8), (6, 0.03, 4.25, 4.7),
    (5, 0.03, 4.54, 10.3), (4, 0.03, 5.02, 26.8)]
LM = [  # (iPad tinyppl, 506 tokens, E159-E160)
    (8, 0.0, 2.56, 0.0), (8, 0.1, 3.33, 2.6), (8, 0.2, 3.96, 17.9), (8, 0.3, 4.5, 8.5), (6, 0.3, 5.21, 16.9)]


def tradeoff():
    W, H, L, R, T, B = 760, 440, 70, 30, 80, 70
    xs = lambda v: L + (v - 2.0) / (5.6 - 2.0) * (W - L - R)
    ys = lambda p: T + (p - (-2)) / (30 - (-2)) * (H - T - B)
    o = []; a = o.append
    a(f'<text class="ink" x="16" y="26" style="font-size:16px;font-weight:600">Speed against quality: the MoE models on the iPad Air</text>')
    a(f'<text class="mut" x="16" y="44">each point is one setting: experts per token (k) and cache bias (b). Up = worse perplexity than unbiased top-8. Shaded: the 2-5 tok/s goal.</text>')
    a(f'<rect class="band" x="{xs(2.0):.1f}" y="{T}" width="{xs(5.0)-xs(2.0):.1f}" height="{H-T-B}"/>')
    for v in (2, 3, 4, 5):
        a(f'<line class="grid" x1="{xs(v):.1f}" x2="{xs(v):.1f}" y1="{T}" y2="{H-B}" stroke-width="1"/>')
        a(f'<text class="mut" x="{xs(v):.1f}" y="{H-B+16}" text-anchor="middle">{v}</text>')
    a(f'<text class="mut" x="{(L+W-R)/2:.0f}" y="{H-B+34}" text-anchor="middle">tokens per second (story prompt, iPad)</text>')
    for p in (0, 5, 10, 20, 30):
        a(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{ys(p):.1f}" y2="{ys(p):.1f}" stroke-width="1"/>')
        a(f'<text class="mut" x="{L-8}" y="{ys(p)+4:.1f}" text-anchor="end">+{p} %</text>')
    for pts, cls, name in ((MB, 1, "Moon-Beam-8B-A1B (Mac perplexity, 3,072 tokens)"), (LM, 2, "Ling-mini-2.0, 16B (iPad perplexity, 506 tokens: noisy)")):
        for k, b, v, p in pts:
            a(f'<circle class="s{cls}" cx="{xs(v):.1f}" cy="{ys(p):.1f}" r="6"/>')
            if (cls, k, b) == (1, 8, 0.05):   # crowded corner: label to the left
                a(f'<text class="mut" x="{xs(v)-2:.1f}" y="{ys(p)+20:.1f}" style="font-size:11px">k{k} b{b:g}</text>')
            else:
                a(f'<text class="mut" x="{xs(v)+9:.1f}" y="{ys(p)+4:.1f}" style="font-size:11px">k{k} b{b:g}</text>')
    a(f'<circle class="s1" cx="24" cy="{H-22}" r="6"/><text class="ink" x="36" y="{H-18}">Moon-Beam-8B-A1B (Mac perplexity, 3,072 tokens)</text>')
    a(f'<circle class="s2" cx="390" cy="{H-22}" r="6"/><text class="ink" x="402" y="{H-18}">Ling-mini-2.0 16B (iPad, 506 tokens: noisy)</text>')
    return svg(W, H, o)


os.makedirs(OUT, exist_ok=True)
for name, fn in (('scoreboard-ipad.svg', scoreboard), ('devices.svg', devices), ('tradeoff-ipad.svg', tradeoff)):
    open(os.path.join(OUT, name), 'w').write(fn())
    print('wrote', os.path.join(OUT, name))
