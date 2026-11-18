"""Shared SVG chart helpers + CSS (extracted from make_report.py)."""
import html, numpy as np
esc = lambda s: html.escape(str(s))

def nice(mx, n=4):
    c = [a * 10.0 ** e for e in range(-3, 6) for a in (1, 2, 2.5, 5)]
    return min(c, key=lambda s: abs(mx / s - n))
def lines(series, xlabel="", ylabel="", ymax=None, fmt="{:g}", height=300, xmax=None):
    """series: [(name, cls, dashed, [(x, y)])]"""
    W, H, L, Rr, T, B = 760, height, 52, 12, 14, 40
    pts = [p for _, _, _, ps in series for p in ps]
    if not pts: return '<p class="dim">no data yet</p>'
    xm = xmax or max(p[0] for p in pts) or 1; ym = ymax or max(p[1] for p in pts) * 1.1 or 1
    st = nice(ym); ym = np.ceil(ym / st) * st
    X = lambda x: L + (W - L - Rr) * x / xm; Y = lambda y: T + (H - T - B) * (1 - y / ym)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for t in np.arange(0, ym + st / 2, st): o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L-6}" y="{Y(t)+3:.1f}" class="tick" text-anchor="end">{esc(fmt.format(t))}</text>')
    xs = nice(xm, 5)
    for t in np.arange(0, xm + xs / 2, xs): o.append(f'<text x="{X(t):.1f}" y="{H-B+15}" class="tick" text-anchor="middle">{t:g}</text>')
    for name, cls, dashed, ps in series:
        if not ps: continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f} {Y(y):.1f}" for i, (x, y) in enumerate(ps))
        dash = 'stroke-dasharray="6 4"' if dashed else ''
        o.append(f'<path d="{d}" class="ln {cls}" {dash}><title>{esc(name)}</title></path>')
        if len(ps) <= 12: o += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" class="dot {cls}"><title>{esc(name)}: {esc(fmt.format(y))} at {x:g}</title></circle>' for x, y in ps]
    o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{H-B}" y2="{H-B}" class="axis"/>')
    if xlabel: o.append(f'<text x="{(W+L)/2:.0f}" y="{H-6}" class="tick" text-anchor="middle">{esc(xlabel)}</text>')
    if ylabel: o.append(f'<text x="11" y="{(H-B)/2:.0f}" class="tick" text-anchor="middle" transform="rotate(-90 11 {(H-B)/2:.0f})">{esc(ylabel)}</text>')
    o.append("</svg>")
    leg = '<div class="legend">' + "".join(f'<span><i class="sw {c} {"dash" if d else ""}"></i>{esc(n)}</span>' for n, c, d, _ in series) + "</div>"
    return leg + "".join(o)
def bars(groups, series, fmt="{:.1f}", ymax=None, ylabel="", yref=None, height=290):
    """series: [(name, cls, [values])]"""
    W, H, L, Rr, T, B = 760, height, 52, 12, 16, 62
    vals = [v for _, _, vs in series for v in vs if v is not None]
    if not vals: return '<p class="dim">no data yet</p>'
    mx = ymax or max(vals) * 1.15; st = nice(mx); mx = np.ceil(mx / st) * st
    Y = lambda v: T + (H - T - B) * (1 - v / mx); pw = W - L - Rr; gw = pw / len(groups); bw = min(34, gw * .8 / len(series))
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for t in np.arange(0, mx + st / 2, st): o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L-6}" y="{Y(t)+3:.1f}" class="tick" text-anchor="end">{esc(fmt.format(t))}</text>')
    if yref: o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{Y(yref[0]):.1f}" y2="{Y(yref[0]):.1f}" class="ref"/><text x="{W-Rr-2}" y="{Y(yref[0])-4:.1f}" class="tick reflab" text-anchor="end">{esc(yref[1])}</text>')
    for gi, g in enumerate(groups):
        gx = L + gw * gi + gw / 2
        o.append(f'<text x="{gx:.1f}" y="{H-B+14}" class="glab" text-anchor="end" transform="rotate(-28 {gx:.1f} {H-B+14})">{esc(g)}</text>')
        for si, (n, c, vs) in enumerate(series):
            v = vs[gi]
            if v is None: continue
            x = gx - len(series) * bw / 2 + si * bw
            o.append(f'<rect x="{x+1:.1f}" y="{Y(v):.1f}" width="{bw-2:.1f}" height="{max(0, H-B-Y(v)):.1f}" class="{c}" rx="2"><title>{esc(n)} · {esc(g)}: {esc(fmt.format(v))}</title></rect><text x="{x+bw/2:.1f}" y="{Y(v)-4:.1f}" class="val" text-anchor="middle">{esc(fmt.format(v))}</text>')
    o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{H-B}" y2="{H-B}" class="axis"/>')
    if ylabel: o.append(f'<text x="11" y="{(H-B)/2:.0f}" class="tick" text-anchor="middle" transform="rotate(-90 11 {(H-B)/2:.0f})">{esc(ylabel)}</text>')
    o.append("</svg>")
    return '<div class="legend">' + "".join(f'<span><i class="sw {c}"></i>{esc(n)}</span>' for n, c, _ in series) + "</div>" + "".join(o)
def tbl(head, rows, cls=""):
    return f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"
def fig(cap, body): return f'<div class="fig"><div class="cap">{cap}</div>{body}</div>'

CSS = """
:root{--bg:#fbfbfa;--fg:#1d2024;--dim:#6b7078;--card:#fff;--line:#e3e4e0;--grid:#ececE8;--s0:#2f6fdc;--s1:#e07b00;--s2:#139b6c;--s3:#a23cc4;--s4:#7a7f87;--s5:#c2185b;--s6:#8d6e63;--accent:#2f6fdc}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14161a;--fg:#e8e9ec;--dim:#9aa0a8;--card:#1b1e23;--line:#2c3037;--grid:#262a31;--s0:#6aa3ff;--s1:#ffa53d;--s2:#4cd6a0;--s3:#d98cf0;--s4:#9aa0a8;--s5:#ff7aa8;--s6:#c9a893;--accent:#6aa3ff}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e8e9ec;--dim:#9aa0a8;--card:#1b1e23;--line:#2c3037;--grid:#262a31;--s0:#6aa3ff;--s1:#ffa53d;--s2:#4cd6a0;--s3:#d98cf0;--s4:#9aa0a8;--s5:#ff7aa8;--s6:#c9a893;--accent:#6aa3ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 80px}h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}h2{font-size:21px;margin:44px 0 8px;border-top:1px solid var(--line);padding-top:26px}h3{font-size:16px;margin:26px 0 6px}
p{margin:8px 0}.sub{color:var(--dim);margin:0 0 18px}.dim{color:var(--dim)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:22px;letter-spacing:-.02em}.card span{color:var(--dim);font-size:13px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 12px 6px;margin:12px 0}.fig .cap{color:var(--dim);font-size:13px;margin:2px 4px 6px}.note{border-left:3px solid var(--accent);background:var(--card);padding:8px 12px;margin:12px 0;border-radius:0 8px 8px 0;font-size:14px}
.chart{width:100%;height:auto;display:block}.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--dim);stroke-width:1}.ref{stroke:var(--dim);stroke-width:1;stroke-dasharray:4 3}
.tick,.glab{fill:var(--dim);font-size:11px}.glab{font-size:11.5px;fill:var(--fg)}.val{fill:var(--fg);font-size:10px}.reflab{paint-order:stroke;stroke:var(--card);stroke-width:3px}
.s0{fill:var(--s0)}.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}.s4{fill:var(--s4)}.s5{fill:var(--s5)}.s6{fill:var(--s6)}path.ln.s5{stroke:var(--s5)}path.ln.s6{stroke:var(--s6)}i.s5{background:var(--s5)}i.s6{background:var(--s6)}path.ln{fill:none;stroke-width:2.2}path.ln.s0{stroke:var(--s0)}path.ln.s1{stroke:var(--s1)}path.ln.s2{stroke:var(--s2)}path.ln.s3{stroke:var(--s3)}path.ln.s4{stroke:var(--s4)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;margin:0 4px 6px}.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}.sw.dash{outline:1px dashed var(--fg);outline-offset:1px}
i.s0{background:var(--s0)}i.s1{background:var(--s1)}i.s2{background:var(--s2)}i.s3{background:var(--s3)}i.s4{background:var(--s4)}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card);border:1px solid var(--line)}th,td{padding:6px 9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--dim);font-weight:600;font-size:12px}table:not(.dense) td{white-space:normal}
ul{padding-left:20px}li{margin:4px 0}.lede{font-size:17px;line-height:1.55;margin:4px 0 12px}.kicker{color:var(--accent);font-weight:600;font-size:13px;letter-spacing:.06em;text-transform:uppercase;margin:0 0 6px}.take{border:1px solid var(--line);border-left:3px solid var(--s2);background:var(--card);padding:10px 14px;border-radius:0 8px 8px 0;margin:16px 0}ul.toc li{margin:7px 0}.cmp{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0}.cmp pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;font-size:12px;white-space:pre-wrap;word-break:break-word;margin:4px 0}@media (max-width:700px){.cmp{grid-template-columns:1fr}}a{color:var(--accent)}ol li{margin:6px 0}code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:13px}@media (max-width:600px){h1{font-size:24px}}"""


CSS = """
:root{--bg:#fbfbfa;--fg:#1d2024;--dim:#6b7078;--card:#fff;--line:#e3e4e0;--grid:#ececE8;--s0:#2f6fdc;--s1:#e07b00;--s2:#139b6c;--s3:#a23cc4;--s4:#7a7f87;--s5:#c2185b;--s6:#8d6e63;--accent:#2f6fdc}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14161a;--fg:#e8e9ec;--dim:#9aa0a8;--card:#1b1e23;--line:#2c3037;--grid:#262a31;--s0:#6aa3ff;--s1:#ffa53d;--s2:#4cd6a0;--s3:#d98cf0;--s4:#9aa0a8;--s5:#ff7aa8;--s6:#c9a893;--accent:#6aa3ff}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e8e9ec;--dim:#9aa0a8;--card:#1b1e23;--line:#2c3037;--grid:#262a31;--s0:#6aa3ff;--s1:#ffa53d;--s2:#4cd6a0;--s3:#d98cf0;--s4:#9aa0a8;--s5:#ff7aa8;--s6:#c9a893;--accent:#6aa3ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 80px}h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}h2{font-size:21px;margin:44px 0 8px;border-top:1px solid var(--line);padding-top:26px}h3{font-size:16px;margin:26px 0 6px}
p{margin:8px 0}.sub{color:var(--dim);margin:0 0 18px}.dim{color:var(--dim)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:22px;letter-spacing:-.02em}.card span{color:var(--dim);font-size:13px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 12px 6px;margin:12px 0}.fig .cap{color:var(--dim);font-size:13px;margin:2px 4px 6px}.note{border-left:3px solid var(--accent);background:var(--card);padding:8px 12px;margin:12px 0;border-radius:0 8px 8px 0;font-size:14px}
.chart{width:100%;height:auto;display:block}.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--dim);stroke-width:1}.ref{stroke:var(--dim);stroke-width:1;stroke-dasharray:4 3}
.tick,.glab{fill:var(--dim);font-size:11px}.glab{font-size:11.5px;fill:var(--fg)}.val{fill:var(--fg);font-size:10px}.reflab{paint-order:stroke;stroke:var(--card);stroke-width:3px}
.s0{fill:var(--s0)}.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}.s4{fill:var(--s4)}.s5{fill:var(--s5)}.s6{fill:var(--s6)}path.ln.s5{stroke:var(--s5)}path.ln.s6{stroke:var(--s6)}i.s5{background:var(--s5)}i.s6{background:var(--s6)}path.ln{fill:none;stroke-width:2.2}path.ln.s0{stroke:var(--s0)}path.ln.s1{stroke:var(--s1)}path.ln.s2{stroke:var(--s2)}path.ln.s3{stroke:var(--s3)}path.ln.s4{stroke:var(--s4)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;margin:0 4px 6px}.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}.sw.dash{outline:1px dashed var(--fg);outline-offset:1px}
i.s0{background:var(--s0)}i.s1{background:var(--s1)}i.s2{background:var(--s2)}i.s3{background:var(--s3)}i.s4{background:var(--s4)}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card);border:1px solid var(--line)}th,td{padding:6px 9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--dim);font-weight:600;font-size:12px}table:not(.dense) td{white-space:normal}
ul{padding-left:20px}li{margin:4px 0}.lede{font-size:17px;line-height:1.55;margin:4px 0 12px}.kicker{color:var(--accent);font-weight:600;font-size:13px;letter-spacing:.06em;text-transform:uppercase;margin:0 0 6px}.take{border:1px solid var(--line);border-left:3px solid var(--s2);background:var(--card);padding:10px 14px;border-radius:0 8px 8px 0;margin:16px 0}ul.toc li{margin:7px 0}.cmp{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0}.cmp pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;font-size:12px;white-space:pre-wrap;word-break:break-word;margin:4px 0}@media (max-width:700px){.cmp{grid-template-columns:1fr}}a{color:var(--accent)}ol li{margin:6px 0}code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:13px}@media (max-width:600px){h1{font-size:24px}}"""
