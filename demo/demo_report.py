"""Demo: yalla multi-configuration morphogenesis report.

Runs three distinct pair-wise ABM simulations — spring relaxation,
differential-adhesion cell sorting, and proliferation growth — and
generates an interactive HTML report with Three.js point-cloud viewers,
Plotly charts, a colored bigraph-viz architecture diagram, and a
navigable JSON tree of the composite document.
"""

import base64
import json
import os
import tempfile
import time as _time

import numpy as np
from process_bigraph import allocate_core

from pbg_yalla.processes import YallaProcess
from pbg_yalla.composites import make_yalla_document


# ── Simulation Configs ───────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'springs',
        'title': 'Spring Relaxation',
        'subtitle': 'Short-range Hookean relaxation (port of springs.cu)',
        'description': (
            'Three hundred agents start in a mildly stretched random sphere '
            'and relax under short-range Hookean springs. Each agent only feels '
            'springs from neighbours within r_cut = 1.0 (like yalla\'s finite '
            'support), pulling pairs towards a relaxed spacing of L_0 = 0.5. '
            'The cluster contracts into a dense packing. Port of yalla\'s '
            'examples/springs.cu with a finite interaction radius for '
            'numerical stability.'
        ),
        'config': {
            'n_cells': 300,
            'force_kernel': 'spring',
            'L_0': 0.5, 'r_cut': 1.0,
            'dt': 0.005, 'damping': 3.0,
            'init': 'random_sphere', 'init_dist': 0.6,
            'n_types': 1,
            'seed': 1,
        },
        'n_snapshots': 30,
        'total_time': 4.0,
        'camera': [4.0, 2.8, 4.0],
        'color_scheme': 'indigo',
        'color_by': 'nearest_distance',
    },
    {
        'id': 'sorting',
        'title': 'Differential Adhesion Sorting',
        'subtitle': 'Two cell types self-organize into inner/outer domains (sorting.cu)',
        'description': (
            'Two hundred agents of two types are initially mixed in a sphere. '
            'The differential_adhesion force kernel scales pair-wise attraction '
            'by a type-dependent strength factor (3x stronger between type-0 '
            'cells). Over time, the strongly-adherent type-0 cells aggregate '
            'into an inner core while type-1 cells are squeezed into an outer '
            'shell — the classic Steinberg differential-adhesion hypothesis, '
            'ported from yalla\'s examples/sorting.cu.'
        ),
        'config': {
            'n_cells': 200,
            'force_kernel': 'differential_adhesion',
            'r_min': 0.5, 'r_max': 1.0, 'dt': 0.01,
            'init': 'random_sphere', 'init_dist': 0.4,
            'n_types': 2, 'type_mode': 'mixed',
            'wall_radius': 2.5, 'wall_strength': 15.0,
            'damping': 2.0,
            'seed': 42,
        },
        'n_snapshots': 30,
        'total_time': 25.0,
        'camera': [4.5, 3.0, 4.5],
        'color_scheme': 'emerald',
        'color_by': 'type',
    },
    {
        'id': 'chemotaxis',
        'title': 'Chemotactic Aggregation',
        'subtitle': 'Type-selective migration up a morphogen gradient',
        'description': (
            'One hundred eighty agents of two types are dispersed in a '
            'cuboidal region. A morphogen source sits at (-4, 0, 0), '
            'producing an exponentially decaying concentration field '
            'c(x) = exp(-|x - source| / L) with L = 2.5. Only type-0 '
            'cells express the receptor, so only they climb the gradient; '
            'type-1 cells stay put unless dragged by adhesion. The responsive '
            'population migrates ballistically toward the source and '
            'aggregates into a compact cluster. This is the core motif '
            'behind morphogen-guided tissue patterning, convergent migration, '
            'and immune-cell homing — and complements yalla\'s '
            'examples/gradient.cu and examples/wnt.cu.'
        ),
        'config': {
            'n_cells': 180,
            'force_kernel': 'relu',
            'r_max': 1.0, 'dt': 0.05,
            'init': 'random_sphere', 'init_dist': 1.2,
            'n_types': 2, 'type_mode': 'mixed',
            'damping': 2.0,
            'chemotaxis_strength': 3.5,
            'chemotaxis_source_x': -4.0,
            'chemotaxis_source_y': 0.0,
            'chemotaxis_source_z': 0.0,
            'chemotaxis_decay_length': 2.5,
            'chemotaxis_responsive_types': '0',
            'seed': 9,
        },
        'n_snapshots': 35,
        'total_time': 14.0,
        'camera': [3.5, 4.0, 6.0],
        'color_scheme': 'amber',
        'color_by': 'type',
        'source_marker': {
            'position': [-4.0, 0.0, 0.0],
            'radius': 2.5,
        },
    },
    {
        'id': 'growth',
        'title': 'Proliferation Growth',
        'subtitle': 'Growing spheroid under ReLU adhesion + cell division (passive_growth.cu)',
        'description': (
            'A small cluster of 60 agents proliferates under the ReLU force '
            'kernel. On every substep each agent divides with a small '
            'probability; daughters inherit type with mild stochastic switching, '
            'producing a growing heterogeneous spheroid. Mirrors the '
            'proliferation kernel in yalla\'s examples/passive_growth.cu, where '
            'mesenchyme cells grow inside an epithelial shell.'
        ),
        'config': {
            'n_cells': 60,
            'force_kernel': 'relu',
            'r_max': 1.0, 'dt': 0.05,
            'init': 'relaxed_sphere', 'init_dist': 0.7,
            'init_relax_steps': 80,
            'n_types': 2, 'type_mode': 'hemispheres',
            'proliferation_rate': 0.12,
            'proliferation_mean_dist': 0.25,
            'proliferation_type_bias': 0.97,
            'n_max': 400,
            'damping': 1.5,
            'seed': 5,
        },
        'n_snapshots': 30,
        'total_time': 20.0,
        'camera': [6.0, 4.0, 6.0],
        'color_scheme': 'rose',
        'color_by': 'type',
    },
]


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'accent': '#34d399', 'text': '#064e3b'},
    'amber': {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#b45309',
              'accent': '#38bdf8', 'text': '#78350f'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'accent': '#fb7185', 'text': '#881337'},
}


def run_simulation(cfg_entry):
    """Run one configuration, returning snapshots and wall-clock runtime."""
    core = allocate_core()
    core.register_link('YallaProcess', YallaProcess)

    t0 = _time.perf_counter()
    proc = YallaProcess(config=cfg_entry['config'], core=core)
    state0 = proc.initial_state()
    snap0 = proc.snapshot()

    snapshots = [_snap(0.0, snap0, state0)]
    interval = cfg_entry['total_time'] / cfg_entry['n_snapshots']

    t = 0.0
    for _ in range(cfg_entry['n_snapshots']):
        result = proc.update({}, interval=interval)
        t += interval
        snap = proc.snapshot()
        snapshots.append(_snap(round(t, 3), snap, result))

    runtime = _time.perf_counter() - t0
    return snapshots, runtime


def _snap(t, raw, summary):
    return {
        'time': t,
        'positions': raw['positions'],
        'types': raw['types'],
        'n_cells': raw['n_cells'],
        'gyration_radius': summary['gyration_radius'],
        'mean_neighbor_distance': summary['mean_neighbor_distance'],
        'sorting_score': summary['sorting_score'],
        'type_radial_spread': summary['type_radial_spread'],
    }


def build_pbg_document(cfg_entry):
    """Build the PBG composite document dict shown in the JSON tree."""
    return make_yalla_document(
        interval=cfg_entry['total_time'] / cfg_entry['n_snapshots'],
        **cfg_entry['config'],
    )


def generate_bigraph_image(cfg_entry):
    """Render the composite architecture as a coloured bigraph-viz PNG."""
    from bigraph_viz import plot_bigraph

    # Simplified document — show only the key output ports to keep the
    # diagram legible.
    doc = {
        'yalla': {
            '_type': 'process',
            'address': 'local:YallaProcess',
            'config': {'force_kernel': cfg_entry['config']['force_kernel']},
            'interval': cfg_entry['total_time'] / cfg_entry['n_snapshots'],
            'inputs': {},
            'outputs': {
                'n_cells': ['stores', 'n_cells'],
                'gyration_radius': ['stores', 'gyration_radius'],
                'sorting_score': ['stores', 'sorting_score'],
                'type_radial_spread': ['stores', 'type_radial_spread'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {
                'n_cells': 'integer',
                'gyration_radius': 'float',
                'sorting_score': 'float',
                'type_radial_spread': 'float',
                'time': 'float',
            }},
            'inputs': {
                'n_cells': ['stores', 'n_cells'],
                'gyration_radius': ['stores', 'gyration_radius'],
                'sorting_score': ['stores', 'sorting_score'],
                'type_radial_spread': ['stores', 'type_radial_spread'],
                'time': ['global_time'],
            },
        },
    }

    cs = COLOR_SCHEMES[cfg_entry['color_scheme']]
    node_colors = {
        ('yalla',): cs['primary'],
        ('emitter',): '#8b5cf6',
        ('stores',): cs['light'],
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def compute_view_params(snapshots):
    """Compute a reasonable world extent from the first/final snapshots."""
    all_xyz = []
    for s in [snapshots[0], snapshots[len(snapshots) // 2], snapshots[-1]]:
        all_xyz.extend(s['positions'])
    arr = np.asarray(all_xyz)
    extent = float(np.max(np.linalg.norm(arr, axis=1)))
    return extent


def generate_html(sim_results, output_path):
    sections = []
    all_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        n0 = snapshots[0]['n_cells']
        n1 = snapshots[-1]['n_cells']
        extent = compute_view_params(snapshots)

        times = [s['time'] for s in snapshots]
        rgs = [s['gyration_radius'] for s in snapshots]
        ncells = [s['n_cells'] for s in snapshots]
        mean_nn = [s['mean_neighbor_distance'] for s in snapshots]
        spreads = [s['type_radial_spread'] for s in snapshots]

        # Build per-snapshot agent data for the viewer
        viewer_snaps = []
        for s in snapshots:
            viewer_snaps.append({
                'time': s['time'],
                'positions': s['positions'],
                'types': s['types'],
                'n_cells': s['n_cells'],
            })

        all_data[sid] = {
            'snapshots': viewer_snaps,
            'camera': cfg['camera'],
            'extent': extent,
            'color_by': cfg['color_by'],
            'scheme': cs,
            'source_marker': cfg.get('source_marker'),
            'charts': {
                'times': times,
                'gyration_radius': rgs,
                'n_cells': ncells,
                'mean_neighbor_distance': mean_nn,
                'type_radial_spread': spreads,
            },
        }

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        rg0, rg1 = rgs[0], rgs[-1]
        rg_pct = f'{rg1/rg0*100:.1f}' if rg0 > 0 else 'N/A'

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Agents (initial)</span><span class="metric-value">{n0:,}</span></div>
        <div class="metric"><span class="metric-label">Agents (final)</span><span class="metric-value">{n1:,}</span></div>
        <div class="metric"><span class="metric-label">Force Kernel</span><span class="metric-value" style="font-size:.85rem;">{cfg['config']['force_kernel']}</span></div>
        <div class="metric"><span class="metric-label">Gyration</span><span class="metric-value">{rg_pct}%</span><span class="metric-sub">{rg0:.2f} &rarr; {rg1:.2f}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
        <div class="metric"><span class="metric-label">Sim Time</span><span class="metric-value">{cfg['total_time']:.1f}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">3D Agent Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          <strong id="agent-count-{sid}">{n0}</strong> agents &middot;
          coloured by <strong>{cfg['color_by'].replace('_',' ')}</strong><br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(snapshots)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Time Series</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-rg-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-n-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-nn-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-spread-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections.append(section)

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}

    html = (
        _HTML_TEMPLATE
        .replace('__NAV_ITEMS__', nav_items)
        .replace('__SECTIONS__', ''.join(sections))
        .replace('__DATA_JSON__', json.dumps(all_data))
        .replace('__DOCS_JSON__', json.dumps(pbg_docs, indent=2))
    )

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ya||a Morphogenesis Simulation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }
.page-header { background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem; }
.page-header h1 { font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; letter-spacing:-.02em; }
.page-header p { color:#64748b; font-size:.95rem; max-width:760px; }
.nav { display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; flex-wrap:wrap; }
.nav-link { padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600; color:#334155;
             transition:all .15s; background:#fff; }
.nav-link:hover { transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }
.sim-section { padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }
.sim-header { display:flex; align-items:center; gap:1rem; margin-bottom:.8rem; padding-left:1rem; }
.sim-number { width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }
.sim-title { font-size:1.5rem; font-weight:700; color:#0f172a; }
.sim-subtitle { font-size:.9rem; color:#64748b; }
.sim-description { color:#475569; font-size:.92rem; margin-bottom:1.5rem; max-width:860px; }
.subsection-title { font-size:1.05rem; font-weight:600; color:#334155; margin:1.5rem 0 .8rem; }
.metrics-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }
.metric { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:.8rem; text-align:center; }
.metric-label { display:block; font-size:.68rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.25rem; }
.metric-value { display:block; font-size:1.25rem; font-weight:700; color:#1e293b; }
.metric-sub { display:block; font-size:.7rem; color:#94a3b8; margin-top:.15rem; }
.viewer-wrap { position:relative; background:#f1f5f9; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }
.mesh-canvas { width:100%; height:520px; display:block; cursor:grab; }
.mesh-canvas:active { cursor:grabbing; }
.viewer-info { position:absolute; top:.8rem; left:.8rem; background:rgba(255,255,255,.92);
                border:1px solid #e2e8f0; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#64748b; backdrop-filter:blur(4px); }
.viewer-info strong { color:#1e293b; }
.slider-controls { position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(241,245,249,.97));
                    padding:1.5rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }
.slider-controls label { font-size:.8rem; color:#64748b; }
.time-slider { flex:1; height:5px; }
.time-val { font-size:.95rem; font-weight:600; color:#334155; min-width:100px; text-align:right; }
.play-btn { background:#fff; border:1.5px solid; padding:.3rem .8rem; border-radius:7px;
             cursor:pointer; font-size:.8rem; font-weight:600; transition:all .15s; }
.play-btn:hover { transform:scale(1.05); }
.charts-row { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }
.chart-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }
.chart { height:280px; }
.pbg-row { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }
.pbg-col { min-width:0; }
.bigraph-img-wrap { background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }
.bigraph-img-wrap img { max-width:100%; height:auto; }
.json-tree { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:520px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }
.jt-key { color:#7c3aed; font-weight:600; }
.jt-str { color:#059669; }
.jt-num { color:#2563eb; }
.jt-bool { color:#d97706; }
.jt-null { color:#94a3b8; }
.jt-toggle { cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }
.jt-toggle:hover { color:#1e293b; }
.jt-collapsed { display:none; }
.jt-bracket { color:#64748b; }
.footer { text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }
@media(max-width:900px) {
  .charts-row,.pbg-row { grid-template-columns:1fr; }
  .sim-section,.page-header { padding:1.5rem; }
}
</style>
</head>
<body>

<div class="page-header">
  <h1>ya&#124;&#124;a Morphogenesis Simulation Report</h1>
  <p>Three pair-wise agent-based simulations wrapped as
  <strong>process-bigraph</strong> Processes. Each configuration is a direct
  port of a yalla example kernel (springs.cu, sorting.cu, passive_growth.cu)
  with interactive 3D agent visualisation, Plotly time series, the
  process-bigraph architecture diagram, and the full composite document tree.</p>
</div>

<div class="nav">__NAV_ITEMS__</div>

__SECTIONS__

<div class="footer">
  Generated by <strong>pbg-yalla</strong> &mdash;
  NumPy port of yalla's pair-wise ABM &mdash;
  wrapped as a process-bigraph Process
</div>

<script>
const DATA = __DATA_JSON__;
const DOCS = __DOCS_JSON__;

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\'' + id + '\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => { html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; });
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }
  if (typeof obj === 'object') {
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\'' + id + '\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    });
    html += '</div><span class="jt-bracket">}</span>';
    return html;
  }
  return String(obj);
}
function toggleJt(id) {
  const el = document.getElementById(id);
  if (el.classList.contains('jt-collapsed')) {
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  } else {
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }
}
Object.keys(DOCS).forEach(sid => {
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
});

// ─── Three.js Agent Viewers ───
const viewers = {};
const playStates = {};

function turbo(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.25) { const s = t/0.25; r=0.19; g=0.07+0.63*s; b=0.99-0.19*s; }
  else if (t < 0.5) { const s = (t-0.25)/0.25; r=0.19+0.11*s; g=0.70+0.15*s; b=0.80-0.55*s; }
  else if (t < 0.75) { const s = (t-0.5)/0.25; r=0.30+0.60*s; g=0.85-0.10*s; b=0.25-0.15*s; }
  else { const s = (t-0.75)/0.25; r=0.90+0.10*s; g=0.75-0.55*s; b=0.10-0.05*s; }
  return [r, g, b];
}

function typeColor(t, scheme) {
  if (t === 0) return hexToRgb(scheme.primary);
  if (t === 1) return hexToRgb(scheme.accent);
  return hexToRgb(scheme.dark || scheme.primary);
}
function hexToRgb(h) {
  const n = parseInt(h.replace('#',''), 16);
  return [((n>>16)&0xff)/255, ((n>>8)&0xff)/255, (n&0xff)/255];
}

function initViewer(sid) {
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 520;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0xf1f5f9);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.01, 200);
  cam.position.set(...d.camera);
  cam.lookAt(0, 0, 0);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.75); dl1.position.set(3,5,4); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0xcbd5e1, 0.35); dl2.position.set(-3,-2,-4); scene.add(dl2);

  const cageR = d.extent * 1.4;
  const cage = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.SphereGeometry(cageR, 16, 12)),
    new THREE.LineBasicMaterial({ color: 0xcbd5e1, transparent: true, opacity: 0.18 }));
  scene.add(cage);

  // Optional morphogen source marker: a translucent sphere at the source
  // position, sized by the decay length, to visualise the gradient field.
  if (d.source_marker) {
    const sm = d.source_marker;
    const srcGlowGeo = new THREE.SphereGeometry(sm.radius, 32, 24);
    const srcGlowMat = new THREE.MeshBasicMaterial({
      color: d.scheme.primary, transparent: true, opacity: 0.08,
      depthWrite: false,
    });
    const srcGlow = new THREE.Mesh(srcGlowGeo, srcGlowMat);
    srcGlow.position.set(sm.position[0], sm.position[1], sm.position[2]);
    scene.add(srcGlow);

    const srcCoreGeo = new THREE.SphereGeometry(sm.radius * 0.12, 24, 18);
    const srcCoreMat = new THREE.MeshBasicMaterial({
      color: d.scheme.primary,
    });
    const srcCore = new THREE.Mesh(srcCoreGeo, srcCoreMat);
    srcCore.position.set(sm.position[0], sm.position[1], sm.position[2]);
    scene.add(srcCore);
  }

  let maxN = 0;
  d.snapshots.forEach(s => { if (s.n_cells > maxN) maxN = s.n_cells; });

  // Render all agents through a single InstancedMesh — one geometry, one
  // material, per-instance colour set via setColorAt (which lazy-creates
  // the instanceColor attribute and wires it into the shader in r128).
  const sphereRadius = Math.max(0.12, d.extent * 0.045);
  const sphereGeo = new THREE.SphereGeometry(sphereRadius, 18, 14);
  const material = new THREE.MeshPhongMaterial({
    color: 0xffffff, shininess: 60, specular: 0xcccccc,
  });
  const imesh = new THREE.InstancedMesh(sphereGeo, material, maxN);
  imesh.frustumCulled = false;
  scene.add(imesh);

  const tmp = new THREE.Object3D();
  const hide = new THREE.Object3D(); hide.scale.set(0.001, 0.001, 0.001); hide.updateMatrix();
  const col = new THREE.Color();

  // Initialise colour attribute by setting every instance once.
  for (let i = 0; i < maxN; i++) {
    imesh.setMatrixAt(i, hide.matrix);
    imesh.setColorAt(i, col.setRGB(1, 1, 1));
  }
  imesh.instanceMatrix.needsUpdate = true;
  if (imesh.instanceColor) imesh.instanceColor.needsUpdate = true;

  function updateFrame(idx) {
    const snap = d.snapshots[idx];
    const n = snap.n_cells;

    let colorVals = null;
    if (d.color_by === 'nearest_distance') {
      colorVals = new Float32Array(n);
      let vmin = Infinity, vmax = -Infinity;
      for (let i = 0; i < n; i++) {
        let best = Infinity;
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const dx = snap.positions[i][0]-snap.positions[j][0];
          const dy = snap.positions[i][1]-snap.positions[j][1];
          const dz = snap.positions[i][2]-snap.positions[j][2];
          const d2 = dx*dx + dy*dy + dz*dz;
          if (d2 < best) best = d2;
        }
        colorVals[i] = Math.sqrt(best);
        if (colorVals[i] < vmin) vmin = colorVals[i];
        if (colorVals[i] > vmax) vmax = colorVals[i];
      }
      if (vmax - vmin < 1e-9) vmax = vmin + 1;
      for (let i = 0; i < n; i++) colorVals[i] = (colorVals[i] - vmin) / (vmax - vmin);
    }

    for (let i = 0; i < n; i++) {
      tmp.position.set(snap.positions[i][0], snap.positions[i][1], snap.positions[i][2]);
      tmp.scale.set(1, 1, 1);
      tmp.updateMatrix();
      imesh.setMatrixAt(i, tmp.matrix);
      let rgb;
      if (d.color_by === 'type') rgb = typeColor(snap.types[i], d.scheme);
      else if (d.color_by === 'nearest_distance') rgb = turbo(colorVals[i]);
      else rgb = hexToRgb(d.scheme.primary);
      col.setRGB(rgb[0], rgb[1], rgb[2]);
      imesh.setColorAt(i, col);
    }
    for (let i = n; i < maxN; i++) imesh.setMatrixAt(i, hide.matrix);

    imesh.instanceMatrix.needsUpdate = true;
    if (imesh.instanceColor) imesh.instanceColor.needsUpdate = true;

    const counter = document.getElementById('agent-count-' + sid);
    if (counter) counter.textContent = n;
  }

  updateFrame(0);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {
    const idx = parseInt(slider.value);
    updateFrame(idx);
    tval.textContent = 't = ' + d.snapshots[idx].time;
  });

  viewers[sid] = { renderer, scene, cam, controls, updateFrame, slider, tval };
  playStates[sid] = { playing:false, interval:null };

  function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, cam); }
  animate();
}

function togglePlay(sid) {
  const ps = playStates[sid]; const v = viewers[sid]; const d = DATA[sid];
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {
    btn.textContent = 'Pause';
    v.controls.autoRotate = false;
    ps.interval = setInterval(() => {
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= d.snapshots.length) idx = 0;
      v.slider.value = idx;
      v.updateFrame(idx);
      v.tval.textContent = 't = ' + d.snapshots[idx].time;
    }, 280);
  } else {
    btn.textContent = 'Play';
    v.controls.autoRotate = true;
    clearInterval(ps.interval);
  }
}

Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly Charts ───
const pLayout = {
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{ color:'#64748b', family:'-apple-system,sans-serif', size:11 },
  margin:{ l:55, r:15, t:35, b:40 },
  xaxis:{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0', title:{ text:'Time', font:{ size:10 } } },
  yaxis:{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' },
};
const pCfg = { responsive:true, displayModeBar:false };

Object.keys(DATA).forEach(sid => {
  const c = DATA[sid].charts; const sc = DATA[sid].scheme;
  Plotly.newPlot('chart-rg-'+sid, [{
    x:c.times, y:c.gyration_radius, type:'scatter', mode:'lines+markers',
    line:{ color:sc.primary, width:2 }, marker:{ size:4 },
    fill:'tozeroy', fillcolor:sc.primary + '22',
  }], {...pLayout, title:{ text:'Gyration Radius', font:{ size:12, color:'#334155' } },
    yaxis:{...pLayout.yaxis, title:{ text:'r_g', font:{ size:10 } } }, showlegend:false }, pCfg);

  Plotly.newPlot('chart-n-'+sid, [{
    x:c.times, y:c.n_cells, type:'scatter', mode:'lines+markers',
    line:{ color:sc.dark || sc.primary, width:2 }, marker:{ size:4 },
  }], {...pLayout, title:{ text:'Agent Count', font:{ size:12, color:'#334155' } },
    yaxis:{...pLayout.yaxis, title:{ text:'N', font:{ size:10 } } }, showlegend:false }, pCfg);

  Plotly.newPlot('chart-nn-'+sid, [{
    x:c.times, y:c.mean_neighbor_distance, type:'scatter', mode:'lines+markers',
    line:{ color:sc.accent, width:2 }, marker:{ size:4 },
  }], {...pLayout, title:{ text:'Mean Nearest-Neighbour Distance', font:{ size:12, color:'#334155' } },
    yaxis:{...pLayout.yaxis, title:{ text:'d_nn', font:{ size:10 } } }, showlegend:false }, pCfg);

  Plotly.newPlot('chart-spread-'+sid, [{
    x:c.times, y:c.type_radial_spread, type:'scatter', mode:'lines+markers',
    line:{ color:'#8b5cf6', width:2 }, marker:{ size:4 },
  }], {...pLayout, title:{ text:'Type Radial Spread (r_type1 - r_type0)', font:{ size:12, color:'#334155' } },
    yaxis:{...pLayout.yaxis, title:{ text:'&Delta;r', font:{ size:10 } } }, showlegend:false }, pCfg);
});

</script>
</body>
</html>"""


def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        snapshots, runtime = run_simulation(cfg)
        sim_results.append((cfg, (snapshots, runtime)))
        print(f'  Runtime: {runtime:.2f}s')
        print(f'  {len(snapshots)} snapshots, final n={snapshots[-1]["n_cells"]}')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)
    return output_path


if __name__ == '__main__':
    path = run_demo()
    import subprocess
    subprocess.run(['open', '-a', 'Safari', path])
