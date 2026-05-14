/* ============================================================
   app.js — System Map Application
   ============================================================ */
(function () {
  'use strict';

  // ── Constants ───────────────────────────────────────────
  const DATA_URL = 'data/system_map.json';
  // Three-tier semantic zoom: galaxy (5 super-clusters), system (22 domains), detail (modules).
  const ZOOM_GALAXY = 0.30;   // below: only the 5 galaxies are shown
  const ZOOM_SYSTEM = 1.00;   // below: galaxies fade, domains visible, modules unlabeled
  const STALE_DAYS  = 30;

  const DOMAIN_HUES = {
    data: 0, features: 16, signals: 33, strategies: 49, execution: 65,
    pipeline: 82, portfolio: 98, qa: 115, risk: 131, accounting: 147,
    paper: 164, events: 180, intel: 197, ml: 213, api: 229, ops: 246,
    compliance: 262, config: 279, utils: 295, reports: 311, experiments: 328,
  };

  const LOC_SCALE = [
    { max: 50,       color: '#1e3a5f' },
    { max: 150,      color: '#1d4ed8' },
    { max: 300,      color: '#7c3aed' },
    { max: 500,      color: '#be185d' },
    { max: Infinity, color: '#ef4444' },
  ];

  const SHORTCUTS = [
    ['/','Suche fokussieren'], ['Esc','Suche / Panel schließen'],
    ['C','Alle Domains einklappen'], ['E','Alle Domains ausklappen'],
    ['F','Fit to screen'], ['R','Reset Zoom (1:1)'],
    ['M','Mini-Map toggle'], ['H','Heat-Map toggle'],
    ['I','Import-Kette (2 Klicks)'], ['P','Print-Mode toggle'],
    ['D','Dark/Light toggle'], ['S','Sidebar toggle'], ['?','Diese Hilfe'],
    ['Ctrl+C (Knoten)','Pfad kopieren'], ['Doppelklick','Nachbarschaft isolieren'],
  ];

  // ── State ────────────────────────────────────────────────
  const state = {
    theme: 'dark',
    heatmap: false,
    pathMode: false,
    pathNodes: [],
    mapData: null,
    fuse: null,
  };

  // ── DOM Refs ────────────────────────────────────────────
  let cy;
  const $ = id => document.getElementById(id);

  // ── Cytoscape Stylesheet — Sake-Bot Mint × Navy ──────────
  // Cytoscape cannot read CSS custom properties, so the palette
  // is duplicated here. Keep in lockstep with tokens.css.
  const PAL = {
    void:       '#14141f',
    voidRaised: '#1c1c2e',
    ink:        '#f0f2f8',
    inkMuted:   '#7a84a0',
    hairline:   'rgba(255,255,255,0.10)',
    hairlineHot:'#42426a',
    gold:       '#6ad6b1',    /* MINT — primary accent (was orange) */
    goldHot:    '#7fe3c0',
    ember:      '#f3a93a',
    cyan:       '#4a9ce8',
    magenta:    '#e87ab6',
    violet:     '#9d7af0',
    ok:         '#6ad6b1',
    warn:       '#f3a93a',
    caution:    '#f3a93a',
    block:      '#ef5a59',
    unknown:    '#5c6278',
    duplicate:  '#9d7af0',
    orphan:     '#f3a93a',
    circular:   '#e87ab6',
  };

  const CY_STYLE = [
    // Base node — a celestial body. Modern sans-serif labels with strong outline.
    { selector: 'node', style: {
      'background-color': '#131826',
      'background-opacity': 1,
      'border-width': 1,
      'border-color': PAL.hairline,
      'label': 'data(label)',
      'font-family': 'Inter, "Segoe UI Variable", "SF Pro Display", system-ui, sans-serif',
      'font-size': 11,
      'font-weight': 500,
      'color': PAL.ink,
      'text-valign': 'bottom',
      'text-halign': 'center',
      'text-margin-y': 8,
      'text-max-width': 160,
      'text-overflow-wrap': 'whitespace',
      'text-outline-color': PAL.void,
      'text-outline-width': 3,
      'text-opacity': 0.85,
      'transition-property': 'opacity, border-color, background-color, border-width',
      'transition-duration': 220,
    }},

    // Galaxy compound — outer constellation cluster. Large, rounded, glowing ember border.
    { selector: 'node[type="galaxy"]', style: {
      'shape': 'round-rectangle',
      'background-color': PAL.gold,
      'background-opacity': 0.025,
      'border-color': PAL.gold,
      'border-width': 1.5,
      'border-opacity': 0.55,
      'border-style': 'solid',
      'font-family': 'Inter, "Segoe UI Variable", system-ui, sans-serif',
      'font-size': 22,
      'font-weight': 700,
      'color': PAL.goldHot,
      'text-valign': 'top',
      'text-halign': 'left',
      'text-margin-x': 20,
      'text-margin-y': -26,
      'text-transform': 'uppercase',
      'text-opacity': 0.95,
      'padding': 72,
      'min-width': 420, 'min-height': 280,
      'corner-radius': 18,
    }},

    // Domain compound — sub-constellation. Rounded, ember hairline.
    { selector: 'node[type="domain"]', style: {
      'shape': 'round-rectangle',
      'background-color': PAL.gold,
      'background-opacity': 0.02,
      'border-color': PAL.gold,
      'border-width': 1,
      'border-opacity': 0.40,
      'border-style': 'solid',
      'font-family': 'Inter, "Segoe UI Variable", system-ui, sans-serif',
      'font-size': 13,
      'font-weight': 600,
      'color': PAL.gold,
      'text-valign': 'top',
      'text-halign': 'left',
      'text-margin-x': 14,
      'text-margin-y': -14,
      'text-transform': 'uppercase',
      'text-opacity': 0.85,
      'padding': 38,
      'min-width': 220, 'min-height': 140,
      'corner-radius': 14,
    }},

    // Module — a luminous star. Strong glow scaled by LOC.
    { selector: 'node[type="module"]', style: {
      'shape': 'ellipse',
      'width':  'mapData(loc, 0, 800, 6, 22)',
      'height': 'mapData(loc, 0, 800, 6, 22)',
      'background-color': PAL.ink,
      'background-opacity': 0.95,
      'border-width': 0,
      'shadow-blur': 24,
      'shadow-color': PAL.ink,
      'shadow-opacity': 0.55,
      'shadow-offset-x': 0,
      'shadow-offset-y': 0,
    }},

    // External API — cyan diamond, like a distant signal source.
    { selector: 'node[type="external_api"]', style: {
      'shape': 'diamond',
      'width': 30, 'height': 30,
      'background-color': PAL.cyan,
      'background-opacity': 0.15,
      'border-color': PAL.cyan,
      'border-width': 1.5,
      'color': PAL.cyan,
      'shadow-blur': 14,
      'shadow-color': PAL.cyan,
      'shadow-opacity': 0.5,
    }},

    // Script — rounded pill, warm.
    { selector: 'node[type="script"]', style: {
      'shape': 'round-rectangle',
      'width': 80, 'height': 24,
      'background-color': PAL.gold,
      'background-opacity': 0.12,
      'border-color': PAL.gold,
      'border-width': 1,
      'color': PAL.goldHot,
      'font-size': 10,
      'corner-radius': 10,
    }},

    // Workflow — magenta hexagon (automation/trigger source).
    { selector: 'node[type="workflow"]', style: {
      'shape': 'hexagon',
      'width': 38, 'height': 38,
      'background-color': PAL.magenta,
      'background-opacity': 0.15,
      'border-color': PAL.magenta,
      'border-width': 1.5,
      'color': PAL.magenta,
      'shadow-blur': 14,
      'shadow-color': PAL.magenta,
      'shadow-opacity': 0.45,
    }},

    // Entry-point — rounded pill, warm-orange.
    { selector: 'node[type="entry_point"]', style: {
      'shape': 'round-rectangle',
      'width': 86, 'height': 28,
      'background-color': PAL.gold,
      'background-opacity': 0.18,
      'border-color': PAL.goldHot,
      'border-width': 1.5,
      'color': PAL.goldHot,
      'font-size': 10,
      'corner-radius': 12,
    }},

    // ── Status → luminous color of the stellar body ─────────
    { selector: 'node[type="module"][status="green"]',  style: { 'background-color': PAL.ok,      'shadow-color': PAL.ok,      'shadow-opacity': 0.7 } },
    { selector: 'node[type="module"][status="yellow"]', style: { 'background-color': PAL.warn,    'shadow-color': PAL.warn,    'shadow-opacity': 0.7 } },
    { selector: 'node[type="module"][status="orange"]', style: { 'background-color': PAL.caution, 'shadow-color': PAL.caution, 'shadow-opacity': 0.7 } },
    { selector: 'node[type="module"][status="red"]',    style: { 'background-color': PAL.block,   'shadow-color': PAL.block,   'shadow-opacity': 0.85 } },
    { selector: 'node[type="module"][status="gray"]',   style: { 'background-color': PAL.unknown, 'background-opacity': 0.5,   'shadow-opacity': 0 } },

    // Border status tint for non-module types
    { selector: 'node[type!="module"][status="green"]',  style: { 'border-color': PAL.ok } },
    { selector: 'node[type!="module"][status="yellow"]', style: { 'border-color': PAL.warn } },
    { selector: 'node[type!="module"][status="orange"]', style: { 'border-color': PAL.caution } },
    { selector: 'node[type!="module"][status="red"]',    style: { 'border-color': PAL.block, 'border-width': 2 } },
    { selector: 'node[type!="module"][status="gray"]',   style: { 'border-color': PAL.unknown, 'border-style': 'dashed' } },

    // ── Special markers — overlayed rings ───────────────────
    { selector: 'node[?orphan]', style: {
      'border-style': 'dashed',
      'border-width': 1.5,
      'border-color': PAL.orphan,
    }},
    { selector: 'node[duplicate_group]', style: {
      'border-color': PAL.duplicate,
      'border-style': 'dashed',
      'border-width': 1.5,
    }},
    { selector: 'node[?in_cycle]', style: {
      'border-color': PAL.circular,
      'border-width': 2,
    }},

    // Selected — a bright ember halo.
    { selector: 'node:selected', style: {
      'overlay-color': PAL.gold,
      'overlay-opacity': 0.26,
      'overlay-padding': 14,
      'border-color': PAL.gold,
      'border-width': 2,
      'shadow-blur': 32,
      'shadow-color': PAL.gold,
      'shadow-opacity': 0.85,
    }},

    // ── States ──────────────────────────────────────────────
    { selector: '.faded',       style: { 'opacity': 0.06 } },
    { selector: '.highlighted', style: { 'opacity': 1, 'z-index': 10 } },

    // ── Galaxy zoom (z < 0.30) — ONLY the 5 galaxy rectangles visible.
    //    Domains, modules, APIs, scripts, workflows, edges — all hidden.
    //    This is the "five constellations" view.
    { selector: '.zoom-galaxy node[type="domain"]',       style: { 'display': 'none' } },
    { selector: '.zoom-galaxy node[type="module"]',       style: { 'display': 'none' } },
    { selector: '.zoom-galaxy node[type="external_api"]', style: { 'display': 'none' } },
    { selector: '.zoom-galaxy node[type="script"]',       style: { 'display': 'none' } },
    { selector: '.zoom-galaxy node[type="workflow"]',     style: { 'display': 'none' } },
    { selector: '.zoom-galaxy node[type="entry_point"]',  style: { 'display': 'none' } },
    { selector: '.zoom-galaxy edge',                      style: { 'display': 'none' } },
    { selector: '.zoom-galaxy node[type="galaxy"]',       style: {
        'font-size': 28, 'border-opacity': 0.9, 'text-opacity': 1,
    }},

    // ── System zoom (0.30 – 1.00) — galaxies recede, domains come forward,
    //    modules visible as small dots without labels.
    { selector: '.zoom-system node[type="galaxy"]',       style: {
        'font-size': 18, 'border-opacity': 0.25, 'text-opacity': 0.5,
    }},
    { selector: '.zoom-system node[type="domain"]',       style: {
        'font-size': 14, 'border-opacity': 0.6, 'text-opacity': 0.9,
    }},
    { selector: '.zoom-system node[type="module"]',       style: { 'label': '', 'text-opacity': 0 } },
    { selector: '.zoom-system node[type="external_api"]', style: { 'label': '', 'text-opacity': 0 } },
    { selector: '.zoom-system node[type="script"]',       style: { 'label': '', 'text-opacity': 0 } },
    { selector: '.zoom-system node[type="workflow"]',     style: { 'label': '', 'text-opacity': 0 } },
    { selector: '.zoom-system edge',                      style: { 'opacity': 0.3 } },

    // ── Detail zoom (z >= 1.00) — galaxies nearly invisible, domains faded,
    //    module labels and edges fully visible.
    { selector: '.zoom-detail node[type="galaxy"]',       style: {
        'font-size': 14, 'border-opacity': 0.1, 'text-opacity': 0.25,
    }},
    { selector: '.zoom-detail node[type="domain"]',       style: {
        'font-size': 12, 'border-opacity': 0.3, 'text-opacity': 0.55,
    }},

    // ── Edges — curved beams between bodies ────────────────
    { selector: 'edge', style: {
      'curve-style': 'bezier',
      'control-point-step-size': 32,
      'target-arrow-shape': 'triangle-backcurve',
      'arrow-scale': 0.85,
      'target-arrow-color': PAL.hairline,
      'line-color': PAL.hairline,
      'width': 0.9,
      'opacity': 0.85,
      'transition-property': 'opacity, line-color, width',
      'transition-duration': 220,
    }},
    { selector: 'edge[kind="import"]', style: {
      'line-color': PAL.hairline,
      'target-arrow-color': PAL.hairline,
      'line-style': 'solid',
      'width': 0.9,
    }},
    { selector: 'edge[kind="api_call"]', style: {
      'line-color': PAL.cyan,
      'target-arrow-color': PAL.cyan,
      'line-style': 'dashed',
      'line-dash-pattern': [4, 3],
      'width': 1.1,
      'opacity': 0.7,
    }},
    { selector: 'edge[kind="data_flow"]', style: {
      'line-color': PAL.gold,
      'target-arrow-color': PAL.gold,
      'line-style': 'solid',
      'width': 1.5,
      'opacity': 0.85,
    }},
    { selector: 'edge[kind="trigger"]', style: {
      'line-color': PAL.magenta,
      'target-arrow-color': PAL.magenta,
      'line-style': 'dotted',
      'width': 1.2,
      'opacity': 0.75,
    }},
    { selector: 'edge[?circular]', style: {
      'line-color': PAL.circular,
      'target-arrow-color': PAL.circular,
      'line-style': 'dashed',
      'line-dash-pattern': [3, 3],
      'width': 1.4,
      'opacity': 0.9,
    }},
    { selector: 'edge.path-highlight', style: {
      'line-color': PAL.goldHot,
      'target-arrow-color': PAL.goldHot,
      'width': 2.5,
      'opacity': 1,
    }},
    { selector: 'edge:selected', style: {
      'line-color': PAL.gold,
      'target-arrow-color': PAL.gold,
      'width': 2,
      'opacity': 1,
    }},
  ];

  const FCOSE_CONFIG = {
    name: 'fcose', quality: 'default', animate: true,
    animationDuration: 500, fit: true, padding: 80,
    nodeDimensionsIncludeLabels: true, uniformNodeDimensions: false,
    packComponents: true,
    // Three-tier repulsion: galaxies spread widely, domains separate cleanly,
    // modules can pack within their domain.
    nodeRepulsion: node => {
      const t = node.data('type');
      if (t === 'galaxy') return 180000;
      if (t === 'domain') return 40000;
      return 7000;
    },
    idealEdgeLength: edge => {
      const sp = edge.source().data('parent');
      const tp = edge.target().data('parent');
      if (sp && sp === tp) return 50;          // same domain
      // Same galaxy? Walk one level up.
      const s = edge.cy().getElementById(sp || '');
      const t = edge.cy().getElementById(tp || '');
      if (s && t && s.data && s.data('parent') && s.data('parent') === t.data('parent')) return 140;
      return 260;                              // cross-galaxy
    },
    edgeElasticity: () => 0.25,
    nestingFactor: 0.08,
    numIter: 4000,
    tile: true, tilingPaddingVertical: 40, tilingPaddingHorizontal: 40,
    gravity: 0.1, gravityRange: 6.0, gravityCompound: 0.6, gravityRangeCompound: 3.0,
  };

  // ── Cytoscape Init ───────────────────────────────────────
  function initCy() {
    if (typeof cytoscape === 'undefined') {
      console.error('[SystemMap] cytoscape.min.js failed to load');
      hideSkeleton();
      const es = $('empty-state');
      if (es) es.classList.remove('hidden');
      return;
    }
    // Register extensions BEFORE creating the instance.
    if (typeof cytoscapeFcose !== 'undefined') cytoscape.use(cytoscapeFcose);
    if (typeof cytoscapePopper !== 'undefined') cytoscape.use(cytoscapePopper);

    cy = cytoscape({
      container: $('cy'),
      style: CY_STYLE,
      elements: [],
      wheelSensitivity: 0.3,
    });
    window.cy = cy;

    // expand-collapse self-registers when its <script> loads after cytoscape.min.js,
    // but does NOT expose a window.cytoscapeExpandCollapse global. Check the method.
    if (typeof cy.expandCollapse === 'function') {
      try {
        cy.expandCollapse({
          layoutBy: { ...FCOSE_CONFIG, animate: true, animationDuration: 350, numIter: 600 },
          animate: true,
          animationDuration: 300,
          fisheye: true,
          undoable: false,
          // Bundle multiple edges of the same kind between two collapsed compounds into one.
          // Drops 3438 import edges to ~80 domain↔domain bundles when all collapsed.
          groupEdgesOfSameTypeOnCollapse: true,
          allowNestedEdgeCollapse: true,
          // Cue is the small ± icon shown on collapsible compounds.
          cueEnabled: true,
          expandCollapseCuePosition: 'top-left',
          expandCollapseCueSize: 14,
          expandCollapseCueLineSize: 10,
          expandCollapseCueSensitivity: 1,
        });
      } catch (err) {
        console.warn('[SystemMap] expand-collapse init failed:', err);
      }
    } else {
      console.warn('[SystemMap] cy.expandCollapse missing — extension did not register');
    }
  }

  // ── Data Loading ─────────────────────────────────────────
  async function loadData() {
    // Prefer embedded data (works on file://)
    if (window.SYSTEM_MAP_DATA) {
      processData(window.SYSTEM_MAP_DATA);
      return;
    }
    // Fallback: fetch (works on http://)
    try {
      const res = await fetch(DATA_URL);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      processData(data);
    } catch (err) {
      console.error('[SystemMap] loadData failed:', err);
      hideSkeleton();
      $('empty-state').classList.remove('hidden');
    }
  }

  function processData(data) {
    state.mapData = data;
    checkStale(data.meta);
    buildElements(data);
    updateMetaInfo(data.meta);
  }

  function buildElements(data) {
    if (!cy) return;

    // Defensive cleanup: Cytoscape throws on dangling edges / missing parents.
    const nodeIds = new Set(data.nodes.map(n => n.id));
    let droppedEdges = 0;
    let reparented = 0;
    const cleanNodes = data.nodes.map(n => {
      if (n.parent && !nodeIds.has(n.parent)) {
        reparented++;
        const { parent, ...rest } = n;
        return rest;
      }
      return n;
    });
    const cleanEdges = data.edges.filter(e => {
      const ok = nodeIds.has(e.source) && nodeIds.has(e.target);
      if (!ok) droppedEdges++;
      return ok;
    });
    if (droppedEdges || reparented) {
      console.warn(`[SystemMap] self-heal: dropped ${droppedEdges} dangling edges, reparented ${reparented} nodes`);
    }

    const elements = [
      ...cleanNodes.map(n => ({ data: n })),
      ...cleanEdges.map(e => ({ data: e })),
    ];
    try {
      cy.add(elements);
    } catch (err) {
      console.error('[SystemMap] cy.add failed:', err);
      hideSkeleton();
      const es = $('empty-state');
      if (es) {
        es.classList.remove('hidden');
        const pre = es.querySelector('.empty-state__pre');
        if (pre) pre.textContent = `[ ⚠ ]  Karte konnte nicht gerendert werden.\n\n${err.message || err}`;
      }
      return;
    }

    const onDone = () => {
      hideSkeleton();
      initFuse(data.nodes);
      initZoom();
      DeepLink.init();
      MiniMap.init();
      // Collapse all compounds by default — 898 nodes is unreadable as a flat view.
      // User drills down by clicking domains or via Expand-All toolbar button.
      try {
        const api = cy.expandCollapse('get');
        if (api && typeof api.collapseAll === 'function') {
          api.collapseAll();
        }
      } catch (err) {
        console.warn('[SystemMap] initial collapseAll failed:', err);
      }
    };

    let layoutCfg;
    try {
      cy.layout({ name: 'fcose', numIter: 1 }).destroy();
      layoutCfg = { ...FCOSE_CONFIG, quality: 'draft', numIter: 800, animationDuration: 200 };
    } catch (_) {
      console.warn('[SystemMap] fcose not available, falling back to cose');
      showToast('warning', 'fcose nicht verfügbar — cose-Fallback wird verwendet');
      layoutCfg = {
        name: 'cose', animate: true, animationDuration: 600,
        fit: true, padding: 40, nodeRepulsion: 400000,
        idealEdgeLength: 80, gravity: 1, numIter: 1000,
      };
    }

    const layout = cy.layout(layoutCfg);
    layout.on('layoutstop', onDone);
    layout.run();
  }

  function hideSkeleton() {
    const sk = $('loading-skeleton');
    if (sk) sk.classList.add('hidden');
  }

  function checkStale(meta) {
    if (!meta || !meta.generated_at) return;
    const days = Math.floor((Date.now() - new Date(meta.generated_at)) / 86_400_000);
    if (days > STALE_DAYS) {
      $('stale-days').textContent = days;
      $('stale-banner').classList.remove('hidden');
    }
  }

  function updateMetaInfo(meta) {
    if (!meta) return;
    document.title = `System Map — ${meta.node_count} Knoten`;

    // KPI strip: derive aggregates from current mapData.
    const data = state.mapData;
    const set = (id, value) => {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    };
    const fmt = n => (typeof n === 'number' ? n.toLocaleString('de-DE') : '—');

    set('kpi-nodes', fmt(meta.node_count));
    set('kpi-edges', fmt(meta.edge_count));

    if (data && Array.isArray(data.nodes)) {
      const orphans = data.nodes.filter(n => n.orphan).length;
      const cycles  = data.nodes.filter(n => n.in_cycle).length;
      set('kpi-orphan', fmt(orphans));
      set('kpi-cycles', fmt(cycles));
    }
    if (meta.source_commit) set('kpi-commit', meta.source_commit);
    if (meta.generated_at) {
      try {
        const d = new Date(meta.generated_at);
        const isoDate = d.toISOString().slice(0, 10);
        set('kpi-generated', isoDate);
      } catch (_) { /* ignore */ }
    }
  }

  // ── Fuse Search ──────────────────────────────────────────
  function initFuse(nodes) {
    if (typeof Fuse === 'undefined') return;
    state.fuse = new Fuse(nodes, {
      keys: ['label', 'path', 'purpose', { name: 'functions', weight: 0.3 }],
      threshold: 0.35, includeScore: true, minMatchCharLength: 2,
    });
  }

  // ── Semantic Zoom ────────────────────────────────────────
  function initZoom() {
    if (!cy) return;
    const apply = debounce(() => {
      const z = cy.zoom();
      if (z < ZOOM_GALAXY) {
        cy.removeClass('zoom-system zoom-detail');
        cy.addClass('zoom-galaxy');
      } else if (z < ZOOM_SYSTEM) {
        cy.removeClass('zoom-galaxy zoom-detail');
        cy.addClass('zoom-system');
      } else {
        cy.removeClass('zoom-galaxy zoom-system');
        cy.addClass('zoom-detail');
      }
    }, 80);
    cy.on('zoom', apply);
    apply();
  }

  // ── Tooltips (Tippy) ────────────────────────────────────
  function initTooltips() {
    if (!cy || typeof tippy === 'undefined') return;

    cy.on('mouseover', 'node', e => {
      const node = e.target;
      if (node._tippy) return;
      const d = node.data();
      const status = d.status || 'gray';
      const purpose = d.purpose ? `<div style="margin-top:4px;color:#9ca3af;font-size:10px">${d.purpose.slice(0,120)}${d.purpose.length > 120 ? '…' : ''}</div>` : '';
      const tests = d.tests_count != null ? `<span style="color:#6b7280">tests: ${d.tests_count}</span>` : '';
      const loc   = d.loc != null ? `<span style="color:#6b7280">  loc: ${d.loc}</span>` : '';
      const fanio = (d.fan_in != null && d.fan_out != null) ? `<span style="color:#6b7280">  ↙${d.fan_in} ↗${d.fan_out}</span>` : '';
      const orphan = d.orphan ? `<span style="color:#f59e0b"> ⚠ orphan</span>` : '';
      const cycle  = d.in_cycle ? `<span style="color:#ec4899"> ↻ cycle</span>` : '';
      const dup    = d.duplicate_group ? `<span style="color:#a855f7"> ⋯ dup</span>` : '';

      const content = `<div style="font:11px/1.6 'JetBrains Mono',monospace;max-width:240px">
        <div style="font-weight:600;color:#f9fafb">${d.label || d.id}</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:2px">${tests}${loc}${fanio}${orphan}${cycle}${dup}</div>
        ${purpose}
      </div>`;

      const ref = node.popperRef ? node.popperRef() : null;
      if (!ref) return;
      node._tippy = tippy(ref, {
        content,
        allowHTML: true,
        delay: [300, 0],
        placement: 'top',
        theme: 'sysmap',
        arrow: true,
        appendTo: document.body,
      });
      node._tippy.show();
    });

    cy.on('mouseout', 'node', e => {
      const node = e.target;
      if (node._tippy) { node._tippy.destroy(); node._tippy = null; }
    });
  }

  // ── Node Events ──────────────────────────────────────────
  function initNodeEvents() {
    if (!cy) return;

    // Click a galaxy → toggle expand/collapse of the whole cluster.
    cy.on('tap', 'node[type="galaxy"]', e => {
      const node = e.target;
      const api = cy.expandCollapse('get');
      if (api) {
        if (api.isCollapsed(node)) {
          api.expand(node);
        } else {
          api.collapse(node);
        }
      }
      // Always frame whatever is now visible inside this galaxy.
      const target = node.descendants().union(node);
      cy.animate({ fit: { eles: target, padding: 60 } }, { duration: 350, easing: 'ease-out-cubic' });
    });

    cy.on('tap', 'node[type="domain"]', e => {
      const node = e.target;
      const api = cy.expandCollapse('get');
      if (api) {
        api.isCollapsed(node) ? api.expand(node) : api.collapse(node);
      }
      cy.animate({ fit: { eles: node, padding: 50 } }, { duration: 300 });
    });

    cy.on('tap', 'node[type="module"], node[type="script"], node[type="workflow"], node[type="entry_point"]', e => {
      openDetailPanel(e.target);
      DeepLink.setHash(e.target.id());
    });

    cy.on('dblclick', 'node', e => {
      const node = e.target;
      const neighborhood = node.closedNeighborhood();
      cy.elements().addClass('faded');
      neighborhood.removeClass('faded').addClass('highlighted');
    });

    // Background tap → clear highlights. The 'core' selector is NOT valid in cytoscape;
    // listen unscoped and check e.target === cy instead.
    cy.on('tap', e => {
      if (e.target === cy) {
        cy.elements().removeClass('faded highlighted');
        if (state.pathMode) clearPathMode();
      }
    });

    cy.on('tap', 'node', e => {
      if (state.pathMode) handlePathClick(e.target);
    });
  }

  // ── Detail Panel ─────────────────────────────────────────
  window.openDetailPanel = function openDetailPanel(node) {
    const d = node.data();
    const panel = $('detail-panel');
    panel.classList.remove('detail-panel--closed');
    document.getElementById('app').classList.add('panel--open');

    $('detail-title').textContent = d.label || d.id;
    const statusEl = $('detail-status');
    statusEl.textContent = d.status || '';
    statusEl.className = `status-badge status-badge--${d.status || 'gray'}`;
    statusEl.setAttribute('aria-label', `Status: ${d.status}`);

    const stripe = $('detail-domain-stripe');
    stripe.style.background = domainColor(d.parent);

    renderMetaTab(d);
    renderImportsTab(d, 'out');
    renderImportsTab(d, 'in');
    renderApisTab(d);
    renderDuplicatesTab(d);
    switchTab('meta');
  };

  function closeDetailPanel() {
    $('detail-panel').classList.add('detail-panel--closed');
    document.getElementById('app').classList.remove('panel--open');
  }

  function renderMetaTab(d) {
    const el = $('tab-content-meta');
    const purpose = d.purpose ? `<div class="purpose-block">${escHtml(d.purpose)}</div>` : '';
    const metrics = `
      <div class="metrics-grid">
        <div class="metric-cell"><span class="metric-cell__value">${d.loc ?? '—'}</span><span class="metric-cell__label">LOC</span></div>
        <div class="metric-cell"><span class="metric-cell__value">${d.tests_count ?? '—'}</span><span class="metric-cell__label">Tests</span></div>
        <div class="metric-cell"><span class="metric-cell__value">${d.fan_in ?? '—'}</span><span class="metric-cell__label">Fan-In</span></div>
        <div class="metric-cell"><span class="metric-cell__value">${d.fan_out ?? '—'}</span><span class="metric-cell__label">Fan-Out</span></div>
        <div class="metric-cell"><span class="metric-cell__value">${d.complexity_score ?? '—'}</span><span class="metric-cell__label">Kompl.</span></div>
        <div class="metric-cell"><span class="metric-cell__value">${d.type_annotation_ratio != null ? Math.round(d.type_annotation_ratio * 100) + '%' : '—'}</span><span class="metric-cell__label">Types</span></div>
      </div>`;
    const path = d.path ? `<div class="path-block"><span>${escHtml(d.path)}</span>
      <button onclick="navigator.clipboard.writeText('${escHtml(d.path)}')" title="Kopieren">copy</button></div>` : '';
    const orphan = d.orphan ? `<div class="warn-notice">⚠ Orphan — keine eingehenden Imports gefunden</div>` : '';
    const cycle  = d.in_cycle ? `<div class="warn-notice" style="border-color:#ec4899;color:#ec4899">↻ Zirkulärer Import</div>` : '';
    const funcs  = d.functions && d.functions.length
      ? `<p class="import-section-title">Funktionen (${d.functions.length})</p>
         <code style="font:11px/1.8 var(--font-mono);color:var(--text-secondary);word-break:break-word">${d.functions.map(escHtml).join(', ')}</code>`
      : '';
    el.innerHTML = purpose + metrics + path + orphan + cycle + funcs;
  }

  function renderImportsTab(d, dir) {
    const el = $(dir === 'out' ? 'tab-content-imports-out' : 'tab-content-imports-in');
    if (!cy) { el.innerHTML = '<p style="color:var(--text-muted)">—</p>'; return; }
    const node = cy.$('#' + CSS.escape(d.id));
    const edges = dir === 'out' ? node.connectedEdges(`edge[source="${d.id}"]`) : node.connectedEdges(`edge[target="${d.id}"]`);
    const items = edges.map(e => {
      const other = dir === 'out' ? cy.$('#' + CSS.escape(e.data('target'))) : cy.$('#' + CSS.escape(e.data('source')));
      const od = other.data();
      return `<li class="import-item" tabindex="0" onclick="window.cy.$('#${CSS.escape(od.id)}').emit('tap')"
        onkeydown="if(event.key==='Enter')window.cy.$('#${CSS.escape(od.id)}').emit('tap')">
        <span class="status-dot status-dot--${od.status || 'gray'}"></span>
        <span class="search-result__label">${escHtml(od.label || od.id)}</span>
        <span class="import-count" style="font-size:10px;color:var(--text-muted)">${escHtml(e.data('kind'))}</span>
      </li>`;
    }).join('');
    const title = dir === 'out' ? 'Ausgehende Imports' : 'Eingehende Imports (ImportedBy)';
    const count = edges.length;
    const warn = dir === 'in' && count === 0 && !d.orphan
      ? `<div class="warn-notice">Keine eingehenden Imports — potentieller Orphan</div>` : '';
    el.innerHTML = `<p class="import-section-title">${title} (${count})</p><ul class="import-list">${items}</ul>${warn}`;
  }

  function renderApisTab(d) {
    const el = $('tab-content-apis');
    if (!cy) { el.innerHTML = '<p style="color:var(--text-muted)">—</p>'; return; }
    const node = cy.$('#' + CSS.escape(d.id));
    const apiEdges = node.connectedEdges('edge[kind="api_call"]');
    if (!apiEdges.length) {
      el.innerHTML = '<p style="color:var(--text-muted);font:12px var(--font-ui)">Keine API-Aufrufe</p>';
      return;
    }
    const items = apiEdges.map(e => {
      const target = cy.$('#' + CSS.escape(e.data('target')));
      const td = target.data();
      return `<li class="import-item" tabindex="0"
        onclick="window.cy.$('#${CSS.escape(td.id)}').emit('tap')">
        ◇ <span class="search-result__label">${escHtml(td.label || td.id)}</span>
      </li>`;
    }).join('');
    el.innerHTML = `<ul class="import-list">${items}</ul>`;
  }

  function renderDuplicatesTab(d) {
    const el = $('tab-content-duplicates');
    if (!d.duplicate_group) {
      el.innerHTML = '<p style="color:var(--text-muted);font:12px var(--font-ui)">Keine Duplikat-Gruppe</p>';
      return;
    }
    const groupId = d.duplicate_group;
    const members = state.mapData
      ? state.mapData.nodes.filter(n => n.duplicate_group === groupId)
      : [];
    const items = members.map(m => `
      <li class="import-item" tabindex="0" onclick="window.cy.$('#${CSS.escape(m.id)}').emit('tap')">
        <span class="status-dot status-dot--${m.status || 'gray'}"></span>
        <span class="search-result__label">${escHtml(m.label)}</span>
      </li>`).join('');
    el.innerHTML = `
      <p class="import-section-title" style="color:var(--marker-duplicate)">Duplikat-Gruppe: ${escHtml(groupId)}</p>
      <ul class="import-list">${items}</ul>
      <div class="warn-notice" style="border-color:var(--marker-duplicate);color:var(--marker-duplicate);margin-top:8px">
        ⚠ ${members.length} Module mit überlappender Funktionalität
      </div>`;
  }

  function switchTab(tabId) {
    document.querySelectorAll('.detail-tab').forEach(t => {
      const active = t.dataset.tab === tabId;
      t.classList.toggle('detail-tab--active', active);
      t.setAttribute('aria-selected', active);
    });
    document.querySelectorAll('.detail-tab-content').forEach(c => {
      c.classList.toggle('detail-tab-content--hidden', !c.id.includes(tabId));
    });
  }

  // ── Heat-Map Mode ────────────────────────────────────────
  function toggleHeatmap() {
    state.heatmap = !state.heatmap;
    const btn = $('tb-heatmap');
    btn && btn.classList.toggle('icon-btn--active', state.heatmap);
    if (state.heatmap) {
      cy.$('node[type="module"]').forEach(n => {
        const loc = n.data('loc') || 0;
        const col = LOC_SCALE.find(s => loc <= s.max)?.color || '#ef4444';
        n.style('background-color', col);
      });
      Legend.setMode('heatmap');
    } else {
      cy.$('node[type="module"]').forEach(n => n.removeStyle('background-color'));
      Legend.setMode('status');
    }
    showToast('info', state.heatmap ? 'Heat-Map: AN' : 'Heat-Map: AUS');
  }

  // ── Import Path Mode ─────────────────────────────────────
  function handlePathClick(node) {
    if (!state.pathMode) return;
    state.pathNodes.push(node);
    node.addClass('highlighted');
    if (state.pathNodes.length === 2) {
      computePath(state.pathNodes[0], state.pathNodes[1]);
      state.pathNodes = [];
    }
  }

  function computePath(src, tgt) {
    const result = cy.elements().aStar({ root: src, goal: tgt, weight: () => 1 });
    if (!result.found) {
      showToast('warning', `Kein Pfad von ${src.data('label')} → ${tgt.data('label')}`);
      clearPathMode();
      return;
    }
    cy.elements().addClass('faded');
    result.path.removeClass('faded').addClass('path-highlight highlighted');
    const pathLabels = result.path.filter('node').map(n => n.data('label')).join(' → ');
    showToast('info', pathLabels, 8000);
  }

  function togglePathMode() {
    state.pathMode = !state.pathMode;
    state.pathNodes = [];
    const btn = $('tb-path');
    btn && btn.classList.toggle('icon-btn--active', state.pathMode);
    if (state.pathMode) {
      showToast('info', 'Import-Kette: erstes Modul klicken…');
    } else {
      clearPathMode();
    }
  }

  function clearPathMode() {
    state.pathMode = false;
    state.pathNodes = [];
    if (cy) cy.elements().removeClass('faded highlighted path-highlight');
    const btn = $('tb-path');
    btn && btn.classList.remove('icon-btn--active');
  }

  // ── Filter System ────────────────────────────────────────
  function initFilters() {
    document.querySelectorAll('input[name="status"], input[name="type"]').forEach(inp => {
      inp.addEventListener('change', applyFilters);
    });
    ['filter-has-tests','filter-orphan','filter-duplicate','filter-circular'].forEach(id => {
      const el = $(id);
      if (el) el.addEventListener('change', applyFilters);
    });
    $('filter-reset')?.addEventListener('click', () => {
      document.querySelectorAll('.filter-group input').forEach(i => { i.checked = true; if(i.type==='checkbox'&&['filter-has-tests','filter-orphan','filter-duplicate','filter-circular'].includes(i.id)) i.checked = false; });
      applyFilters();
    });
  }

  function applyFilters() {
    if (!cy) return;
    const activeStatuses = [...document.querySelectorAll('input[name="status"]:checked')].map(i => i.value);
    const activeTypes    = [...document.querySelectorAll('input[name="type"]:checked')].map(i => i.value);
    const onlyTests    = $('filter-has-tests')?.checked;
    const onlyOrphan   = $('filter-orphan')?.checked;
    const onlyDup      = $('filter-duplicate')?.checked;
    const onlyCircular = $('filter-circular')?.checked;

    cy.nodes().forEach(n => {
      const d = n.data();
      let show = activeStatuses.includes(d.status || 'gray') && activeTypes.includes(d.type || 'module');
      if (onlyTests    && !(d.tests_count > 0)) show = false;
      if (onlyOrphan   && !d.orphan)            show = false;
      if (onlyDup      && !d.duplicate_group)   show = false;
      if (onlyCircular && !d.in_cycle)          show = false;
      n.style('display', show ? 'element' : 'none');
    });
    updateFilterChips(activeStatuses, activeTypes, { onlyTests, onlyOrphan, onlyDup, onlyCircular });
  }

  function updateFilterChips(statuses, types, special) {
    const container = $('filter-chips');
    if (!container) return;
    const chips = [];
    const allStatuses = ['green','yellow','orange','red','gray'];
    const allTypes    = ['domain','module','external_api','script','workflow'];
    allStatuses.forEach(s => { if (!statuses.includes(s)) chips.push(`${s} ×`); });
    allTypes.forEach(t => { if (!types.includes(t)) chips.push(`${t} ×`); });
    if (special.onlyTests)    chips.push('mit Tests ×');
    if (special.onlyOrphan)   chips.push('Orphans ×');
    if (special.onlyDup)      chips.push('Dups ×');
    if (special.onlyCircular) chips.push('Zirkular ×');
    container.innerHTML = chips.map(c => `<span class="chip chip--active">${c}</span>`).join('');
  }

  // ── Search ───────────────────────────────────────────────
  function initSearch() {
    const input    = $('search-input');
    const dropdown = $('search-dropdown');
    if (!input || !dropdown) return;

    input.addEventListener('input', () => {
      const q = input.value.trim();
      if (!q || !state.fuse) { dropdown.hidden = true; return; }
      const results = state.fuse.search(q, { limit: 8 });
      renderSearchDropdown(results, dropdown);
    });

    input.addEventListener('keydown', e => {
      if (e.key === 'Escape') { dropdown.hidden = true; input.blur(); }
      if (e.key === 'Enter')  { jumpToFirstResult(dropdown); }
      if (e.key === 'ArrowDown') moveFocus(dropdown, 1);
      if (e.key === 'ArrowUp')   moveFocus(dropdown, -1);
    });

    document.addEventListener('click', e => {
      if (!input.contains(e.target) && !dropdown.contains(e.target)) dropdown.hidden = true;
    });
  }

  function renderSearchDropdown(results, dropdown) {
    if (!results.length) { dropdown.hidden = true; return; }
    dropdown.innerHTML = results.map(r => {
      const d = r.item;
      const domainId = d.parent || '';
      const domainName = domainId.replace('domain:', '');
      return `<li role="option" class="search-result" tabindex="-1" data-id="${escHtml(d.id)}">
        <span class="status-dot status-dot--${d.status || 'gray'}"></span>
        <span class="search-result__label">${escHtml(d.label)}</span>
        <span class="search-result__domain">${escHtml(domainName)}</span>
      </li>`;
    }).join('');
    dropdown.hidden = false;
    dropdown.querySelectorAll('.search-result').forEach(li => {
      li.addEventListener('click', () => jumpToNode(li.dataset.id));
      li.addEventListener('keydown', e => { if (e.key === 'Enter') jumpToNode(li.dataset.id); });
    });
  }

  function jumpToNode(id) {
    if (!cy) return;
    const node = cy.$('#' + CSS.escape(id));
    if (!node.length) return;
    cy.animate({ fit: { eles: node, padding: 80 } }, { duration: 400 });
    if (id.startsWith('module:')) openDetailPanel(node);
    $('search-dropdown').hidden = true;
    $('search-input').value = '';
    DeepLink.setHash(id);
  }

  function jumpToFirstResult(dropdown) {
    const first = dropdown.querySelector('.search-result');
    if (first) jumpToNode(first.dataset.id);
  }

  function moveFocus(dropdown, dir) {
    const items = [...dropdown.querySelectorAll('.search-result')];
    const cur   = items.indexOf(document.activeElement);
    const next  = Math.max(0, Math.min(items.length - 1, cur + dir));
    items[next]?.focus();
  }

  // ── Toolbar ──────────────────────────────────────────────
  function initToolbar() {
    $('tb-sidebar')?.addEventListener('click', toggleSidebar);
    $('sidebar-toggle')?.addEventListener('click', toggleSidebar);
    $('tb-fit')?.addEventListener('click', () => cy?.fit(undefined, 40));
    $('tb-reset')?.addEventListener('click', () => cy?.zoom(1));
    $('tb-collapse')?.addEventListener('click', () => {
      const api = cy?.expandCollapse?.('get');
      if (api && typeof api.collapseAll === 'function') {
        api.collapseAll();
        cy.fit(undefined, 60);
      }
    });
    $('tb-expand')?.addEventListener('click', () => {
      const api = cy?.expandCollapse?.('get');
      if (api && typeof api.expandAll === 'function') {
        api.expandAll();
        cy.fit(undefined, 60);
      }
    });
    $('tb-minimap')?.addEventListener('click', () => MiniMap.toggle());
    $('tb-heatmap')?.addEventListener('click', toggleHeatmap);
    $('tb-path')?.addEventListener('click', togglePathMode);
    $('tb-export')?.addEventListener('click', exportPng);
    $('tb-print')?.addEventListener('click', () => window.print());
    $('tb-theme')?.addEventListener('click', toggleTheme);
    $('tb-about')?.addEventListener('click', openShortcutModal);
    $('detail-close')?.addEventListener('click', closeDetailPanel);
    $('stale-banner__close')?.addEventListener('click', () => $('stale-banner').classList.add('hidden'));
    $('shortcut-modal-close')?.addEventListener('click', closeShortcutModal);
    $('shortcut-modal')?.addEventListener('click', e => { if (e.target === $('shortcut-modal')) closeShortcutModal(); });

    document.querySelectorAll('.detail-tab').forEach(tab => {
      tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
  }

  function toggleSidebar() {
    document.getElementById('app').classList.toggle('sidebar--collapsed');
  }

  function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', state.theme);
    try { localStorage.setItem('sysmap_theme', state.theme); } catch(_) {}
  }

  function exportPng() {
    if (!cy) return;
    const png = cy.png({ full: true, scale: 2, bg: '#0a0f1e' });
    const a   = document.createElement('a');
    a.href = png; a.download = 'system_map.png'; a.click();
  }

  function openShortcutModal() {
    const modal = $('shortcut-modal');
    const table = $('shortcut-table');
    if (table && !table.innerHTML) {
      table.innerHTML = SHORTCUTS.map(([k, v]) =>
        `<tr><td>${escHtml(k)}</td><td>${escHtml(v)}</td></tr>`).join('');
    }
    modal.classList.remove('modal-overlay--hidden');
    modal.focus();
  }

  function closeShortcutModal() {
    $('shortcut-modal').classList.add('modal-overlay--hidden');
  }

  // ── Keyboard ─────────────────────────────────────────────
  function initKeyboard() {
    document.addEventListener('keydown', e => {
      const tag = document.activeElement?.tagName;
      const inInput = tag === 'INPUT' || tag === 'TEXTAREA';

      if (!inInput) {
        switch (e.key.toLowerCase()) {
          case '/': e.preventDefault(); $('search-input')?.focus(); break;
          case 'f': cy?.fit(undefined, 40); break;
          case 'r': cy?.zoom(1); break;
          case 'm': MiniMap.toggle(); break;
          case 'h': toggleHeatmap(); break;
          case 'i': togglePathMode(); break;
          case 'p': window.print(); break;
          case 'd': toggleTheme(); break;
          case 's': toggleSidebar(); break;
          case 'c': {
            const api = cy?.expandCollapse?.('get');
            if (api?.collapseAll) { api.collapseAll(); cy.fit(undefined, 60); }
            break;
          }
          case 'e': {
            const api = cy?.expandCollapse?.('get');
            if (api?.expandAll) { api.expandAll(); cy.fit(undefined, 60); }
            break;
          }
          case '?': openShortcutModal(); break;
        }
        if (e.ctrlKey && e.key.toLowerCase() === 'c' && cy?.$(":selected").length) {
          const path = cy.$(':selected').first().data('path') || '';
          navigator.clipboard.writeText(path).then(() => showToast('info', 'Pfad kopiert'));
        }
      }
      if (e.key === 'Escape') {
        $('search-dropdown').hidden = true;
        closeDetailPanel();
        closeShortcutModal();
        if (state.pathMode) clearPathMode();
        cy?.elements().removeClass('faded highlighted');
      }
    });
  }

  // ── Toast ────────────────────────────────────────────────
  window.showToast = function showToast(type, text, duration = 4000) {
    const container = $('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.textContent = text;
    container.appendChild(toast);
    const remove = () => {
      toast.classList.add('toast--out');
      toast.addEventListener('animationend', () => toast.remove(), { once: true });
      setTimeout(() => toast.remove(), 500);
    };
    const timer = setTimeout(remove, duration);
    toast.addEventListener('mouseenter', () => clearTimeout(timer));
    toast.addEventListener('mouseleave', () => setTimeout(remove, 1000));
  };

  // ── Helpers ──────────────────────────────────────────────
  function domainColor(parentId) {
    if (!parentId) return 'var(--border-default)';
    const name = parentId.replace('domain:', '');
    const hue = DOMAIN_HUES[name] ?? 200;
    return `hsl(${hue}, 65%, 55%)`;
  }

  function escHtml(s) {
    return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  // ── Bootstrap ────────────────────────────────────────────
  function restorePrefs() {
    try {
      const theme = localStorage.getItem('sysmap_theme') || 'dark';
      state.theme = theme;
      document.documentElement.setAttribute('data-theme', theme);
    } catch(_) {
      document.documentElement.setAttribute('data-theme', 'dark');
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    restorePrefs();
    initCy();
    loadData();
    initFilters();
    initSearch();
    initToolbar();
    initKeyboard();
    initNodeEvents();
    initTooltips();
    Legend.init();
  });

})();
