/* ============================================================
   app.js — System Map Application
   ============================================================ */
(function () {
  'use strict';

  // ── Constants ───────────────────────────────────────────
  const DATA_URL = 'data/system_map.json';
  const ZOOM_GALAXY = 0.4;
  const ZOOM_SYSTEM = 1.2;
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

  // ── Cytoscape Stylesheet ─────────────────────────────────
  const CY_STYLE = [
    { selector: 'node', style: {
      'background-color': '#1f2937', 'border-width': 1.5, 'border-color': '#374151',
      'label': 'data(label)', 'font-family': 'JetBrains Mono, ui-monospace, monospace',
      'font-size': 11, 'color': '#9ca3af', 'text-valign': 'center', 'text-halign': 'center',
      'text-max-width': 120, 'text-overflow-wrap': 'whitespace',
      'transition-property': 'opacity, border-color', 'transition-duration': 200,
    }},
    { selector: 'node[type="domain"]', style: {
      'shape': 'round-rectangle', 'background-color': '#1e2d4a', 'border-color': '#3b82f6',
      'border-width': 2, 'font-size': 13, 'font-weight': 600,
      'font-family': 'Inter, system-ui, sans-serif', 'color': '#f9fafb',
      'text-valign': 'top', 'padding': 20, 'min-width': 180, 'min-height': 100,
    }},
    { selector: 'node[type="module"]', style: {
      'shape': 'ellipse',
      'width': 'mapData(loc, 0, 800, 28, 60)', 'height': 'mapData(loc, 0, 800, 28, 60)',
    }},
    { selector: 'node[type="external_api"]', style: {
      'shape': 'diamond', 'width': 40, 'height': 40,
      'background-color': '#1a1a2e', 'border-color': '#3b82f6', 'border-width': 2,
    }},
    { selector: 'node[type="script"]', style: {
      'shape': 'round-rectangle', 'width': 80, 'height': 30,
      'background-color': '#1e3a2f', 'border-color': '#22c55e',
    }},
    { selector: 'node[type="workflow"]', style: {
      'shape': 'hexagon', 'width': 45, 'height': 45,
      'background-color': '#2d1a3e', 'border-color': '#a855f7',
    }},
    { selector: 'node[type="entry_point"]', style: {
      'shape': 'round-rectangle', 'width': 80, 'height': 30,
      'background-color': '#1e3a2f', 'border-color': '#22c55e', 'border-width': 3,
    }},
    { selector: 'node[status="green"]',  style: { 'border-color': '#22c55e' } },
    { selector: 'node[status="yellow"]', style: { 'border-color': '#eab308' } },
    { selector: 'node[status="orange"]', style: { 'border-color': '#f97316' } },
    { selector: 'node[status="red"]',    style: { 'border-color': '#ef4444', 'border-width': 2.5 } },
    { selector: 'node[status="gray"]',   style: { 'border-color': '#6b7280', 'border-style': 'dashed' } },
    { selector: 'node[?orphan]',         style: { 'border-style': 'double', 'border-width': 4, 'border-color': '#f59e0b' } },
    { selector: 'node[duplicate_group]', style: { 'border-color': '#a855f7', 'border-style': 'dashed', 'border-width': 2 } },
    { selector: 'node[?in_cycle]',       style: { 'border-color': '#ec4899', 'border-width': 2.5 } },
    { selector: 'node:selected',         style: { 'overlay-color': '#3b82f6', 'overlay-opacity': 0.15, 'overlay-padding': 6 } },
    { selector: '.faded',     style: { 'opacity': 0.08 } },
    { selector: '.highlighted', style: { 'opacity': 1, 'z-index': 10 } },
    { selector: '.zoom-galaxy node:not([type="domain"])', style: { 'label': '' } },

    { selector: 'edge', style: {
      'curve-style': 'bezier', 'target-arrow-shape': 'triangle',
      'target-arrow-color': '#64748b', 'line-color': '#64748b',
      'width': 1.5, 'opacity': 0.6,
      'transition-property': 'opacity, line-color', 'transition-duration': 200,
    }},
    { selector: 'edge[kind="import"]',    style: { 'line-color': '#64748b', 'line-style': 'solid', 'width': 1.5 } },
    { selector: 'edge[kind="api_call"]',  style: { 'line-color': '#3b82f6', 'line-style': 'dashed', 'line-dash-pattern': [6, 3], 'width': 2, 'target-arrow-color': '#3b82f6' } },
    { selector: 'edge[kind="data_flow"]', style: { 'line-color': '#14b8a6', 'line-style': 'solid', 'width': 3, 'target-arrow-color': '#14b8a6' } },
    { selector: 'edge[kind="trigger"]',   style: { 'line-color': '#a855f7', 'line-style': 'dotted', 'width': 2, 'target-arrow-color': '#a855f7' } },
    { selector: 'edge[?circular]',        style: { 'line-color': '#ec4899', 'line-style': 'dashed', 'line-dash-pattern': [4, 4], 'width': 2 } },
    { selector: 'edge.path-highlight',    style: { 'line-color': '#f59e0b', 'width': 3, 'opacity': 1 } },
    { selector: 'edge:selected',          style: { 'line-color': '#3b82f6', 'width': 2.5 } },
  ];

  const FCOSE_CONFIG = {
    name: 'fcose', quality: 'default', animate: true,
    animationDuration: 500, fit: true, padding: 40,
    nodeDimensionsIncludeLabels: true, uniformNodeDimensions: false,
    packComponents: true,
    nodeRepulsion: () => 6500,
    idealEdgeLength: () => 80,
    edgeElasticity: () => 0.45,
    nestingFactor: 0.1, numIter: 2500,
    tile: true, tilingPaddingVertical: 20, tilingPaddingHorizontal: 20,
    gravity: 0.25, gravityRange: 3.8, gravityCompound: 1.0, gravityRangeCompound: 1.5,
  };

  // ── Cytoscape Init ───────────────────────────────────────
  function initCy() {
    if (typeof cytoscape === 'undefined') return;
    if (typeof cytoscapeFcose !== 'undefined') cytoscape.use(cytoscapeFcose);
    cy = cytoscape({
      container: $('cy'),
      style: CY_STYLE,
      elements: [],
      wheelSensitivity: 0.3,
    });
    window.cy = cy;

    if (typeof cytoscapeExpandCollapse !== 'undefined') {
      cy.expandCollapse({ layoutBy: FCOSE_CONFIG, animate: true, animationDuration: 250 });
    }
    if (typeof cytoscapePopper !== 'undefined') {
      cytoscape.use(cytoscapePopper);
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
    const elements = [
      ...data.nodes.map(n => ({ data: n })),
      ...data.edges.map(e => ({ data: e })),
    ];
    cy.add(elements);

    const layout = cy.layout({ ...FCOSE_CONFIG, quality: 'draft', numIter: 800, animationDuration: 200 });
    layout.on('layoutstop', () => {
      hideSkeleton();
      initFuse(data.nodes);
      initZoom();
      DeepLink.init();
      MiniMap.init();
    });
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

    cy.on('tap', 'core', e => {
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
      const theme = localStorage.getItem('sysmap_theme');
      if (theme) { state.theme = theme; document.documentElement.setAttribute('data-theme', theme); }
    } catch(_) {}
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
