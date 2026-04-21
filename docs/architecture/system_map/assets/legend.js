/* ============================================================
   legend.js — Status & Heat-Map Legend
   ============================================================ */
const Legend = (() => {
  let container;
  let _mode = 'status';

  const STATUS_ROWS = [
    { color: '#22c55e', label: 'green — verdrahtet + getestet' },
    { color: '#eab308', label: 'yellow — verdrahtet, wenige Tests' },
    { color: '#f97316', label: 'orange — verdrahtet, 0 Tests' },
    { color: '#ef4444', label: 'red — blockiert / NotImplementedError' },
    { color: '#6b7280', label: 'gray — unbekannt' },
  ];
  const SPECIAL_ROWS = [
    { color: '#a855f7', label: 'Duplikat-Gruppe',  dashed: true },
    { color: '#f59e0b', label: 'Orphan' },
    { color: '#ec4899', label: 'Zirkulärer Import' },
  ];
  const EDGE_ROWS = [
    { color: '#64748b', label: 'import',    style: 'solid' },
    { color: '#3b82f6', label: 'api_call',  style: 'dashed' },
    { color: '#14b8a6', label: 'data_flow', style: 'solid' },
    { color: '#a855f7', label: 'trigger',   style: 'dotted' },
  ];
  const HEAT_ROWS = [
    { color: '#1e3a5f', label: '< 50 LOC' },
    { color: '#1d4ed8', label: '< 150 LOC' },
    { color: '#7c3aed', label: '< 300 LOC' },
    { color: '#be185d', label: '< 500 LOC' },
    { color: '#ef4444', label: '> 500 LOC' },
  ];

  function dot(color, dashed) {
    const s = `background:${color};width:10px;height:10px;border-radius:2px;flex-shrink:0;${dashed ? `outline:2px dashed ${color};outline-offset:1px;background:transparent;` : ''}`;
    return `<span style="${s}"></span>`;
  }
  function edgeLine(color, style) {
    const border = style === 'dashed' ? `border-top:2px dashed ${color}` :
                   style === 'dotted' ? `border-top:2px dotted ${color}` :
                                        `border-top:2px solid ${color}`;
    return `<span style="width:24px;display:inline-block;${border}"></span>`;
  }

  function renderStatus() {
    const rows = STATUS_ROWS.map(r =>
      `<div class="legend-row">${dot(r.color)} <span>${r.label}</span></div>`
    ).join('');
    const special = SPECIAL_ROWS.map(r =>
      `<div class="legend-row">${dot(r.color, r.dashed)} <span>${r.label}</span></div>`
    ).join('');
    const edges = EDGE_ROWS.map(r =>
      `<div class="legend-row">${edgeLine(r.color, r.style)} <span>${r.label}</span></div>`
    ).join('');
    return `
      <div class="legend-title">Status</div>${rows}
      <div style="margin-top:8px"></div>
      ${special}
      <div style="margin-top:8px"></div>
      <div class="legend-title" style="margin-top:4px">Edges</div>${edges}
    `;
  }

  function renderHeatmap() {
    const rows = HEAT_ROWS.map(r =>
      `<div class="legend-row">${dot(r.color)} <span>${r.label}</span></div>`
    ).join('');
    return `<div class="legend-title">LOC-Dichte</div>${rows}`;
  }

  function render() {
    if (!container) return;
    container.innerHTML = _mode === 'heatmap' ? renderHeatmap() : renderStatus();
  }

  return {
    init() {
      container = document.getElementById('legend-container');
      render();
    },
    setMode(mode) {
      _mode = mode;
      render();
    },
    getMode() { return _mode; },
  };
})();
