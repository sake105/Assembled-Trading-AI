/* ============================================================
   minimap.js — cytoscape-navigator Mini-Map
   ============================================================ */
const MiniMap = (() => {
  let _visible = false;
  let _navigator = null;
  const STORAGE_KEY = 'sysmap_minimap';

  function show() {
    const el = document.getElementById('minimap');
    if (!el || !window.cy) return;
    el.classList.remove('hidden');
    if (!_navigator && window.cy.navigator) {
      _navigator = window.cy.navigator({
        container: '#minimap',
        viewLiveFramerate: 0,
        thumbnailEventFramerate: 30,
        thumbnailLiveFramerate: false,
        dblClickDelay: 200,
        removeCustomContainer: false,
        rerenderDelay: 100,
      });
    }
    _visible = true;
    try { localStorage.setItem(STORAGE_KEY, '1'); } catch(_) {}
    const btn = document.getElementById('tb-minimap');
    if (btn) btn.classList.add('icon-btn--active');
  }

  function hide() {
    const el = document.getElementById('minimap');
    if (el) el.classList.add('hidden');
    _visible = false;
    try { localStorage.setItem(STORAGE_KEY, '0'); } catch(_) {}
    const btn = document.getElementById('tb-minimap');
    if (btn) btn.classList.remove('icon-btn--active');
  }

  function toggle() {
    _visible ? hide() : show();
  }

  return {
    init() {
      let stored;
      try { stored = localStorage.getItem(STORAGE_KEY); } catch(_) {}
      if (stored === '1') show();
    },
    show, hide, toggle,
    isVisible() { return _visible; },
  };
})();
