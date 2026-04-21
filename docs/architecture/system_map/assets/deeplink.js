/* ============================================================
   deeplink.js — URL-Hash-Routing
   Erwartet window.cy (Cytoscape-Instanz) und window.showToast
   ============================================================ */
const DeepLink = (() => {
  function apply(hash) {
    if (!hash || !window.cy) return;
    const id = hash.replace(/^#/, '');
    if (!id) return;
    const node = window.cy.$('#' + CSS.escape(id));
    if (!node.length) {
      if (window.showToast) window.showToast('error', `Knoten nicht gefunden: ${id}`);
      return;
    }
    window.cy.animate(
      { fit: { eles: node, padding: 80 } },
      { duration: 400, easing: 'ease-out-cubic' }
    );
    if (id.startsWith('module:') && window.openDetailPanel) {
      window.openDetailPanel(node);
    }
    if (window.showToast) window.showToast('info', `→ ${node.data('label')}`);
  }

  function setHash(nodeId) {
    try {
      history.replaceState(null, '', '#' + nodeId);
    } catch (_) { /* file:// blocks history API in some browsers */ }
  }

  return {
    init() {
      window.addEventListener('hashchange', () => apply(location.hash));
      if (location.hash) {
        // Verzögern bis cy initialisiert
        const wait = setInterval(() => {
          if (window.cy) { clearInterval(wait); apply(location.hash); }
        }, 100);
        setTimeout(() => clearInterval(wait), 5000);
      }
    },
    setHash,
    apply,
  };
})();
