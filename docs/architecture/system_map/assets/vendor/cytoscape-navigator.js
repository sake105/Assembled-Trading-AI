/* cytoscape-navigator stub — Mini-Map not available
   Download manually from: https://github.com/cytoscape/cytoscape.js-navigator
   or via: npm pack cytoscape.js-navigator
   Place the built UMD file here as cytoscape-navigator.js
   The app degrades gracefully without this file. */
if (typeof cytoscape !== 'undefined' && !cytoscape.prototype.navigator) {
  // Minimal no-op stub so minimap.js doesn't throw
  cytoscape('core', 'navigator', function(opts) {
    console.warn('[SystemMap] cytoscape-navigator not loaded — Mini-Map disabled');
    return null;
  });
}
