'use strict';
/* Renders the small, validated subgraph returned by /api/knowledge-graph/related as layered SVG. */
const GRAPH_LAYERS = {
  part: 0, chapter: 0, legalissue: 0, userright: 0, amendment: 0, legalscenario: 0,
  article: 1,
  clause: 2, subclause: 2,
};

function renderLegalGraph(graph) {
  if (!graph?.nodes?.length || !graph?.edges?.length) {
    return '<div class="graph-empty">No validated graph connections are available for this answer.</div>';
  }
  const escape = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[ch]);
  const typeOf = node => String(node.entity_type || '').toLowerCase();
  const rowsMap = new Map();
  for (const node of graph.nodes.slice(0, 24)) {
    const layer = GRAPH_LAYERS[typeOf(node)] ?? 3;
    if (!rowsMap.has(layer)) rowsMap.set(layer, []);
    rowsMap.get(layer).push(node);
  }
  const rows = [...rowsMap.keys()].sort().map(k => rowsMap.get(k));
  const nodeWidth = 176, nodeHeight = 62, gap = 22, padding = 22, rowGap = 112;
  const widestRow = Math.max(...rows.map(row => row.length), 1);
  const width = Math.max(480, padding * 2 + widestRow * nodeWidth + (widestRow - 1) * gap);
  const height = padding * 2 + rows.length * nodeHeight + (rows.length - 1) * (rowGap - nodeHeight) + 14;
  const positions = new Map();
  rows.forEach((row, r) => {
    const rowWidth = row.length * nodeWidth + (row.length - 1) * gap;
    const offset = (width - rowWidth) / 2;
    row.forEach((node, i) => positions.set(node.id, {x: offset + i * (nodeWidth + gap), y: padding + r * rowGap}));
  });
  const wrapLabel = label => {
    const words = String(label ?? '').split(/\s+/), lines = [];
    let line = '';
    for (const word of words) {
      if (line && `${line} ${word}`.length > 24) { lines.push(line); line = word; } else line = line ? `${line} ${word}` : word;
    }
    if (line) lines.push(line);
    return lines.length > 3 ? [...lines.slice(0, 2), lines[2] + '…'] : lines;
  };
  const edges = graph.edges.map((edge, i) => {
    const from = positions.get(edge.source), to = positions.get(edge.target);
    if (!from || !to) return '';
    const sameRow = from.y === to.y;
    const x1 = from.x + nodeWidth / 2, x2 = to.x + nodeWidth / 2;
    const y1 = sameRow ? from.y : (from.y < to.y ? from.y + nodeHeight : from.y);
    const y2 = sameRow ? to.y : (from.y < to.y ? to.y : to.y + nodeHeight);
    const bend = sameRow ? 18 + (i % 4) * 9 : 0;
    const midY = sameRow ? Math.min(from.y, to.y) - bend : (y1 + y2) / 2;
    const relation = String(edge.relation || '').replaceAll('_', ' ');
    const tip = `${relation} (${edge.status || 'verified'}). ${edge.provenance || ''}`;
    return `<g class="graph-edge"><title>${escape(tip)}</title><path d="M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}" marker-end="url(#graph-arrow)"/><text x="${(x1 + x2) / 2}" y="${midY - 4}">${escape(relation)}</text></g>`;
  }).join('');
  const nodes = graph.nodes.slice(0, 24).map(node => {
    const p = positions.get(node.id);
    if (!p) return '';
    const lines = wrapLabel(node.label);
    const firstY = p.y + nodeHeight / 2 - (lines.length - 1) * 7;
    const text = lines.map((line, i) => `<tspan x="${p.x + nodeWidth / 2}" y="${firstY + i * 14}">${escape(line)}</tspan>`).join('');
    return `<g class="graph-node ${escape(typeOf(node))}"><title>${escape(node.entity_type + ': ' + node.label + ' (' + (node.status || '') + ')')}</title><rect x="${p.x}" y="${p.y}" width="${nodeWidth}" height="${nodeHeight}" rx="12"/><text>${text}</text></g>`;
  }).join('');
  return `<svg class="graph-visual" role="img" aria-label="Legal connections graph" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}"><defs><marker id="graph-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z"/></marker></defs>${edges}${nodes}</svg>
    <div class="graph-legend">Hover a connection to see the source text it was derived from. Only verified or curated relationships are shown.</div>`;
}
