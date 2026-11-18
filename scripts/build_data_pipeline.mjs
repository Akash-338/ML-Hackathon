#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const EARTH_RADIUS_M = 6_371_008.8;
const CELL_SIZE_M = 250;
const MAX_SEARCH_M = 3_000;
const MAX_CANDIDATES = 8;

function parseArgs(argv) {
  const args = { data: 'Data/Data', output: 'artifacts/data-pipeline', help: false };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--help' || argv[i] === '-h') args.help = true;
    else if (argv[i] === '--data' && argv[i + 1]) args.data = argv[++i];
    else if (argv[i] === '--output' && argv[i + 1]) args.output = argv[++i];
    else throw new Error(`Unknown or incomplete option: ${argv[i]}`);
  }
  return args;
}

function usage() {
  return 'Usage: node scripts/build_data_pipeline.mjs [--data Data/Data] [--output artifacts/data-pipeline]';
}

function parseRows(file, columns) {
  if (!fs.existsSync(file)) return { rows: [], missing: true, malformed: 0, physical_lines: 0 };
  const rows = [];
  let malformed = 0;
  for (const [lineIndex, line] of fs.readFileSync(file, 'utf8').split(/\r?\n/).entries()) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const cells = trimmed.split(/[\s,]+/);
    if (cells.length < columns || cells.slice(0, columns).some((cell) => cell === '')) {
      malformed += 1;
      continue;
    }
    const values = cells.slice(0, columns).map(Number);
    if (values.some((value) => !Number.isFinite(value))) {
      malformed += 1;
      continue;
    }
    rows.push({ values, line: lineIndex + 1 });
  }
  return { rows, missing: false, malformed, physical_lines: fs.readFileSync(file, 'utf8').split(/\r?\n/).filter((line) => line.trim() && !line.trim().startsWith('#')).length };
}

function haversine(lon1, lat1, lon2, lat2) {
  const radians = Math.PI / 180;
  const dLat = (lat2 - lat1) * radians;
  const dLon = (lon2 - lon1) * radians;
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(lat1 * radians) * Math.cos(lat2 * radians) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.sqrt(Math.min(1, a)));
}

function bearing(a, b) {
  const radians = Math.PI / 180;
  const p1 = a.lat * radians;
  const p2 = b.lat * radians;
  const dl = (b.lon - a.lon) * radians;
  return (Math.atan2(Math.sin(dl) * Math.cos(p2), Math.cos(p1) * Math.sin(p2)
    - Math.sin(p1) * Math.cos(p2) * Math.cos(dl)) * 180 / Math.PI + 360) % 360;
}

function pointSegment(point, segment) {
  const dx = segment.b.x - segment.a.x;
  const dy = segment.b.y - segment.a.y;
  const lengthSquared = dx * dx + dy * dy;
  const t = lengthSquared === 0 ? 0 : Math.max(0, Math.min(1,
    ((point.x - segment.a.x) * dx + (point.y - segment.a.y) * dy) / lengthSquared));
  const x = segment.a.x + t * dx;
  const y = segment.a.y + t * dy;
  return { distance: Math.hypot(point.x - x, point.y - y), t };
}

function project(lon, lat, cosLat) {
  const radians = Math.PI / 180;
  return { x: lon * radians * EARTH_RADIUS_M * cosLat, y: lat * radians * EARTH_RADIUS_M };
}

function bucketKey(x, y) {
  return `${x},${y}`;
}

function makeSpatialIndex(arcs) {
  const buckets = new Map();
  for (const arc of arcs) {
    const minX = Math.floor(Math.min(arc.a.x, arc.b.x) / CELL_SIZE_M);
    const maxX = Math.floor(Math.max(arc.a.x, arc.b.x) / CELL_SIZE_M);
    const minY = Math.floor(Math.min(arc.a.y, arc.b.y) / CELL_SIZE_M);
    const maxY = Math.floor(Math.max(arc.a.y, arc.b.y) / CELL_SIZE_M);
    for (let x = minX; x <= maxX; x += 1) {
      for (let y = minY; y <= maxY; y += 1) {
        const key = bucketKey(x, y);
        if (!buckets.has(key)) buckets.set(key, []);
        buckets.get(key).push(arc);
      }
    }
  }
  return buckets;
}

function nearestArcs(point, index, count = MAX_CANDIDATES, maxDistance = MAX_SEARCH_M) {
  const cellX = Math.floor(point.x / CELL_SIZE_M);
  const cellY = Math.floor(point.y / CELL_SIZE_M);
  const cells = Math.ceil(maxDistance / CELL_SIZE_M);
  const seen = new Set();
  const matches = [];
  for (let ring = 0; ring <= cells; ring += 1) {
    for (let dx = -ring; dx <= ring; dx += 1) {
      for (let dy = -ring; dy <= ring; dy += 1) {
        if (ring > 0 && Math.abs(dx) !== ring && Math.abs(dy) !== ring) continue;
        const nearby = index.get(bucketKey(cellX + dx, cellY + dy)) ?? [];
        for (const arc of nearby) {
          if (seen.has(arc.id)) continue;
          seen.add(arc.id);
          const hit = pointSegment(point, arc);
          if (hit.distance <= maxDistance) matches.push({ arc, ...hit });
        }
      }
    }
    if (matches.length >= count) {
      matches.sort((a, b) => a.distance - b.distance || a.arc.id - b.arc.id);
      const boundary = [
        point.x - (cellX - ring) * CELL_SIZE_M,
        (cellX + ring + 1) * CELL_SIZE_M - point.x,
        point.y - (cellY - ring) * CELL_SIZE_M,
        (cellY + ring + 1) * CELL_SIZE_M - point.y,
      ];
      if (matches[count - 1].distance <= Math.min(...boundary)) break;
    }
  }
  return matches.sort((a, b) => a.distance - b.distance || a.arc.id - b.arc.id).slice(0, count);
}

function minDistanceForRoute(point, routeIds, arcById) {
  let best = null;
  for (const id of routeIds) {
    const arc = arcById.get(id);
    if (!arc) continue;
    const hit = pointSegment(point, arc);
    if (!best || hit.distance < best.distance || (hit.distance === best.distance && id < best.arc.id)) {
      best = { arc, ...hit };
    }
  }
  return best;
}

function summarizeTime(track) {
  const gaps = [];
  let nonIncreasing = 0;
  for (let i = 1; i < track.length; i += 1) {
    const gap = track[i].timestamp - track[i - 1].timestamp;
    if (gap <= 0) nonIncreasing += 1;
    else gaps.push(gap);
  }
  gaps.sort((a, b) => a - b);
  const median = gaps.length ? gaps[Math.floor(gaps.length / 2)] : null;
  return { positive_gap_count: gaps.length, non_increasing_gap_count: nonIncreasing, median_positive_gap: median };
}

function parseTrip(partition, directoryPath) {
  const tripId = path.basename(directoryPath);
  const file = (suffix) => path.join(directoryPath, `${tripId}.${suffix}`);
  const trackParsed = parseRows(file('track'), 3);
  const nodesParsed = parseRows(file('nodes'), 2);
  const arcsParsed = parseRows(file('arcs'), 2);
  const routePresent = fs.existsSync(file('route'));
  const track = trackParsed.rows.map(({ values, line }) => ({ lon: values[0], lat: values[1], timestamp: values[2], sourceIndex: line - 1 }));
  const nodes = [];
  for (const { values, line } of nodesParsed.rows) nodes[line - 1] = { lon: values[0], lat: values[1] };
  const arcs = [];
  let invalidArcReferences = 0;
  for (const row of arcsParsed.rows) {
    const id = row.line - 1;
    const [from, to] = row.values;
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to < 0 || !nodes[from] || !nodes[to]) {
      invalidArcReferences += 1;
      continue;
    }
    arcs[id] = { id, from, to, sourceLine: row.line, start: nodes[from], end: nodes[to] };
  }
  // Student-test route labels are deliberately never read or parsed.
  const routeParsed = partition === 'train' ? parseRows(file('route'), 1) : null;
  const route = routeParsed?.rows.map(({ values }) => values[0]) ?? null;
  const validArcIds = new Set(arcs.filter(Boolean).map((arc) => arc.id));
  const invalidRouteReferences = route ? route.filter((id) => !Number.isInteger(id) || !validArcIds.has(id)).length : null;
  const lats = nodes.map((node) => node.lat).filter(Number.isFinite);
  const cosLat = Math.cos((lats.length ? lats.reduce((sum, value) => sum + value, 0) / lats.length : 0) * Math.PI / 180);
  const projectedNodes = nodes.map((node) => node ? ({ ...node, ...project(node.lon, node.lat, cosLat) }) : undefined);
  const projectedArcs = arcs.filter(Boolean).map((arc) => ({
    id: arc.id,
    from: arc.from,
    to: arc.to,
    a: { x: projectedNodes[arc.from].x, y: projectedNodes[arc.from].y },
    b: { x: projectedNodes[arc.to].x, y: projectedNodes[arc.to].y },
    length: haversine(arc.start.lon, arc.start.lat, arc.end.lon, arc.end.lat),
    heading: bearing(arc.start, arc.end),
  }));
  return {
    id: tripId,
    partition,
    file,
    track,
    nodes,
    arcs,
    sourceRowCounts: {
      track: trackParsed.physical_lines,
      nodes: nodesParsed.physical_lines,
      arcs: arcsParsed.physical_lines,
      route: routeParsed?.physical_lines ?? null,
    },
    projectedNodes,
    projectedArcs,
    route,
    routePresent,
    routeMalformed: routeParsed?.malformed ?? null,
    cosLat,
    index: makeSpatialIndex(projectedArcs),
    malformed: { track: trackParsed.malformed, nodes: nodesParsed.malformed, arcs: arcsParsed.malformed, route: routeParsed?.malformed ?? null },
    missing: { track: trackParsed.missing, nodes: nodesParsed.missing, arcs: arcsParsed.missing, route: partition === 'train' ? (routeParsed?.missing ?? true) : null },
    invalidArcReferences,
    invalidRouteReferences,
    invalidRouteIds: route ? route.filter((id) => !Number.isInteger(id) || !validArcIds.has(id)) : null,
  };
}

function arcFeature(point, previous, candidate) {
  const gpsBearing = previous ? bearing(previous, point) : null;
  let bearingDifference = null;
  if (gpsBearing !== null) {
    const raw = Math.abs(gpsBearing - candidate.arc.heading) % 360;
    bearingDifference = Math.min(raw, 360 - raw);
  }
  const dx = candidate.arc.b.x - candidate.arc.a.x;
  const dy = candidate.arc.b.y - candidate.arc.a.y;
  const metersAlong = Math.hypot(dx, dy) * candidate.t;
  const arcLength = candidate.arc.length;
  const timeDelta = previous ? point.timestamp - previous.timestamp : null;
  const stepDistance = previous ? haversine(previous.lon, previous.lat, point.lon, point.lat) : null;
  return {
    gps_to_arc_distance_m: Number(candidate.distance.toFixed(3)),
    gps_heading_difference_deg: bearingDifference === null ? null : Number(bearingDifference.toFixed(3)),
    arc_length_m: Number(arcLength.toFixed(3)),
    distance_from_arc_start_m: Number(metersAlong.toFixed(3)),
    fraction_along_arc: Number(candidate.t.toFixed(6)),
    time_delta_s: timeDelta,
    gps_speed_mps: timeDelta > 0 && stepDistance !== null ? Number((stepDistance / timeDelta).toFixed(3)) : null,
  };
}

function makeCandidateRows(trip) {
  const rows = [];
  const arcById = new Map(trip.projectedArcs.map((arc) => [arc.id, arc]));
  for (let observationIndex = 0; observationIndex < trip.track.length; observationIndex += 1) {
    const point = trip.track[observationIndex];
    const previous = observationIndex > 0 ? trip.track[observationIndex - 1] : null;
    const projected = project(point.lon, point.lat, trip.cosLat);
    let candidates = nearestArcs(projected, trip.index);
    let labelArcId = null;
    if (trip.route) {
      const routeMatch = minDistanceForRoute(projected, trip.route, arcById);
      labelArcId = routeMatch?.arc.id ?? null;
      if (labelArcId !== null && !candidates.some((candidate) => candidate.arc.id === labelArcId)) {
        candidates = [...candidates.slice(0, Math.max(0, MAX_CANDIDATES - 1)), routeMatch];
        candidates.sort((a, b) => a.distance - b.distance || a.arc.id - b.arc.id);
      }
    }
    candidates.forEach((candidate, rank) => rows.push({
      schema_version: 'pathsnap.candidates.v1',
      trip_id: trip.id,
      partition: trip.partition,
      observation_index: point.sourceIndex,
      timestamp: point.timestamp,
      observation: { lon: point.lon, lat: point.lat },
      candidate_arc_id: candidate.arc.id,
      candidate_rank: rank,
      features: arcFeature(point, previous, candidate),
      label: labelArcId === null ? null : Number(candidate.arc.id === labelArcId),
      label_method: labelArcId === null ? null : 'nearest_route_arc_by_geodesic_segment_distance',
    }));
  }
  return rows;
}

function bounds(points) {
  const result = { min_lon: Infinity, max_lon: -Infinity, min_lat: Infinity, max_lat: -Infinity };
  for (const point of points) {
    if (!point) continue;
    result.min_lon = Math.min(result.min_lon, point.lon);
    result.max_lon = Math.max(result.max_lon, point.lon);
    result.min_lat = Math.min(result.min_lat, point.lat);
    result.max_lat = Math.max(result.max_lat, point.lat);
  }
  return Number.isFinite(result.min_lon) ? result : null;
}

function tripSummary(trip) {
  const summary = {
    trip_id: trip.id,
    partition: trip.partition,
    track_rows: trip.track.length,
    track_source_rows: trip.sourceRowCounts.track,
    nodes_rows: trip.nodes.length,
    nodes_source_rows: trip.sourceRowCounts.nodes,
    arcs_rows: trip.projectedArcs.length,
    arcs_source_rows: trip.sourceRowCounts.arcs,
    route_rows: trip.partition === 'train' ? (trip.route?.length ?? 0) : null,
    route_source_rows: trip.partition === 'train' ? trip.sourceRowCounts.route : null,
    route_file_present: trip.routePresent,
    route_data_status: trip.partition === 'train' ? 'read_for_training_labels' : 'withheld_from_pipeline',
    malformed_track_rows: trip.malformed.track,
    malformed_nodes_rows: trip.malformed.nodes,
    malformed_arcs_rows: trip.malformed.arcs,
    malformed_route_rows: trip.partition === 'train' ? trip.malformed.route : null,
    invalid_arc_node_references: trip.invalidArcReferences,
    invalid_route_arc_references: trip.invalidRouteReferences,
    coordinate_bounds: bounds(trip.track),
    timestamp_gaps: summarizeTime(trip.track),
  };
  return summary;
}

function renderMap(trip, outputFile) {
  if (!trip.track.length) return;
  const extent = bounds([...trip.track, ...trip.nodes]);
  const { min_lon: minLon, max_lon: maxLon, min_lat: minLat, max_lat: maxLat } = extent;
  const width = 1000;
  const height = 720;
  const padding = 30;
  const sx = (width - 2 * padding) / (maxLon - minLon || 1);
  const sy = (height - 2 * padding) / (maxLat - minLat || 1);
  const scale = Math.min(sx, sy);
  const xy = (p) => [padding + (p.lon - minLon) * scale, height - padding - (p.lat - minLat) * scale];
  const trackPoints = trip.track.map((p) => xy(p).map((n) => n.toFixed(2)).join(',')).join(' ');
  const routeSegments = [];
  for (const id of trip.route ?? []) {
    const arc = trip.arcs[id];
    if (!arc) continue;
    const [x1, y1] = xy(arc.start);
    const [x2, y2] = xy(arc.end);
    routeSegments.push(`<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" />`);
  }
  const svg = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="#fff"/><text x="24" y="24" font-family="sans-serif" font-size="16">Training trip ${trip.id}: GPS track and reference route (actual dataset)</text>
<g stroke="#9aa5b1" stroke-width="1" opacity="0.65">${routeSegments.join('')}</g>
<polyline points="${trackPoints}" fill="none" stroke="#d62728" stroke-width="2.5"/><text x="24" y="${height - 36}" font-family="sans-serif" font-size="14" fill="#d62728">GPS track</text>
<text x="120" y="${height - 36}" font-family="sans-serif" font-size="14" fill="#65717d">Reference route arcs shown in gray</text>
</svg>`;
  fs.writeFileSync(outputFile, svg);
}

function writeJsonLine(stream, value) {
  stream.write(`${JSON.stringify(value)}\n`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const dataRoot = path.resolve(args.data);
  const outputRoot = path.resolve(args.output);
  if (!fs.existsSync(dataRoot)) throw new Error(`Dataset directory not found: ${dataRoot}`);
  const partitionNames = ['train', 'student_test'];
  const summaries = [];
  const reportTrips = [];
  fs.mkdirSync(outputRoot, { recursive: true });
  const candidatePath = path.join(outputRoot, 'candidates.jsonl');
  const candidateStream = fs.createWriteStream(candidatePath, { encoding: 'utf8' });
  let candidateRows = 0;
  let labeledRows = 0;
  let unlabeledRows = 0;
  let firstTrain = null;

  for (const partition of partitionNames) {
    const dir = path.join(dataRoot, partition);
    if (!fs.existsSync(dir)) throw new Error(`Missing dataset partition directory: ${dir}`);
    const trips = fs.readdirSync(dir, { withFileTypes: true }).filter((entry) => entry.isDirectory()).sort((a, b) => a.name.localeCompare(b.name, 'en'));
    for (const entry of trips) {
      const trip = parseTrip(partition, path.join(dir, entry.name));
      const summary = tripSummary(trip);
      summaries.push(summary);
      reportTrips.push(trip);
      if (partition === 'train' && !firstTrain) firstTrain = trip;
      for (const row of makeCandidateRows(trip)) {
        writeJsonLine(candidateStream, row);
        candidateRows += 1;
        if (row.label === null) unlabeledRows += 1;
        else labeledRows += 1;
      }
      console.log(`${partition}/${trip.id}: ${trip.track.length} observations, ${trip.projectedArcs.length} arcs`);
    }
  }
  candidateStream.end();
  const report = {
    schema_version: 'pathsnap.data-report.v1',
    source: path.relative(process.cwd(), dataRoot),
    generated_at: new Date().toISOString(),
    partitions: Object.fromEntries(partitionNames.map((name) => [name, summaries.filter((s) => s.partition === name).length])),
    trip_count: summaries.length,
    candidate_rows: candidateRows,
    labeled_candidate_rows: labeledRows,
    unlabeled_candidate_rows: unlabeledRows,
    student_test_label_policy: 'student_test route files are not read or emitted; all test candidate labels are null',
    trips: summaries,
  };
  fs.writeFileSync(path.join(outputRoot, 'data_quality_report.json'), `${JSON.stringify(report, null, 2)}\n`);
  const csvHeader = Object.keys(summaries[0] ?? { trip_id: '' }).filter((key) => key !== 'coordinate_bounds' && key !== 'timestamp_gaps');
  const csv = [csvHeader.join(','), ...summaries.map((summary) => csvHeader.map((key) => JSON.stringify(summary[key] ?? '')).join(','))].join('\n');
  fs.writeFileSync(path.join(outputRoot, 'trip_summary.csv'), `${csv}\n`);
  if (firstTrain) renderMap(firstTrain, path.join(outputRoot, `training_trip_${firstTrain.id}_map.svg`));
  fs.writeFileSync(path.join(outputRoot, 'README.txt'), `PathSnap data pipeline outputs\n\n- candidates.jsonl: one row per observation/candidate arc; train labels use the nearest reference-route arc by geodesic point-to-segment distance.\n- data_quality_report.json: partition and per-trip counts, format/ID validation, coordinate bounds, and timestamp-gap summary.\n- trip_summary.csv: compact per-trip metrics.\n- training_trip_${firstTrain?.id ?? 'none'}_map.svg: GPS track and reference route from a training trip.\n\nCandidate features are listed in each row under features. Arc IDs are line-index IDs local to their trip. Student-test route files are never read.\n`);
  console.log(`Wrote ${candidateRows} candidate rows and EDA outputs to ${outputRoot}`);
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  console.error(usage());
  process.exitCode = 1;
}
