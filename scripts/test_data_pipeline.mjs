import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pathsnap-pipeline-'));
const data = path.join(root, 'Data');
const output = path.join(root, 'artifacts');

function writeTrip(partition, tripId, options = {}) {
  const directory = path.join(data, partition, tripId);
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, `${tripId}.track`), options.track ?? '8.5952 49.8440 1\n8.5956 49.8442 3\n');
  fs.writeFileSync(path.join(directory, `${tripId}.nodes`), options.nodes ?? '8.5950 49.8440\n8.5960 49.8445\n');
  fs.writeFileSync(path.join(directory, `${tripId}.arcs`), options.arcs ?? '0 1\n');
  fs.writeFileSync(path.join(directory, `${tripId}.route`), options.route ?? '0\n');
}

try {
  writeTrip('train', 'train001');
  writeTrip('student_test', 'test001', {
    route: 'this-route-label-must-never-be-read\n',
  });
  writeTrip('evaluator_test', 'hidden001', {
    track: '999 999 evaluator-secret\n',
    route: 'evaluator-secret-label\n',
  });

  const script = path.resolve('scripts/build_data_pipeline.mjs');
  const run = spawnSync(process.execPath, [
    script,
    '--data',
    data,
    '--output',
    output,
    '--expect-train',
    '1',
    '--expect-student-test',
    '1',
  ], { encoding: 'utf8' });
  assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);

  const candidates = fs.readFileSync(path.join(output, 'candidates.jsonl'), 'utf8')
    .trim().split('\n').filter(Boolean).map(JSON.parse);
  assert.equal(candidates.length, 4);
  assert(candidates.every((row) => row.schema_version === 'pathsnap.candidates.v1'));
  assert(candidates.some((row) => row.partition === 'train' && row.candidate_arc_id === 0 && row.label === 1));
  assert(candidates.filter((row) => row.partition === 'student_test').every((row) => row.label === null));
  assert(candidates.every((row) => row.trip_id !== 'hidden001'));

  const report = JSON.parse(fs.readFileSync(path.join(output, 'data_quality_report.json'), 'utf8'));
  assert.deepEqual(report.partitions, { train: 1, student_test: 1 });
  assert.equal(report.trips.some((trip) => trip.trip_id === 'hidden001'), false);

  for (const entry of fs.readdirSync(output)) {
    const file = path.join(output, entry);
    if (!fs.statSync(file).isFile()) continue;
    const content = fs.readFileSync(file, 'utf8');
    assert.doesNotMatch(content, /evaluator[_ -]?test|evaluator-secret/i);
  }

  console.log('Data pipeline privacy and output test passed.');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
