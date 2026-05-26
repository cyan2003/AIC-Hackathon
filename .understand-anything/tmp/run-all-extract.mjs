import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const projectRoot = 'C:/Users/Damien/OneDrive/Desktop/AIC-Hackathon';
const skillDir =
  'C:/Users/Damien/.cursor/plugins/cache/understand-anything/understand-anything/470cc01dc5f9236a93eb704afdd479cd5db79710/skills/understand';
const batches = JSON.parse(
  readFileSync(join(projectRoot, '.understand-anything/intermediate/batches.json'), 'utf8')
);

for (const batch of batches.batches) {
  const idx = batch.batchIndex;
  const inputPath = join(projectRoot, `.understand-anything/tmp/ua-file-analyzer-input-${idx}.json`);
  const outputPath = join(projectRoot, `.understand-anything/tmp/ua-file-extract-results-${idx}.json`);
  writeFileSync(
    inputPath,
    JSON.stringify({
      projectRoot,
      batchFiles: batch.files,
      batchImportData: batch.batchImportData,
    })
  );
  const r = spawnSync(
    'node',
    [join(skillDir, 'extract-structure.mjs'), inputPath, outputPath],
    { encoding: 'utf8', cwd: projectRoot }
  );
  if (r.status !== 0) {
    console.error(`Batch ${idx} failed:`, r.stderr);
    process.exit(1);
  }
  console.log(`Batch ${idx} extracted: ${batch.files.length} files`);
}
