import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const projectRoot = 'C:/Users/Damien/OneDrive/Desktop/AIC-Hackathon';
const scan = JSON.parse(
  readFileSync(join(projectRoot, '.understand-anything/intermediate/scan-result.json'), 'utf8')
);
writeFileSync(
  join(projectRoot, '.understand-anything/tmp/ua-import-map-input.json'),
  JSON.stringify({ projectRoot, files: scan.files })
);
