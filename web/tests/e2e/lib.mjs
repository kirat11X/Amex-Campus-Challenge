// Shared helpers for the browser smoke tests (playwright-core + any Chromium).
//   CHROME_PATH=/path/to/chrome  BASE_URL=http://localhost:5190  node tests/e2e/<script>.mjs
import { chromium } from 'playwright-core';

export const BASE = process.env.BASE_URL ?? 'http://localhost:5190';
export const SHOTS = process.env.SHOTS_DIR ?? '.';

export async function launch(extraArgs = []) {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || undefined,
    headless: true,
    args: ['--no-sandbox', ...extraArgs],
  });
  const ctx = await browser.newContext({ viewport: { width: 1360, height: 900 } });
  const page = await ctx.newPage();
  const logs = [];
  page.on('console', (m) => ['error', 'warning'].includes(m.type()) && logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  return { browser, ctx, page, logs };
}

let failed = 0;
export function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`);
  if (!ok) failed++;
}
export const finish = () => {
  console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
  process.exitCode = failed ? 1 : 0;
};
