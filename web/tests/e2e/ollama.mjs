// End-to-end through the real UI against a local Ollama (the "My Ollama" engine):
// connect → customer lookup + chart → distribution → segment compare → retrieval → ad-hoc query
// (downloads the full customer table in the browser) → upload a file and ask about it.
//   OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M node tests/e2e/ollama.mjs
import { readFileSync } from 'node:fs';
import { writeFileSync } from 'node:fs';
import { BASE, SHOTS, check, finish, launch } from './lib.mjs';

const golden = JSON.parse(readFileSync(new URL('../golden/tools.json', import.meta.url)));
const goldenCount = golden.find((c) => c.tool === 'query_segment' && c.args.filters === 'f3 >= 1 and profit > 0' && c.args.agg === 'count').result.count;

const { browser, page, logs } = await launch();
await page.goto(BASE);
await page.getByRole('heading', { name: 'ProfitLens', level: 1 }).waitFor({ timeout: 30000 });
await page.getByRole('tab', { name: 'My Ollama' }).click();
if (process.env.OLLAMA_MODEL) await page.getByLabel('Model').fill(process.env.OLLAMA_MODEL);
await page.getByRole('button', { name: 'Connect' }).click();
await page.locator('.badge-ok', { hasText: '(your Ollama)' }).waitFor({ timeout: 20000 });
check('connected to Ollama from the browser (CORS ok)', true);

async function ask(text, timeout = 360000) {
  const before = await page.locator('.msg.assistant').count();
  await page.getByLabel('Message').fill(text);
  await page.getByRole('button', { name: 'Send' }).click();
  const msg = page.locator('.msg.assistant').nth(before);
  await msg.waitFor({ timeout: 10000 });
  await msg.locator('.status').waitFor({ state: 'detached', timeout });
  return {
    text: (await msg.locator('.md, .bubble p').first().innerText().catch(() => '')).trim(),
    charts: await msg.locator('figure.chart').count(),
    tools: (await msg.locator('.tools').innerText().catch(() => '')).replace('🛠 ', ''),
    error: await msg.evaluate((n) => n.classList.contains('error')),
    msg,
  };
}
const show = (label, r) => console.log(`\n--- ${label}\ntools: ${r.tools || '(none)'}  charts: ${r.charts}\n${r.text.slice(0, 700)}\n`);

// 1. customer lookup (example button) → waterfall
await page.locator('button.example', { hasText: 'Profit breakdown for customer 1' }).click();
await page.locator('.msg.assistant').first().waitFor({ timeout: 10000 });
await page.locator('.msg.assistant').first().locator('.status').waitFor({ state: 'detached', timeout: 360000 });
let m = page.locator('.msg.assistant').first();
let r = { text: await m.locator('.md').first().innerText(), charts: await m.locator('figure.chart').count(), tools: await m.locator('.tools').innerText().catch(() => '') };
show('customer 1', r);
check('customer 1: used get_customer_profile', r.tools.includes('get_customer_profile'), r.tools);
check('customer 1: waterfall chart rendered', r.charts === 1);
check('customer 1: reports the real profit −$683.52', /683\.52/.test(r.text));
await page.screenshot({ path: `${SHOTS}/chat-customer1.png`, fullPage: false });

// 2. distribution → histogram
r = await ask('Show the profit distribution');
show('distribution', r);
check('distribution: histogram rendered', r.charts >= 1);
check('distribution: used get_distribution', r.tools.includes('get_distribution'), r.tools);

// 3. segment compare
r = await ask('Compare lounge usage between top 20% and rest');
show('lounge', r);
check('lounge: used segment_compare', r.tools.includes('segment_compare'), r.tools);
check('lounge: cites the means 0.58 vs 0.44', /0\.58/.test(r.text) && /0\.44/.test(r.text));

// 4. term comparison chart
r = await ask('Which cost term hurts profitability the most?');
show('cost terms', r);
check('terms: used a term/segment tool', /term_contribution_summary|segment_compare/.test(r.tools), r.tools);

// 5. retrieval over the core knowledge base (embeds the question in the browser)
r = await ask('How were the weights derived? Cite your source.');
show('retrieval', r);
check('retrieval: answer mentions the single-lever methodology', /single[- ]lever|response[- ]curve/i.test(r.text));

// 6. ad-hoc filter → full customer table downloaded in the browser
r = await ask('Using query_segment, how many customers have f3 >= 1 and profit > 0? Give the exact count.');
show('ad-hoc query', r);
check('query: used query_segment', r.tools.includes('query_segment'), r.tools);
check(`query: reports the exact count (${goldenCount})`, r.text.replace(/,/g, '').includes(String(goldenCount)));

// 7. upload a file, then ask about it
await page.getByRole('tab', { name: '📂 Digest your own files?' }).click();
const file = '/tmp/zephyr-notes.txt';
writeFileSync(file, 'Internal memo.\n\nThe Zephyr-9 partner program pays a 7.5% rebate on dining spend for enrolled cardmembers. ' +
  'Enrolment closes on 31 March. Questions go to the partnerships desk.\n');
await page.locator('.drop input[type=file]').setInputFiles(file);
await page.getByText('zephyr-notes.txt digested').waitFor({ timeout: 120000 });
check('upload: file digested and listed', await page.locator('table td', { hasText: 'zephyr-notes.txt' }).count() === 1);
await page.screenshot({ path: `${SHOTS}/digest.png` });
await page.getByRole('tab', { name: '💬 Chat' }).click();
check('upload: knowledge base counts the new chunks', /Knowledge base: 125 core \+ [1-9]\d* of your chunks/.test(await page.locator('body').innerText()));
r = await ask('What rebate does the Zephyr-9 partner program pay, and on what spend? Cite the file.');
show('uploaded file', r);
check('upload: answer uses the uploaded file', /7\.5\s*%/.test(r.text));
check('upload: cites the filename', /zephyr-notes/i.test(r.text));
await page.screenshot({ path: `${SHOTS}/chat-final.png`, fullPage: false });

const bad = logs.filter((l) => !/Failed to load resource|favicon|No available adapters/.test(l));
check('no console errors', bad.length === 0, bad.join(' | ').slice(0, 400));
await browser.close();
finish();
