// Headless Chromium has no real WebGPU adapter, which is exactly the "no GPU" path:
// the page must load, show the data, explain the situation and offer the Ollama fallback.
import { BASE, SHOTS, check, finish, launch } from './lib.mjs';

const { browser, page, logs } = await launch();
await page.goto(BASE);
await page.getByRole('heading', { name: 'ProfitLens', level: 1 }).waitFor({ timeout: 30000 });

const text = async () => page.locator('body').innerText();
await page.waitForFunction(() => document.body.innerText.includes('Formula matches submitted predictions'), null, { timeout: 30000 });
let t = await text();
check('integrity badge', t.includes('Formula matches submitted predictions (500,000/500,000)'));
check('context stats badge', t.includes('Context stats verified'));
check('knowledge base badge', /Knowledge base: 125 core \+ 0 of your chunks/.test(t));
check('dataset cards', t.includes('500,000') && t.includes('$2,012.86') && t.includes('$4,354.85'));
check('six example prompts', (await page.locator('button.example').count()) === 6);

await page.waitForFunction(() => /No GPU adapter|no WebGPU|GPU adapter/i.test(document.body.innerText), null, { timeout: 15000 }).catch(() => {});
t = await text();
check('explains missing WebGPU', /no GPU adapter|no WebGPU|adapter request failed/i.test(t), t.match(/WebGPU[^.]*\./)?.[0]);
check('falls back to the Ollama tab', await page.getByLabel('Ollama URL').isVisible());
check('chat is disabled until a model is ready', await page.getByLabel('Message').isDisabled());
check('examples disabled until a model is ready', await page.locator('button.example').first().isDisabled());

await page.getByRole('tab', { name: '📂 Digest your own files?' }).click();
check('digest tab renders', (await text()).includes('No files digested yet'));
await page.screenshot({ path: `${SHOTS}/no-gpu.png` });

const bad = logs.filter((l) => !/Failed to load resource|favicon|No available adapters/.test(l));
check('no console errors', bad.length === 0, bad.join(' | ').slice(0, 300));
await browser.close();
finish();
