import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm';

// Runs the model off the main thread; the page talks to it through CreateWebWorkerMLCEngine.
const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg: MessageEvent) => handler.onmessage(msg);
