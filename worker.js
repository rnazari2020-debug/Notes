// Runs Whisper in a background thread so the page stays responsive.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;

let asr = null;
let loadedModel = null;

self.onmessage = async (e) => {
  const { id, model, audio, lang } = e.data;
  try {
    if (!asr || loadedModel !== model) {
      asr = await pipeline('automatic-speech-recognition', model, {
        device: 'wasm',
        dtype: 'q8',
        progress_callback: (p) => self.postMessage({ id, type: 'progress', p }),
      });
      loadedModel = model;
    }
    const opts = { task: 'transcribe', chunk_length_s: 30, stride_length_s: 5, return_timestamps: false };
    if (lang) opts.language = lang;
    const out = await asr(audio, opts);
    self.postMessage({ id, type: 'done', text: (out.text || '').trim() });
  } catch (err) {
    self.postMessage({ id, type: 'error', message: String((err && err.message) || err) });
  }
};
