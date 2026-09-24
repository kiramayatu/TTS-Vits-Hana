const $ = (id) => document.getElementById(id);
let currentBlobUrl = null;
let lastBlob = null;
let lastFilename = 'hana-tts.wav';
let streamSocket = null;
let streamBlobs = [];
let audioContext = null;
let scheduledUntil = 0;
let playbackSources = [];
let streamGeneration = 0;
let streamGapMs = 30;
let activeGeneration = null;
let statusRequestInFlight = false;

const DEFAULT_VOICE_PROFILE = Object.freeze({
  speed: 1.0,
  noise_scale: 0.667,
  noise_scale_w: 0.8,
  sentence_max: 140
});
const PROFILE_STORAGE_KEY = 'hanaVits.voiceProfiles.v1';

function readProfiles() {
  try {
    const raw = localStorage.getItem(PROFILE_STORAGE_KEY);
    const data = raw ? JSON.parse(raw) : {};
    return (data && typeof data === 'object' && !Array.isArray(data)) ? data : {};
  } catch (_) {
    return {};
  }
}

function writeProfiles(profiles) {
  localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profiles));
}

function currentVoiceProfile() {
  return {
    speed: Number($('speed').value),
    noise_scale: Number($('noiseScale').value),
    noise_scale_w: Number($('noiseScaleW').value),
    sentence_max: Number($('sentenceMax').value)
  };
}

function applyVoiceProfile(profile) {
  const merged = {...DEFAULT_VOICE_PROFILE, ...profile};
  $('speed').value = merged.speed;
  $('noiseScale').value = merged.noise_scale;
  $('noiseScaleW').value = merged.noise_scale_w;
  $('sentenceMax').value = merged.sentence_max;
  $('speed').dispatchEvent(new Event('input'));
  $('noiseScale').dispatchEvent(new Event('input'));
  $('noiseScaleW').dispatchEvent(new Event('input'));
  $('sentenceMax').dispatchEvent(new Event('input'));
}

function refreshProfileSelect(selectedName = '') {
  const profiles = readProfiles();
  const names = Object.keys(profiles).sort((a, b) => a.localeCompare(b));
  const select = $('profileSelect');
  select.innerHTML = names.length
    ? names.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('')
    : '<option value="">No saved profiles</option>';
  if (selectedName && profiles[selectedName]) select.value = selectedName;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
}

async function loadStatus() {
  if (statusRequestInFlight) return;
  statusRequestInFlight = true;
  try {
    const res = await fetch('/status', {cache: 'no-store'});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const dtype = data.dtype || 'unknown';
    $('status').textContent = data.loaded ? `Ready · ${data.device} · ${dtype}` : 'Model not loaded';

    const gpu = data.gpu;
    if (gpu && Number.isFinite(Number(gpu.memory_total_mb)) && Number(gpu.memory_total_mb) > 0) {
      const allocated = Number(gpu.memory_allocated_mb || 0);
      const reserved = Number(gpu.memory_reserved_mb || 0);
      const total = Number(gpu.memory_total_mb);
      const percent = (allocated / total) * 100;
      $('gpuInfo').textContent = `${gpu.name || 'CUDA GPU'} · VRAM ${allocated.toFixed(0)} / ${total.toFixed(0)} MB (${percent.toFixed(1)}%) · reserved ${reserved.toFixed(0)} MB`;
    } else {
      $('gpuInfo').textContent = 'GPU VRAM: unavailable';
    }
  } catch (err) {
    $('status').textContent = 'Server unavailable';
    $('gpuInfo').textContent = 'GPU VRAM: unavailable';
    $('message').textContent = String(err);
  } finally {
    statusRequestInFlight = false;
  }
}

async function loadVoiceOptions() {
  try {
    const speakerRes = await fetch('/speakers', {cache: 'no-store'});
    if (!speakerRes.ok) throw new Error(`HTTP ${speakerRes.status}`);
    const speakerData = await speakerRes.json();
    const currentSpeaker = $('speaker').value;
    const currentLanguage = $('language').value;
    $('speaker').innerHTML = Object.keys(speakerData.speakers)
      .map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
    $('language').innerHTML = speakerData.languages
      .map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
    if (currentSpeaker && [...$('speaker').options].some(o => o.value === currentSpeaker)) $('speaker').value = currentSpeaker;
    if (currentLanguage && [...$('language').options].some(o => o.value === currentLanguage)) $('language').value = currentLanguage;
  } catch (err) {
    $('message').textContent = `Could not load voice options: ${err.message || err}`;
  }
}

function setRange(id, outputId, digits = 1) {
  const input = $(id);
  const output = $(outputId);
  const update = () => output.textContent = Number(input.value).toFixed(digits);
  input.addEventListener('input', update);
  update();
}

setRange('speed', 'speedValue', 1);
setRange('noiseScale', 'noiseScaleValue', 3);
setRange('noiseScaleW', 'noiseScaleWValue', 3);
setRange('sentenceMax', 'sentenceMaxValue', 0);

$('saveProfile').addEventListener('click', () => {
  const name = $('profileName').value.trim();
  if (!name) {
    $('message').textContent = 'Enter a profile name first.';
    $('profileName').focus();
    return;
  }
  const profiles = readProfiles();
  profiles[name] = currentVoiceProfile();
  try {
    writeProfiles(profiles);
    refreshProfileSelect(name);
    $('message').textContent = `Profile saved: ${name}`;
  } catch (err) {
    $('message').textContent = `Could not save profile: ${err.message || err}`;
  }
});

$('loadProfile').addEventListener('click', () => {
  const name = $('profileSelect').value;
  const profiles = readProfiles();
  if (!name || !profiles[name]) {
    $('message').textContent = 'Select a saved profile first.';
    return;
  }
  applyVoiceProfile(profiles[name]);
  $('profileName').value = name;
  $('message').textContent = `Profile loaded: ${name}`;
});

$('deleteProfile').addEventListener('click', () => {
  const name = $('profileSelect').value;
  if (!name) {
    $('message').textContent = 'Select a saved profile first.';
    return;
  }
  const profiles = readProfiles();
  delete profiles[name];
  writeProfiles(profiles);
  refreshProfileSelect();
  if ($('profileName').value === name) $('profileName').value = '';
  $('message').textContent = `Profile deleted: ${name}`;
});

$('resetVoice').addEventListener('click', () => {
  applyVoiceProfile(DEFAULT_VOICE_PROFILE);
  $('message').textContent = 'Voice controls reset to defaults.';
});

refreshProfileSelect();

function payload() {
  return {
    text: $('text').value,
    speaker: $('speaker').value,
    language: $('language').value,
    speed: Number($('speed').value),
    noise_scale: Number($('noiseScale').value),
    noise_scale_w: Number($('noiseScaleW').value),
    sentence_max: Number($('sentenceMax').value)
  };
}

function showMetrics(headers) {
  const metrics = [
    ['generation', `${Number(headers.get('X-TTS-Generation-Ms') || 0).toFixed(0)} ms`],
    ['audio', `${Number(headers.get('X-TTS-Audio-Seconds') || 0).toFixed(2)} s`],
    ['RTF', Number(headers.get('X-TTS-RTF') || 0).toFixed(3)],
    ['device', headers.get('X-TTS-Device') || 'unknown'],
    ['dtype', headers.get('X-TTS-Dtype') || 'unknown'],
    ['cache', headers.get('X-TTS-Cache-Hit') === '1' ? 'hit' : 'miss'],
    ['text cache', headers.get('X-TTS-Text-Cache-Hit') === '1' ? 'hit' : 'miss'],
    ['preprocess', `${Number(headers.get('X-TTS-Preprocess-Ms') || 0).toFixed(1)} ms`],
    ['inference', `${Number(headers.get('X-TTS-Inference-Ms') || 0).toFixed(1)} ms`],
    ['GPU memory', `${Number(headers.get('X-TTS-GPU-Memory-MB') || 0).toFixed(0)} MB`],
    ['segments', headers.get('X-TTS-Segments') || '1']
  ];
  $('metrics').innerHTML = metrics.map(([k,v]) => `<span class="metric"><b>${k}</b>: ${escapeHtml(v)}</span>`).join('');
}

function blobFilename(contentDisposition, fallback = 'hana-tts.wav') {
  return contentDisposition?.match(/filename="([^"]+)"/)?.[1] || fallback;
}

async function generateWithProgressivePlayback(runId) {
  streamBlobs = [];
  streamGapMs = 30;
  stopPlayback();

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws/tts`);
  socket.binaryType = 'blob';
  streamSocket = socket;

  let segmentCount = 0;
  let totalAudio = 0;
  let totalGeneration = 0;
  let completed = false;
  let settled = false;

  return await new Promise((resolve, reject) => {
    const cleanup = () => {
      if (activeGeneration?.runId === runId) activeGeneration = null;
      if (streamSocket === socket) streamSocket = null;
    };

    const fail = (error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error instanceof Error ? error : new Error(String(error)));
    };

    const cancel = () => {
      if (settled) return;
      settled = true;
      // The browser may still be CONNECTING when Stop is pressed. In that
      // state we cannot send a message yet, so onopen below must detect the
      // stale generation and close without sending start/append/flush.
      if (socket.readyState === WebSocket.OPEN) {
        try { socket.send(JSON.stringify({action: 'cancel'})); } catch (_) {}
      }
      try {
        if (socket.readyState !== WebSocket.CLOSED && socket.readyState !== WebSocket.CLOSING) {
          socket.close(1000, 'user stop');
        }
      } catch (_) {}
      cleanup();
      reject(new Error('Generation cancelled'));
    };

    activeGeneration = {runId, cancel, socket};

    socket.onopen = () => {
      // Critical race guard: Stop can happen while the socket is CONNECTING.
      if (runId !== streamGeneration || settled) {
        cancel();
        return;
      }
      try {
        socket.send(JSON.stringify({
          action: 'start',
          speaker: $('speaker').value,
          language: $('language').value,
          speed: Number($('speed').value),
          noise_scale: Number($('noiseScale').value),
          noise_scale_w: Number($('noiseScaleW').value),
          max_chars: Number($('sentenceMax').value)
        }));
        socket.send(JSON.stringify({action: 'append', text: $('text').value}));
        socket.send(JSON.stringify({action: 'flush'}));
      } catch (err) {
        fail(err);
      }
    };

    socket.onmessage = async (event) => {
      if (runId !== streamGeneration || settled) return;
      if (typeof event.data === 'string') {
        let data;
        try {
          data = JSON.parse(event.data);
        } catch (err) {
          fail(new Error(`Invalid server message: ${err.message || err}`));
          return;
        }
        if (data.type === 'ready') {
          streamGapMs = Number(data.segment_gap_ms ?? 30);
          $('message').textContent = `Generating ${data.max_chars}-character sentence chunks and playing as ready…`;
        } else if (data.type === 'segment') {
          segmentCount += 1;
          totalAudio += Number(data.audio_seconds || 0);
          totalGeneration += Number(data.generation_ms || 0);
          $('message').textContent = `Playing generated sentence ${segmentCount}: ${data.text}`;
          $('metrics').innerHTML = [
            ['segments', segmentCount],
            ['audio generated', `${totalAudio.toFixed(2)} s`],
            ['generation', `${totalGeneration.toFixed(0)} ms`],
            ['last RTF', Number(data.rtf || 0).toFixed(3)],
            ['cache', data.cache_hit ? 'hit' : 'miss']
          ].map(([k,v]) => `<span class="metric"><b>${k}</b>: ${escapeHtml(v)}</span>`).join('');
        } else if (data.type === 'done') {
          if (settled) return;
          completed = true;
          if (!streamBlobs.length) {
            fail(new Error('The TTS server returned no audio segments'));
            return;
          }
          try {
            const combined = await combineWavs(streamBlobs);
            if (runId !== streamGeneration || settled) return;
            lastBlob = combined;
            if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
            currentBlobUrl = URL.createObjectURL(combined);
            $('audio').src = currentBlobUrl;
            $('download').disabled = false;
            lastFilename = `hana-${Date.now()}.wav`;
            $('message').textContent = `Done. Generated ${segmentCount} sentence chunk(s) and started playback.`;
            $('metrics').innerHTML += `<span class="metric"><b>final audio</b>: ${totalAudio.toFixed(2)} s</span>`;
            settled = true;
            cleanup();
            resolve({segmentCount, totalAudio, totalGeneration});
          } catch (err) {
            fail(err);
          } finally {
            try { socket.close(1000, 'generation complete'); } catch (_) {}
          }
        } else if (data.type === 'cancelled') {
          fail(new Error('Generation cancelled'));
        } else if (data.type === 'error') {
          fail(new Error(data.detail || 'TTS server error'));
        }
        return;
      }

      // Do not schedule stale audio that was decoded after Stop was pressed.
      if (runId !== streamGeneration || settled) return;
      streamBlobs.push(event.data);
      try {
        await scheduleStreamBlob(event.data, runId);
      } catch (err) {
        if (runId === streamGeneration && !settled) {
          fail(new Error(`Playback error: ${err.message || err}`));
        }
      }
    };

    socket.onerror = () => fail(new Error('WebSocket TTS connection failed.'));

    socket.onclose = (event) => {
      const wasCurrent = runId === streamGeneration;
      cleanup();
      if (!settled && wasCurrent && event.code !== 1000) {
        fail(new Error(`TTS connection closed unexpectedly (code ${event.code}).`));
      }
    };
  });
}

$('generate').addEventListener('click', async () => {
  const runId = ++streamGeneration;
  const button = $('generate');
  button.disabled = true;
  $('stop').disabled = false;
  $('download').disabled = true;
  $('metrics').innerHTML = '';
  $('message').textContent = 'Starting generation…';

  try {
    await ensureAudioContext();
    await generateWithProgressivePlayback(runId);
  } catch (err) {
    if (runId === streamGeneration) {
      $('message').textContent = `Error: ${err.message || err}`;
    }
    if (runId === streamGeneration && String(err.message || '').toLowerCase().includes('cancelled')) {
      $('message').textContent = 'Generation stopped.';
    }
  } finally {
    if (runId === streamGeneration) {
      button.disabled = false;
      $('stop').disabled = playbackSources.length === 0 && streamSocket === null;
    }
  }
});

async function ensureAudioContext() {
  if (!audioContext || audioContext.state === 'closed') {
    audioContext = new AudioContext();
  }
  if (audioContext.state === 'suspended') await audioContext.resume();
  return audioContext;
}

async function scheduleStreamBlob(blob, runId = streamGeneration) {
  const ctx = await ensureAudioContext();
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
  if (runId !== streamGeneration) return;
  if (!activeGeneration || activeGeneration.runId !== runId) return;
  const start = Math.max(ctx.currentTime + 0.03, scheduledUntil);
  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ctx.destination);
  source.onended = () => {
    playbackSources = playbackSources.filter(item => item !== source);
    if (playbackSources.length === 0 && streamSocket === null) $('stop').disabled = true;
  };
  source.start(start);
  playbackSources.push(source);
  scheduledUntil = start + audioBuffer.duration + (streamGapMs / 1000);
}

function stopPlayback() {
  for (const source of playbackSources) {
    try { source.onended = null; source.stop(); } catch (_) {}
  }
  playbackSources = [];
  scheduledUntil = 0;
}

function pcm16FromWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const riff = new TextDecoder().decode(new Uint8Array(arrayBuffer, 0, 4));
  if (riff !== 'RIFF') throw new Error('Expected WAV/RIFF data');
  let offset = 12;
  let sampleRate = 22050;
  let channels = 1;
  let bits = 16;
  let pcmOffset = -1;
  let pcmLength = 0;
  while (offset + 8 <= view.byteLength) {
    const id = new TextDecoder().decode(new Uint8Array(arrayBuffer, offset, 4));
    const size = view.getUint32(offset + 4, true);
    if (id === 'fmt ') {
      const format = view.getUint16(offset + 8, true);
      channels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bits = view.getUint16(offset + 22, true);
      if (format !== 1 || bits !== 16) throw new Error('Only PCM16 WAV is supported for combined download');
    } else if (id === 'data') {
      pcmOffset = offset + 8;
      pcmLength = size;
      break;
    }
    offset += 8 + size + (size % 2);
  }
  if (pcmOffset < 0 || channels !== 1 || bits !== 16) throw new Error('Unsupported WAV layout');
  return { bytes: new Uint8Array(arrayBuffer, pcmOffset, pcmLength), sampleRate };
}

async function combineWavs(blobs) {
  const parts = await Promise.all(blobs.map(async blob => pcm16FromWav(await blob.arrayBuffer())));
  if (!parts.length) throw new Error('No generated audio segments available');
  const sampleRate = parts[0].sampleRate;
  if (parts.some(p => p.sampleRate !== sampleRate)) throw new Error('Segment sample rates do not match');
  const gapBytes = Math.max(0, Math.round(sampleRate * streamGapMs / 1000) * 2);
  const total = parts.reduce((sum, p) => sum + p.bytes.byteLength, 0) + gapBytes * Math.max(0, parts.length - 1);
  const out = new ArrayBuffer(44 + total);
  const view = new DataView(out);
  const writeText = (offset, text) => { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)); };
  writeText(0, 'RIFF');
  view.setUint32(4, 36 + total, true);
  writeText(8, 'WAVE');
  writeText(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, 'data');
  view.setUint32(40, total, true);
  let cursor = 44;
  const gap = new Uint8Array(gapBytes);
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    new Uint8Array(out, cursor, part.bytes.byteLength).set(part.bytes);
    cursor += part.bytes.byteLength;
    if (i + 1 < parts.length && gapBytes > 0) {
      new Uint8Array(out, cursor, gapBytes).set(gap);
      cursor += gapBytes;
    }
  }
  return new Blob([out], {type: 'audio/wav'});
}

$('stop').addEventListener('click', () => {
  streamGeneration += 1;
  stopPlayback();
  const active = activeGeneration;
  if (active) {
    try { active.cancel(); } catch (_) {}
  } else if (streamSocket) {
    try {
      if (streamSocket.readyState === WebSocket.OPEN) {
        streamSocket.send(JSON.stringify({action: 'cancel'}));
      }
    } catch (_) {}
    try {
      if (streamSocket.readyState !== WebSocket.CLOSED && streamSocket.readyState !== WebSocket.CLOSING) {
        streamSocket.close(1000, 'user stop');
      }
    } catch (_) {}
    streamSocket = null;
  }
  activeGeneration = null;
  $('generate').disabled = false;
  $('stop').disabled = true;
  $('message').textContent = 'Generation/playback stopped.';
});

$('download').addEventListener('click', () => {
  if (!lastBlob) return;
  const link = document.createElement('a');
  link.href = currentBlobUrl;
  link.download = lastFilename;
  document.body.appendChild(link);
  link.click();
  link.remove();
});

loadStatus();
loadVoiceOptions();
setInterval(loadStatus, 2000);
