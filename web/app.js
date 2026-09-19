const $ = (id) => document.getElementById(id);
let currentBlobUrl = null;
let lastBlob = null;
let lastFilename = 'hana-tts.wav';

async function loadStatus() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    $('status').textContent = data.loaded ? `Ready · ${data.device}` : 'Model not loaded';
    const speakerRes = await fetch('/speakers');
    const speakerData = await speakerRes.json();
    $('speaker').innerHTML = Object.keys(speakerData.speakers)
      .map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
    $('language').innerHTML = speakerData.languages
      .map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
  } catch (err) {
    $('status').textContent = 'Server unavailable';
    $('message').textContent = String(err);
  }
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
}

$('speed').addEventListener('input', () => {
  $('speedValue').value = `${Number($('speed').value).toFixed(1)}×`;
  $('speedValue').textContent = `${Number($('speed').value).toFixed(1)}×`;
});

$('generate').addEventListener('click', async () => {
  const button = $('generate');
  button.disabled = true;
  $('download').disabled = true;
  $('message').textContent = 'Generating…';
  $('metrics').innerHTML = '';

  try {
    const payload = {
      text: $('text').value,
      speaker: $('speaker').value,
      language: $('language').value,
      speed: Number($('speed').value)
    };
    const response = await fetch('/tts', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try { detail = (await response.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }

    lastBlob = await response.blob();
    if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = URL.createObjectURL(lastBlob);
    $('audio').src = currentBlobUrl;
    $('download').disabled = false;
    lastFilename = response.headers.get('Content-Disposition')?.match(/filename="([^"]+)"/)?.[1] || 'hana-tts.wav';
    $('message').textContent = 'Done.';

    const metrics = [
      ['generation', `${Number(response.headers.get('X-TTS-Generation-Ms') || 0).toFixed(0)} ms`],
      ['audio', `${Number(response.headers.get('X-TTS-Audio-Seconds') || 0).toFixed(2)} s`],
      ['RTF', Number(response.headers.get('X-TTS-RTF') || 0).toFixed(3)],
      ['device', response.headers.get('X-TTS-Device') || 'unknown'],
      ['GPU memory', `${Number(response.headers.get('X-TTS-GPU-Memory-MB') || 0).toFixed(0)} MB`]
    ];
    $('metrics').innerHTML = metrics.map(([k,v]) => `<span class="metric"><b>${k}</b>: ${escapeHtml(v)}</span>`).join('');
  } catch (err) {
    $('message').textContent = `Error: ${err.message || err}`;
  } finally {
    button.disabled = false;
  }
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
