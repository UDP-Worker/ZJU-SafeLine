const input = document.getElementById("videoInput");
const nameBox = document.getElementById("fileName");
const videoEl = document.getElementById("streamVideo");
const statusText = document.getElementById("statusText");
const statusChip = document.getElementById("statusChip");
const videoHint = document.getElementById("videoHint");
if (input && nameBox) {
  input.addEventListener("change", () => {
    if (input.files && input.files.length > 0) {
      nameBox.textContent = input.files[0].name;
    } else {
      nameBox.textContent = "未选择任何文件";
    }
  });
}
const setVideoSource = (url) => {
  if (!url || !videoEl) return;
  if (videoEl.dataset.src === url) return;
  videoEl.dataset.src = url;
  videoEl.src = url;
  videoEl.load();
  const playPromise = videoEl.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {});
  }
  if (videoHint) {
    videoHint.style.display = "none";
  }
};
const updateStatus = (payload) => {
  if (!payload) return;
  if (payload.playback_url) {
    setVideoSource(payload.playback_url);
  }
  if (videoHint) {
    videoHint.style.display = payload.playback_url ? "none" : "block";
  }
  const label = payload.label || "idle";
  const prob = typeof payload.prob === "number" ? payload.prob : 0;
  if (statusText) {
    statusText.textContent = `${label} (${prob.toFixed(2)})`;
  }
  if (statusChip) {
    statusChip.classList.toggle("bad", label === "abnormal");
    statusChip.classList.toggle("good", label === "normal");
  }
};
const sendPlayback = () => {
  if (!videoEl || !videoEl.src) return;
  const payload = {
    time: videoEl.currentTime || 0,
    paused: videoEl.paused,
  };
  fetch("/playback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).catch(() => {});
};
let playbackTimer = null;
const startPlaybackSync = () => {
  if (playbackTimer) return;
  playbackTimer = setInterval(sendPlayback, 250);
};
const stopPlaybackSync = () => {
  if (!playbackTimer) return;
  clearInterval(playbackTimer);
  playbackTimer = null;
};
if (videoEl) {
  videoEl.addEventListener("play", () => {
    startPlaybackSync();
    sendPlayback();
  });
  videoEl.addEventListener("pause", () => {
    stopPlaybackSync();
    sendPlayback();
  });
  videoEl.addEventListener("seeked", () => {
    sendPlayback();
  });
  videoEl.addEventListener("loadedmetadata", () => {
    sendPlayback();
  });
}
const refreshStatus = () => {
  fetch("/status")
    .then((resp) => resp.json())
    .then((payload) => updateStatus(payload))
    .catch(() => {});
};
const connectWs = () => {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
  ws.onmessage = (evt) => {
    try {
      const payload = JSON.parse(evt.data);
      updateStatus(payload);
    } catch (err) {}
  };
  ws.onclose = () => {
    setTimeout(connectWs, 2000);
  };
};
refreshStatus();
connectWs();
