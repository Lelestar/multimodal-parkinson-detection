const typingArea = document.querySelector("#typing-area");
const resetButton = document.querySelector("#reset-button");
const submitButton = document.querySelector("#submit-button");
const resultPanel = document.querySelector("#result-panel");
const sampleStatusLabel = document.querySelector("#sample-status-label");
const sampleStatusDetail = document.querySelector("#sample-status-detail");
const sampleProgressBar = document.querySelector("#sample-progress-bar");
const events = [];
const sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
const startedAt = new Date().toISOString();
const minimumKeystrokes = 300;
const recommendedKeystrokes = 600;

const leftCodes = new Set([
  "KeyQ", "KeyW", "KeyE", "KeyR", "KeyT", "KeyA", "KeyS", "KeyD", "KeyF", "KeyG",
  "KeyZ", "KeyX", "KeyC", "KeyV", "KeyB"
]);
const rightCodes = new Set([
  "KeyY", "KeyU", "KeyI", "KeyO", "KeyP", "KeyH", "KeyJ", "KeyK", "KeyL", "KeyN", "KeyM"
]);
const spacePunctCodes = new Set([
  "Space", "Comma", "Period", "Semicolon", "Slash", "Minus", "Equal", "Quote", "Backquote",
  "BracketLeft", "BracketRight", "Backslash", "Enter"
]);
const excludedCodes = new Set([
  "Backspace", "Delete", "ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight",
  "AltLeft", "AltRight", "MetaLeft", "MetaRight", "CapsLock", "Tab", "Escape"
]);
const activeKeys = new Map();
let validKeystrokes = 0;

function keySide(code) {
  if (leftCodes.has(code)) return "left";
  if (rightCodes.has(code)) return "right";
  return "other";
}

function keyCategory(code) {
  if (spacePunctCodes.has(code)) return "space_punct";
  if (leftCodes.has(code) || rightCodes.has(code)) return "letter";
  if (code.startsWith("Digit")) return "digit";
  return "other";
}

function captureEvent(event) {
  const timestamp = performance.now();
  events.push({
    type: event.type,
    code: event.code,
    key: event.key,
    repeat: event.repeat,
    timestamp_ms: timestamp,
    key_side: keySide(event.code),
    key_category: keyCategory(event.code)
  });
  updateValidKeystrokes(event, timestamp);
  updateSampleStatus();
}

typingArea.addEventListener("keydown", captureEvent);
typingArea.addEventListener("keyup", captureEvent);
typingArea.addEventListener("paste", (event) => {
  event.preventDefault();
});

function updateValidKeystrokes(event, timestamp) {
  if (!event.code || excludedCodes.has(event.code)) return;
  if (event.type === "keydown") {
    if (!event.repeat && !activeKeys.has(event.code)) {
      activeKeys.set(event.code, timestamp);
    }
    return;
  }
  if (event.type === "keyup" && activeKeys.has(event.code)) {
    const pressTimestamp = activeKeys.get(event.code);
    activeKeys.delete(event.code);
    const holdSeconds = (timestamp - pressTimestamp) / 1000;
    if (holdSeconds >= 0 && holdSeconds <= 5) {
      validKeystrokes += 1;
    }
  }
}

function sampleLabel(count) {
  if (count < minimumKeystrokes) return "Échantillon trop court";
  if (count < recommendedKeystrokes) return "Analyse possible";
  return "Échantillon confortable";
}

function updateSampleStatus() {
  const cappedProgress = Math.min(validKeystrokes, recommendedKeystrokes);
  const progressPercent = Math.round((cappedProgress / recommendedKeystrokes) * 100);
  sampleStatusLabel.textContent = sampleLabel(validKeystrokes);
  sampleProgressBar.style.width = `${progressPercent}%`;
  submitButton.disabled = validKeystrokes < minimumKeystrokes;
  sampleStatusDetail.textContent =
    `${validKeystrokes} / ${minimumKeystrokes} frappes valides minimum. Recommandé : ${recommendedKeystrokes} ou plus.`;
}

resetButton.addEventListener("click", () => {
  typingArea.value = "";
  events.length = 0;
  activeKeys.clear();
  validKeystrokes = 0;
  updateSampleStatus();
  resultPanel.hidden = true;
  typingArea.focus();
});

function percent(value) {
  if (value === null || value === undefined) return "-";
  return `${Math.round(Number(value) * 1000) / 10}%`;
}

function statusText(status) {
  const labels = {
    ok: "Analyse réalisée",
    insufficient_data: "Données insuffisantes",
    error: "Erreur"
  };
  return labels[status] ?? "-";
}

function riskText(label) {
  const labels = {
    low: "Signal faible",
    moderate: "Signal modéré",
    elevated: "Signal élevé"
  };
  return labels[label] ?? "-";
}

function storeKeyboardResult(result) {
  if (result.status !== "ok") return;
  sessionStorage.setItem("parkinson_result_keyboard", JSON.stringify({
    ...result,
    saved_at: new Date().toISOString()
  }));
}

function renderResult(result) {
  resultPanel.hidden = false;
  document.querySelector("#result-score").textContent = percent(result.score);
  document.querySelector("#result-label").textContent = riskText(result.label);
  document.querySelector("#result-confidence").textContent = percent(result.confidence);
  document.querySelector("#result-keystrokes").textContent = result.details?.valid_keystrokes ?? "-";

  const warningList = document.querySelector("#result-warnings");
  warningList.replaceChildren();
  for (const warning of result.warnings ?? []) {
    const item = document.createElement("li");
    item.textContent = warning;
    warningList.appendChild(item);
  }
}

submitButton.addEventListener("click", async () => {
  submitButton.disabled = true;
  submitButton.textContent = "Analyse...";
  try {
    const response = await fetch("/api/keyboard/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, started_at: startedAt, events })
    });
    const result = await response.json();
    storeKeyboardResult(result);
    renderResult(result);
  } catch (error) {
    renderResult({
      status: "error",
      warnings: [`Erreur réseau ou serveur: ${error}`],
      details: {}
    });
  } finally {
    submitButton.textContent = "Analyser";
    updateSampleStatus();
  }
});

updateSampleStatus();
