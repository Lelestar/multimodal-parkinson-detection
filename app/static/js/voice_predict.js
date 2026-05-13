/**
 * Exemples issus de deux lignes réelles de parkinsons.data (dataset Oxford / UCI).
 * Clés = noms de colonnes attendus par l'API.
 */
const VOICE_EXAMPLES = {
  control: {
    "MDVP:Fo(Hz)": 197.076,
    "MDVP:Fhi(Hz)": 206.896,
    "MDVP:Flo(Hz)": 192.055,
    "MDVP:Jitter(%)": 0.00289,
    "MDVP:Jitter(Abs)": 0.00001,
    "MDVP:RAP": 0.00166,
    "MDVP:PPQ": 0.00168,
    "Jitter:DDP": 0.00498,
    "MDVP:Shimmer": 0.01098,
    "MDVP:Shimmer(dB)": 0.097,
    "Shimmer:APQ3": 0.00563,
    "Shimmer:APQ5": 0.0068,
    "MDVP:APQ": 0.00802,
    "Shimmer:DDA": 0.01689,
    NHR: 0.00339,
    HNR: 26.775,
    RPDE: 0.422229,
    DFA: 0.741367,
    spread1: -7.3483,
    spread2: 0.177551,
    D2: 1.743867,
    PPE: 0.085569
  },
  parkinson: {
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784,
    "MDVP:Jitter(Abs)": 0.00007,
    "MDVP:RAP": 0.0037,
    "MDVP:PPQ": 0.00554,
    "Jitter:DDP": 0.01109,
    "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426,
    "Shimmer:APQ3": 0.02182,
    "Shimmer:APQ5": 0.0313,
    "MDVP:APQ": 0.02971,
    "Shimmer:DDA": 0.06545,
    NHR: 0.02211,
    HNR: 21.033,
    RPDE: 0.414783,
    DFA: 0.815285,
    spread1: -4.813031,
    spread2: 0.266482,
    D2: 2.301442,
    PPE: 0.284654
  }
};

function fillForm(example) {
  for (const input of document.querySelectorAll("[data-voice-feature]")) {
    const key = input.getAttribute("data-voice-feature");
    if (key && Object.prototype.hasOwnProperty.call(example, key)) {
      input.value = String(example[key]);
    }
  }
}

function collectPayload() {
  const features = {};
  for (const input of document.querySelectorAll("[data-voice-feature]")) {
    const key = input.getAttribute("data-voice-feature");
    const raw = input.value.trim();
    if (raw === "") {
      continue;
    }
    features[key] = raw.includes(",") ? raw.replace(",", ".") : raw;
  }
  return { features };
}

function percent(value) {
  if (value === null || value === undefined) return "-";
  return `${Math.round(Number(value) * 1000) / 10}%`;
}

function riskText(label) {
  const labels = {
    low: "Signal faible",
    moderate: "Signal modéré",
    elevated: "Signal élevé"
  };
  return labels[label] ?? "-";
}

function statusFr(status) {
  const m = { ok: "ok", insufficient_data: "données insuffisantes", error: "erreur" };
  return m[status] ?? status;
}

function storeVoiceResult(result) {
  if (result.status !== "ok") return;
  sessionStorage.setItem(
    "parkinson_result_voice",
    JSON.stringify({
      ...result,
      saved_at: new Date().toISOString()
    })
  );
}

function renderResult(result) {
  const panel = document.querySelector("#result-panel");
  panel.hidden = false;
  document.querySelector("#result-status").textContent = statusFr(result.status);
  document.querySelector("#result-label").textContent = riskText(result.label);
  document.querySelector("#result-score").textContent = percent(result.score);
  document.querySelector("#result-confidence").textContent = percent(result.confidence);
  document.querySelector("#result-model").textContent = result.details?.model_name ?? "-";
  document.querySelector("#result-details").textContent = JSON.stringify(result.details ?? {}, null, 2);

  const warningList = document.querySelector("#result-warnings");
  warningList.replaceChildren();
  for (const warning of result.warnings ?? []) {
    const item = document.createElement("li");
    item.textContent = warning;
    warningList.appendChild(item);
  }
}

document.querySelector("#btn-example-control").addEventListener("click", () => fillForm(VOICE_EXAMPLES.control));
document.querySelector("#btn-example-pd").addEventListener("click", () => fillForm(VOICE_EXAMPLES.parkinson));

document.querySelector("#btn-predict").addEventListener("click", async () => {
  const btn = document.querySelector("#btn-predict");
  btn.disabled = true;
  btn.textContent = "Analyse...";
  try {
    const response = await fetch("/api/voice/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload())
    });
    const result = await response.json();
    storeVoiceResult(result);
    renderResult(result);
  } catch (error) {
    renderResult({
      modality: "voice",
      status: "error",
      confidence: 0.0,
      warnings: [`Erreur réseau ou serveur : ${error}`],
      details: {}
    });
  } finally {
    btn.disabled = false;
    btn.textContent = "Prédire";
  }
});
