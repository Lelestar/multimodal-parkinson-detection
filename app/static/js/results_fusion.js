const fusionPanel = document.querySelector("#fusion-panel");
const fusionEmpty = document.querySelector("#fusion-empty");
const modalityResults = document.querySelector("#modality-results");
const fusionWarnings = document.querySelector("#fusion-warnings");

const expectedModalities = [
  { key: "keyboard", label: "Clavier" },
  { key: "voice", label: "Voix" },
  { key: "drawing", label: "Dessin" }
];

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

function statusText(status) {
  const labels = {
    ok: "Disponible",
    insufficient_data: "Données insuffisantes",
    error: "Erreur",
    missing: "Non réalisé"
  };
  return labels[status] ?? "-";
}

function readStoredPrediction(modality) {
  const raw = sessionStorage.getItem(`parkinson_result_${modality}`);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function renderWarnings(warnings) {
  fusionWarnings.replaceChildren();
  for (const warning of warnings ?? []) {
    const item = document.createElement("li");
    item.textContent = warning;
    fusionWarnings.appendChild(item);
  }
}

function renderModalityCards(predictionsByModality) {
  modalityResults.replaceChildren();
  for (const modality of expectedModalities) {
    const prediction = predictionsByModality[modality.key];
    const card = document.createElement("article");
    card.className = "modality-card";

    const title = document.createElement("h2");
    title.textContent = modality.label;
    card.appendChild(title);

    const status = document.createElement("p");
    status.className = "modality-status";
    status.textContent = prediction ? statusText(prediction.status) : "Non réalisé";
    card.appendChild(status);

    if (prediction?.status === "ok") {
      const score = document.createElement("p");
      score.textContent = `${riskText(prediction.label)} · ${percent(prediction.score)}`;
      card.appendChild(score);
    } else if (modality.key === "keyboard") {
      const link = document.createElement("a");
      link.className = "button";
      link.href = "/keyboard";
      link.textContent = "Faire le test";
      card.appendChild(link);
    } else if (modality.key === "drawing") {
      const link = document.createElement("a");
      link.className = "button";
      link.href = "/drawing";
      link.textContent = "Faire le test";
      card.appendChild(link);
    } else {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "À implémenter";
      card.appendChild(tag);
    }

    modalityResults.appendChild(card);
  }
}

async function loadFusion() {
  const predictions = expectedModalities
    .map((modality) => readStoredPrediction(modality.key))
    .filter(Boolean);

  const predictionsByModality = Object.fromEntries(
    predictions.map((prediction) => [prediction.modality, prediction])
  );
  renderModalityCards(predictionsByModality);

  if (predictions.length === 0) {
    fusionEmpty.hidden = false;
    fusionPanel.hidden = true;
    return;
  }

  fusionEmpty.hidden = true;
  fusionPanel.hidden = false;

  try {
    const response = await fetch("/api/fusion", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ predictions })
    });
    const result = await response.json();
    document.querySelector("#fusion-label").textContent = riskText(result.label);
    document.querySelector("#fusion-score").textContent = percent(result.score);
    document.querySelector("#fusion-confidence").textContent = percent(result.confidence);
    document.querySelector("#fusion-used").textContent = (result.used_modalities ?? []).length
      ? result.used_modalities.map((key) => expectedModalities.find((item) => item.key === key)?.label ?? key).join(", ")
      : "-";
    renderWarnings(result.warnings);
  } catch (error) {
    document.querySelector("#fusion-label").textContent = "-";
    document.querySelector("#fusion-score").textContent = "-";
    document.querySelector("#fusion-confidence").textContent = "-";
    document.querySelector("#fusion-used").textContent = "-";
    renderWarnings([`Erreur réseau ou serveur : ${error}`]);
  }
}

loadFusion();

// Reset all results 
function resetAllResults() {
  for (const modality of expectedModalities) {
    sessionStorage.removeItem(`parkinson_result_${modality.key}`);
  }
  window.location.reload();
}

document.querySelector("#reset-all-button")?.addEventListener("click", resetAllResults);
