const guideCanvas   = document.querySelector("#guide-canvas");
const drawCanvas    = document.querySelector("#draw-canvas");
const drawingActions = document.querySelector("#drawing-actions");
const waitingHint   = document.querySelector("#waiting-hint");
const resetButton   = document.querySelector("#reset-button");
const submitButton  = document.querySelector("#submit-button");
const resultPanel   = document.querySelector("#result-panel");

const guideCtx = guideCanvas.getContext("2d");
const drawCtx  = drawCanvas.getContext("2d");

let isDrawing  = false;
let hasDrawn   = false;
let lastPoint  = null;   
let smoothPos  = null;  

function drawGuide() {
  const w        = guideCanvas.width;
  const h        = guideCanvas.height;
  const cx       = w / 2;
  const cy       = h / 2;
  const turns    = 3;
  const maxTheta = turns * 2 * Math.PI;
  const maxR     = (Math.min(w, h) / 2) * 0.88;
  const b        = maxR / maxTheta;   

  guideCtx.fillStyle = "#ffffff";
  guideCtx.fillRect(0, 0, w, h);

  guideCtx.beginPath();
  const steps = 3000;
  for (let i = 0; i <= steps; i++) {
    const theta = (i / steps) * maxTheta;
    const r = b * theta;
    const x = cx - r * Math.cos(theta);  
    const y = cy + r * Math.sin(theta);  
    if (i === 0) guideCtx.moveTo(x, y);
    else         guideCtx.lineTo(x, y);
  }
  guideCtx.strokeStyle = "#111111";  
  guideCtx.lineWidth   = 1.5;
  guideCtx.stroke();
}

function clearDrawing() {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
}

function getPos(event) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width  / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  const src = event.touches ? event.touches[0] : event;
  return {
    x: (src.clientX - rect.left) * scaleX,
    y: (src.clientY - rect.top)  * scaleY,
  };
}

function onPointerDown(event) {
  event.preventDefault();
  isDrawing = true;
  const { x, y } = getPos(event);
  lastPoint = { x, y };
  smoothPos  = { x, y };
  drawCtx.beginPath();
  drawCtx.moveTo(x, y);
  drawCtx.strokeStyle = "#1a1a1a";   
  drawCtx.lineWidth   = 6;           
  drawCtx.lineCap     = "round";
  drawCtx.lineJoin    = "round";
}

// Smoothing factor: 0 = never moves, 1 = no smoothing.
const EMA_ALPHA = 0.45;

function drawSegment(raw) {
  smoothPos = {
    x: smoothPos.x + EMA_ALPHA * (raw.x - smoothPos.x),
    y: smoothPos.y + EMA_ALPHA * (raw.y - smoothPos.y),
  };
  const mid = {
    x: (lastPoint.x + smoothPos.x) / 2,
    y: (lastPoint.y + smoothPos.y) / 2,
  };
  drawCtx.beginPath();
  drawCtx.moveTo(lastPoint.x, lastPoint.y);
  drawCtx.quadraticCurveTo(smoothPos.x, smoothPos.y, mid.x, mid.y);
  drawCtx.stroke();
  lastPoint = mid;
}

function onPointerMove(event) {
  if (!isDrawing || !lastPoint) return;
  event.preventDefault();
  const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
  for (const e of events) {
    drawSegment(getPos(e));
  }
}

function onPointerUp(event) {
  if (!isDrawing) return;
  event.preventDefault();
  isDrawing = false;
  lastPoint = null;
  smoothPos = null;
  hasDrawn  = true;
  waitingHint.hidden  = true;
  drawingActions.hidden = false;
}

drawCanvas.addEventListener("mousedown",  onPointerDown);
drawCanvas.addEventListener("mousemove",  onPointerMove);
drawCanvas.addEventListener("mouseup",    onPointerUp);
drawCanvas.addEventListener("mouseleave", onPointerUp);

drawCanvas.addEventListener("touchstart",  onPointerDown, { passive: false });
drawCanvas.addEventListener("touchmove",   onPointerMove, { passive: false });
drawCanvas.addEventListener("touchend",    onPointerUp,   { passive: false });
drawCanvas.addEventListener("touchcancel", onPointerUp,   { passive: false });

resetButton.addEventListener("click", () => {
  clearDrawing();
  hasDrawn = false;
  drawingActions.hidden = true;
  waitingHint.hidden    = false;
  resultPanel.hidden    = true;
  document.querySelector("#result-label").textContent      = "";
  document.querySelector("#result-score").textContent      = "";
  document.querySelector("#result-confidence").textContent = "";
  document.querySelector("#result-warnings").replaceChildren();
  sessionStorage.removeItem("parkinson_result_drawing");
});

function percent(value) {
  if (value === null || value === undefined) return "-";
  return `${Math.round(Number(value) * 1000) / 10}%`;
}

function riskText(label) {
  const labels = { low: "Signal faible", moderate: "Signal modéré", elevated: "Signal élevé" };
  return labels[label] ?? "-";
}

function storeDrawingResult(result) {
  if (result.status !== "ok") return;
  sessionStorage.setItem("parkinson_result_drawing", JSON.stringify({
    ...result,
    saved_at: new Date().toISOString(),
  }));
}

function renderResult(result) {
  resultPanel.hidden = false;
  document.querySelector("#result-label").textContent      = riskText(result.label);
  document.querySelector("#result-score").textContent      = percent(result.score);
  document.querySelector("#result-confidence").textContent = percent(result.confidence);

  const warningList = document.querySelector("#result-warnings");
  warningList.replaceChildren();
  for (const warning of result.warnings ?? []) {
    const item = document.createElement("li");
    item.textContent = warning;
    warningList.appendChild(item);
  }
}

submitButton.addEventListener("click", async () => {
  submitButton.disabled    = true;
  submitButton.textContent = "Analyse…";
  resultPanel.hidden = true;

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width  = drawCanvas.width;
  tempCanvas.height = drawCanvas.height;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.fillStyle = "#ffffff";
  tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
  tempCtx.drawImage(drawCanvas, 0, 0);

  const image_b64 = tempCanvas.toDataURL("image/png");  

  try {
    const response = await fetch("/api/drawing/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_b64 }),
    });
    const result = await response.json();
    storeDrawingResult(result);
    renderResult(result);
  } catch (error) {
    renderResult({
      status: "error",
      warnings: [`Erreur réseau ou serveur : ${error}`],
    });
  } finally {
    submitButton.disabled    = false;
    submitButton.textContent = "Analyser le tracé";
  }
});

drawGuide();
