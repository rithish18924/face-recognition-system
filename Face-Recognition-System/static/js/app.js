/* static/js/app.js — Face Recognition Upload & Recognition UI */
"use strict";

const dropZone     = document.getElementById("drop-zone");
const fileInput    = document.getElementById("file-input");
const previewArea  = document.getElementById("preview-area");
const previewImg   = document.getElementById("preview-img");
const recognizeBtn = document.getElementById("recognize-btn");
const clearBtn     = document.getElementById("clear-btn");
const resultCard   = document.getElementById("result-card");
const spinner      = document.getElementById("spinner");
const tryAgainBtn  = document.getElementById("try-again-btn");
const uploadSection = document.getElementById("upload-section");

let currentFile = null;

// ── File selection ──────────────────────────────────────────────────────────
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) loadFile(file);
});

function loadFile(file) {
  const allowedTypes = ["image/jpeg","image/png","image/bmp","image/webp"];
  if (!allowedTypes.includes(file.type)) {
    alert("Unsupported format. Please use JPG, PNG, BMP, or WEBP.");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    alert("File too large. Maximum size is 8 MB.");
    return;
  }
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  dropZone.classList.add("hidden");
  previewArea.classList.remove("hidden");
  resultCard.classList.add("hidden");
}

// ── Drag & drop ─────────────────────────────────────────────────────────────
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});
dropZone.addEventListener("click", () => fileInput.click());

// ── Clear ───────────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", resetUI);
tryAgainBtn.addEventListener("click", resetUI);

function resetUI() {
  currentFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewArea.classList.add("hidden");
  dropZone.classList.remove("hidden");
  resultCard.classList.add("hidden");
}

// ── Recognize ───────────────────────────────────────────────────────────────
recognizeBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  spinner.classList.remove("hidden");
  recognizeBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("image", currentFile);

    const resp = await fetch("/predict", {
      method: "POST",
      body: formData,
      headers: { "Accept": "application/json" }
    });

    const contentType = resp.headers.get("content-type") || "";
    let data;

    if (contentType.includes("application/json")) {
      data = await resp.json();
    } else {
      const rawText = await resp.text();
      try {
        data = JSON.parse(rawText);
      } catch (_) {
        spinner.classList.add("hidden");
        recognizeBtn.disabled = false;
        showError("Server returned non-JSON response (HTTP " + resp.status + "). Check server logs.");
        return;
      }
    }

    spinner.classList.add("hidden");
    recognizeBtn.disabled = false;

    if (!resp.ok || data.error) {
      showError(data.error || "Server error (HTTP " + resp.status + ").");
      return;
    }

    renderResult(data);
  } catch (err) {
    spinner.classList.add("hidden");
    recognizeBtn.disabled = false;
    showError("Network error: " + err.message);
  }
});

// ── Render result ───────────────────────────────────────────────────────────
function renderResult(data) {
  const badge   = document.getElementById("status-badge");
  const identity= document.getElementById("detail-identity");
  const confBar = document.getElementById("conf-bar");
  const confVal = document.getElementById("detail-conf");
  const latency = document.getElementById("detail-latency");
  const top5    = document.getElementById("top5-list");
  const errMsg  = document.getElementById("detail-error");
  const resultImg = document.getElementById("result-img");

  // Fallback handler if image fails to load via URL: fallback to inline base64
  resultImg.onerror = function() {
    if (data.annotated_image && !resultImg.src.startsWith("data:")) {
      resultImg.src = "data:image/png;base64," + data.annotated_image;
    } else if (previewImg && previewImg.src && resultImg.src !== previewImg.src) {
      resultImg.src = previewImg.src;
    }
  };

  // Annotated image
  if (data.annotated_image_url) {
    resultImg.src = data.annotated_image_url;
  } else if (data.annotated_image) {
    if (data.annotated_image.startsWith("/") || data.annotated_image.startsWith("http") || data.annotated_image.startsWith("data:")) {
      resultImg.src = data.annotated_image;
    } else {
      resultImg.src = "data:image/png;base64," + data.annotated_image;
    }
  } else if (previewImg && previewImg.src) {
    resultImg.src = previewImg.src;
  }

  // Status badge
  badge.className = "status-badge";
  const statusMap = {
    SUCCESS:       { cls: "status-SUCCESS",   label: "✓ Known Identity" },
    UNKNOWN:       { cls: "status-UNKNOWN",   label: "✕ Unknown Person" },
    UNCERTAIN:     { cls: "status-UNCERTAIN", label: "? Uncertain Match" },
    NO_FACE:       { cls: "status-NO_FACE",   label: "No Face Detected" },
    MODEL_ERROR:   { cls: "status-ERROR",     label: "Processing Error" },
    INVALID_INPUT: { cls: "status-ERROR",     label: "Invalid Input" },
  };
  const s = statusMap[data.status] || { cls: "status-ERROR", label: data.status };
  badge.classList.add(s.cls);
  badge.textContent = s.label;

  // Identity & confidence
  identity.textContent = data.predicted_identity || "—";
  const conf = data.confidence ?? 0;
  confVal.textContent = (conf * 100).toFixed(1) + "%";
  confBar.style.width = (conf * 100).toFixed(0) + "%";

  // Latency
  latency.textContent = (data.inference_time_ms ?? 0).toFixed(0) + " ms";

  // Error message or multiple faces notice
  if (data.error_message) {
    errMsg.textContent = data.error_message;
    errMsg.classList.remove("hidden");
    errMsg.style.color = "";
  } else if (data.faces_note) {
    errMsg.textContent = "ℹ " + data.faces_note;
    errMsg.classList.remove("hidden");
    errMsg.style.color = "#f59e0b";
  } else {
    errMsg.classList.add("hidden");
  }

  // Top-5 predictions
  top5.innerHTML = "";
  const confs = data.all_confidences || {};
  const sorted = Object.entries(confs).sort((a, b) => b[1] - a[1]).slice(0, 5);
  for (const [name, prob] of sorted) {
    const pct = (prob * 100).toFixed(1);
    const item = document.createElement("div");
    item.className = "top5-item";
    item.innerHTML = `
      <span class="top5-name" title="${name}">${name}</span>
      <div class="top5-bar-bg">
        <div class="top5-bar" style="width:${pct}%"></div>
      </div>
      <span class="top5-val">${pct}%</span>`;
    top5.appendChild(item);
  }

  uploadSection.classList.add("hidden");
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(msg) {
  const errMsg = document.getElementById("detail-error");
  errMsg.textContent = msg;
  errMsg.classList.remove("hidden");
  const badge = document.getElementById("status-badge");
  badge.className = "status-badge status-ERROR";
  badge.textContent = "Error";
  const resultImg = document.getElementById("result-img");
  if (previewImg && previewImg.src) {
    resultImg.src = previewImg.src;
  }
  uploadSection.classList.add("hidden");
  resultCard.classList.remove("hidden");
}

// ── Supported Identities Panel & Search ─────────────────────────────────────
(function initIdentitiesPanel() {
  const toggleBtn = document.getElementById("identities-toggle");
  const content   = document.getElementById("identities-content");
  const chevron   = document.getElementById("identities-chevron");
  const searchInp = document.getElementById("identities-search");
  const grid      = document.getElementById("identities-grid");
  const countEl   = document.getElementById("identities-count");

  if (toggleBtn && content) {
    toggleBtn.addEventListener("click", () => {
      const isHidden = content.classList.contains("hidden");
      if (isHidden) {
        content.classList.remove("hidden");
        chevron.textContent = "▲ Hide list";
      } else {
        content.classList.add("hidden");
        chevron.textContent = "▼ View all enrolled persons";
      }
    });
  }

  // Live client-side search filter
  if (searchInp && grid) {
    searchInp.addEventListener("input", (e) => {
      const query = e.target.value.trim().toLowerCase();
      const pills = grid.querySelectorAll(".identity-pill");
      let visibleCount = 0;
      pills.forEach((pill) => {
        const name = pill.getAttribute("data-name") || pill.textContent.toLowerCase();
        if (name.includes(query)) {
          pill.style.display = "inline-flex";
          visibleCount++;
        } else {
          pill.style.display = "none";
        }
      });
      if (countEl && query.length > 0) {
        countEl.textContent = `${visibleCount} found`;
      } else if (countEl) {
        countEl.textContent = pills.length;
      }
    });
  }

  // Dynamic fetch if pills are empty (e.g. static html serving)
  if (grid && grid.children.length === 0) {
    fetch("/api/identities")
      .then((res) => res.json())
      .then((data) => {
        if (data && data.identities && data.identities.length > 0) {
          if (countEl) countEl.textContent = data.identities.length;
          grid.innerHTML = "";
          data.identities.forEach((id) => {
            const pill = document.createElement("span");
            pill.className = "identity-pill";
            pill.setAttribute("data-name", id.toLowerCase());
            pill.textContent = id;
            grid.appendChild(pill);
          });
        }
      })
      .catch((err) => console.warn("Could not fetch identities dynamically:", err));
  }
})();
