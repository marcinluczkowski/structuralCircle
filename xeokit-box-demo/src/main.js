import {
  Viewer,
  Mesh,
  ReadableGeometry,
  buildBoxGeometry,
  PhongMaterial,
  DirLight,
  WebIFCLoaderPlugin,
} from "@xeokit/xeokit-sdk";

const viewer = new Viewer({
  canvasId: "xeokit_canvas",
  transparent: true,
  dtxEnabled: true,
});

viewer.camera.eye = [4, 3, 5];
viewer.camera.look = [0, 0, 0];
viewer.camera.up = [0, 1, 0];

viewer.scene.clearLights();
new DirLight(viewer.scene, {
  id: "key",
  dir: [0.4, -0.85, -0.35],
  color: [1, 1, 1],
  intensity: 1.2,
  space: "world",
});

new DirLight(viewer.scene, {
  id: "fill",
  dir: [-0.6, -0.2, 0.75],
  color: [0.85, 0.9, 1],
  intensity: 0.45,
  space: "world",
});

const sharedGeometry = new ReadableGeometry(
  viewer.scene,
  buildBoxGeometry({
    center: [0, 0, 0],
    xSize: 1,
    ySize: 1,
    zSize: 1,
  }),
);

const spacingSlider = document.getElementById("spacing");
const countSlider = document.getElementById("count");
const resetButton = document.getElementById("reset-view");
const zoomExtentsButton = document.getElementById("zoom-extents");
const dropZoneCsv = document.getElementById("drop-zone-csv");
const dropZoneIfc = document.getElementById("drop-zone-ifc");
const fileInput = document.getElementById("file-input");
const folderInput = document.getElementById("folder-input");
const ifcInput = document.getElementById("ifc-input");
const pickFileBtn = document.getElementById("pick-file");
const pickFolderBtn = document.getElementById("pick-folder");
const pickIfcBtn = document.getElementById("pick-ifc");
const previewMaterialsBtn = document.getElementById("preview-materials");
const previewBuildingBtn = document.getElementById("preview-building");
const previewMatchingBtn = document.getElementById("preview-matching");
const dashboardContext = document.getElementById("dashboard-context");
const materialSummary = document.getElementById("material-summary");
const materialTableBody = document.getElementById("material-table-body");
const ifcSummary = document.getElementById("ifc-summary");
const ifcTableBody = document.getElementById("ifc-table-body");
const openDemandBankWindowBtn = document.getElementById("open-demand-bank-window");
const matchingPanel = document.getElementById("matching-panel");
const matchingStatus = document.getElementById("matching-status");
const matchingSummary = document.getElementById("matching-summary");
const matchingTableBody = document.getElementById("matching-table-body");
const matchingViz = document.getElementById("matching-viz");
const canvasEl = document.getElementById("xeokit_canvas");
const hoverInfo = document.getElementById("hover-info");

/** @type {"materials" | "building" | "matching"} */
let previewMode = "materials";

/** @type {Mesh[]} */
let demandMeshes = [];
/** @type {{ w: number; h: number; l: number }[]} */
let demandSizes = [];
/** @type {{ id: string; type: string; name: string; w: number; h: number; l: number; vol: number }[]} */
let ifcDemandSizes = [];
const materialInfoById = new Map();
const ifcInfoById = new Map();
let selectedMaterialIndex = -1;
let selectedIfcObjectId = null;
/** @type {{ demandId: string; demand: { id: string; type: string; name: string; w: number; h: number; l: number; vol: number }; candidates: { materialIndex: number; w: number; h: number; l: number; waste: number; volDiffPct: number; similarityScore: number; rotated: boolean }[] } | null} */
let activeMatch = null;
const bestCandidateByDemandId = new Map();

/** @type {{ minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number } | null} */
let sceneBounds = null;

/** Max IFC demand rows listed in tables (same cap as before). */
const IFC_DEMAND_TABLE_MAX_ROWS = 800;

/** @type {Window | null} */
let demandBankPopupWindow = null;

/** @type {WebIFCLoaderPlugin | null} */
let ifcLoader = null;
/** @type {Promise<WebIFCLoaderPlugin> | null} */
let ifcLoaderPromise = null;
/** Loaded IFC scene model (xeokit SceneModel). */
let ifcModel = null;

function destroyDemandMeshesOnly() {
  for (const mesh of demandMeshes) {
    mesh.destroy();
  }
  demandMeshes = [];
}

function detectDelimiter(line) {
  const comma = (line.match(/,/g) || []).length;
  const semicolon = (line.match(/;/g) || []).length;
  const tabs = (line.match(/\t/g) || []).length;
  if (semicolon >= comma && semicolon >= tabs && semicolon > 0) return ";";
  if (tabs > comma && tabs > semicolon) return "\t";
  return ",";
}

function parseCSVLine(line, delimiter) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (c === delimiter && !inQuotes) {
      out.push(cur.trim());
      cur = "";
      continue;
    }
    cur += c;
  }
  out.push(cur.trim());
  return out;
}

function normalizeHeader(h) {
  return h
    .toLowerCase()
    .replace(/^\ufeff/, "")
    .trim()
    .replace(/\s+/g, "_");
}

function pickColumnIndex(headers, names) {
  const norm = headers.map(normalizeHeader);
  for (const name of names) {
    const n = normalizeHeader(name);
    const i = norm.indexOf(n);
    if (i !== -1) return i;
  }

  for (const name of names) {
    const n = normalizeHeader(name);
    const i = norm.findIndex((h) => h.includes(n));
    if (i !== -1) return i;
  }

  return -1;
}

function parseDemandCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) {
    throw new Error("CSV needs a header row and at least one data row.");
  }

  const delimiter = detectDelimiter(lines[0]);
  const headers = parseCSVLine(lines[0], delimiter);
  const iW = pickColumnIndex(headers, ["width", "widht", "w", "breadth", "x"]);
  const iH = pickColumnIndex(headers, ["height", "heigh", "h", "y"]);
  const iL = pickColumnIndex(headers, ["length", "lenght", "l", "len", "depth", "z"]);
  const iQ = pickColumnIndex(headers, ["quantity", "qty", "count", "antal"]);

  if (iW === -1 || iH === -1 || iL === -1) {
    throw new Error("CSV must contain width, height, and length columns.");
  }

  const rows = [];
  for (let r = 1; r < lines.length; r++) {
    const cells = parseCSVLine(lines[r], delimiter);
    let w = parseFloat((cells[iW] ?? "").replace(",", "."));
    let h = parseFloat((cells[iH] ?? "").replace(",", "."));
    const l = parseFloat((cells[iL] ?? "").replace(",", "."));

    if (!Number.isFinite(w) || !Number.isFinite(h) || !Number.isFinite(l)) {
      continue;
    }

    if (w > 20) w /= 1000;
    if (h > 20) h /= 1000;

    if (w <= 0 || h <= 0 || l <= 0) {
      continue;
    }

    const qRaw = iQ === -1 ? "1" : cells[iQ] ?? "1";
    const quantity = Math.max(1, Math.min(20, parseInt(qRaw, 10) || 1));
    for (let i = 0; i < quantity; i++) {
      rows.push({ w, h, l });
    }
  }

  if (rows.length === 0) {
    throw new Error("No valid rows in the CSV data.");
  }

  return rows;
}

function readDirectoryEntriesAll(dir) {
  return new Promise((resolve, reject) => {
    const reader = dir.createReader();
    const acc = [];
    const read = () => {
      reader.readEntries(
        (entries) => {
          if (entries.length === 0) {
            resolve(acc);
            return;
          }
          acc.push(...entries);
          read();
        },
        reject,
      );
    };
    read();
  });
}

async function collectCsvFilesFromEntry(entry) {
  if (!entry) return [];

  if (entry.isFile) {
    return new Promise((resolve, reject) => {
      if (!entry.name.toLowerCase().endsWith(".csv")) {
        resolve([]);
        return;
      }
      entry.file(
        (file) => resolve([file]),
        reject,
      );
    });
  }

  if (entry.isDirectory) {
    const children = await readDirectoryEntriesAll(entry);
    const nested = await Promise.all(
      children.map((e) => collectCsvFilesFromEntry(e)),
    );
    return nested.flat();
  }

  return [];
}

function vecNormalize(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

const VIEW_DIR = vecNormalize([0.92, 0.52, 0.85]);

function applyCameraFromBounds(distFactor, extraPadding) {
  if (!sceneBounds) return;

  const { minX, maxX, minY, maxY, minZ, maxZ } = sceneBounds;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  const diagonal = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
  const dist = diagonal * distFactor + extraPadding;

  viewer.camera.look = [cx, cy, cz];
  viewer.camera.eye = [
    cx + VIEW_DIR[0] * dist,
    cy + VIEW_DIR[1] * dist,
    cz + VIEW_DIR[2] * dist,
  ];
  viewer.camera.up = [0, 1, 0];
}

function fitZoomExtents() {
  applyCameraFromBounds(0.48, 0.15);
}

function resetViewCamera() {
  applyCameraFromBounds(0.72, 0.45);
}

function jumpCameraToIfc() {
  const flight = viewer.cameraFlight;
  if (ifcModel && flight && typeof flight.jumpTo === "function") {
    flight.jumpTo(ifcModel);
  }
}

function hslToRgb(h, s, l) {
  if (s === 0) return [l, l, l];
  const hue2rgb = (p, q, t) => {
    let tt = t;
    if (tt < 0) tt += 1;
    if (tt > 1) tt -= 1;
    if (tt < 1 / 6) return p + (q - p) * 6 * tt;
    if (tt < 1 / 2) return q;
    if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [
    hue2rgb(p, q, h + 1 / 3),
    hue2rgb(p, q, h),
    hue2rgb(p, q, h - 1 / 3),
  ];
}

function boxColor(index) {
  return hslToRgb((index * 0.12) % 1, 0.5, 0.58);
}

function selectedCount() {
  if (demandSizes.length === 0) return 0;
  const raw = parseInt(countSlider.value, 10);
  return Math.max(1, Math.min(raw, demandSizes.length));
}

function fmtM(value) {
  return `${value.toFixed(3)} m`;
}

function fmtM3(value) {
  return `${value.toFixed(3)} m³`;
}

function clearEl(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function appendSummaryRow(dl, label, value) {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  dl.append(dt, dd);
}

function clearMaterialHighlight() {
  if (selectedMaterialIndex >= 0 && selectedMaterialIndex < demandMeshes.length) {
    demandMeshes[selectedMaterialIndex].highlighted = false;
  }
  selectedMaterialIndex = -1;
}

function hideHoverInfo() {
  if (!hoverInfo) return;
  hoverInfo.hidden = true;
}

function positionHoverInfo(clientX, clientY) {
  if (!hoverInfo || !canvasEl) return;
  const rect = canvasEl.getBoundingClientRect();
  const x = Math.min(rect.width - 12, Math.max(8, clientX - rect.left + 12));
  const y = Math.min(rect.height - 12, Math.max(8, clientY - rect.top + 12));
  hoverInfo.style.left = `${x}px`;
  hoverInfo.style.top = `${y}px`;
}

function showHoverInfoText(clientX, clientY, text) {
  if (!hoverInfo) return;
  positionHoverInfo(clientX, clientY);
  hoverInfo.textContent = text;
  hoverInfo.hidden = false;
}

function showHoverInfoNode(clientX, clientY, node) {
  if (!hoverInfo) return;
  positionHoverInfo(clientX, clientY);
  hoverInfo.replaceChildren(node);
  hoverInfo.hidden = false;
}

function clearIfcHighlight() {
  if (selectedIfcObjectId) {
    const prev = viewer.scene.objects?.[selectedIfcObjectId];
    if (prev) prev.highlighted = false;
  }
  selectedIfcObjectId = null;
}

function flyToAABB(aabb) {
  if (!aabb || aabb.length < 6) return;
  const flight = viewer.cameraFlight;
  if (flight && typeof flight.flyTo === "function") {
    flight.flyTo(aabb);
  }
}

function materialFitsDemand(material, demand) {
  const direct = material.w >= demand.w && material.h >= demand.h;
  const rotated = material.w >= demand.h && material.h >= demand.w;
  if (!direct && !rotated) return { fits: false, rotated: false };
  if (material.l < demand.l) return { fits: false, rotated: false };
  return { fits: true, rotated: !direct && rotated };
}

function buildCandidatesForDemand(demand) {
  /** @type {{ materialIndex: number; w: number; h: number; l: number; waste: number; volDiffPct: number; similarityScore: number; rotated: boolean }[]} */
  const candidates = [];
  for (let i = 0; i < demandSizes.length; i++) {
    const material = demandSizes[i];
    const fit = materialFitsDemand(material, demand);
    if (!fit.fits) continue;
    const supplyVol = material.w * material.h * material.l;
    const waste = supplyVol - demand.vol;
    const volDiffPct = demand.vol > 0 ? (100 * waste) / demand.vol : 0;
    const similarityScore = Math.max(0, 100 - volDiffPct);
    candidates.push({
      materialIndex: i,
      w: material.w,
      h: material.h,
      l: material.l,
      waste,
      volDiffPct,
      similarityScore,
      rotated: fit.rotated,
    });
  }
  candidates.sort((a, b) => {
    if (b.similarityScore !== a.similarityScore) {
      return b.similarityScore - a.similarityScore;
    }
    return a.waste - b.waste;
  });
  return candidates;
}

function getBestCandidateForDemand(demand) {
  const cached = bestCandidateByDemandId.get(demand.id);
  if (cached !== undefined) return cached;
  const best = buildCandidatesForDemand(demand)[0] || null;
  bestCandidateByDemandId.set(demand.id, best);
  return best;
}

function createKeyValueRow(key, value) {
  const k = document.createElement("span");
  k.textContent = key;
  const v = document.createElement("span");
  v.textContent = value;
  return [k, v];
}

function createHoverMatchCard(demand, best) {
  const card = document.createElement("div");

  const title = document.createElement("div");
  title.className = "hover-info__title";
  title.textContent = demand.type;
  card.appendChild(title);

  const meta = document.createElement("div");
  meta.className = "hover-info__meta";
  meta.textContent = demand.name;
  card.appendChild(meta);

  const kv = document.createElement("div");
  kv.className = "hover-info__kv";
  kv.append(...createKeyValueRow("Demand ID", demand.id));
  kv.append(
    ...createKeyValueRow(
      "Demand",
      `${demand.w.toFixed(2)}×${demand.h.toFixed(2)}×${demand.l.toFixed(2)} m`,
    ),
  );
  kv.append(...createKeyValueRow("Demand vol", `${demand.vol.toFixed(3)} m³`));
  card.appendChild(kv);

  const divider = document.createElement("div");
  divider.className = "hover-info__divider";
  card.appendChild(divider);

  if (!best) {
    const none = document.createElement("div");
    none.textContent = "No suitable supply element found.";
    card.appendChild(none);
    return card;
  }

  const supplyVol = best.w * best.h * best.l;
  const supplyW = best.rotated ? best.h : best.w;
  const supplyH = best.rotated ? best.w : best.h;

  const kv2 = document.createElement("div");
  kv2.className = "hover-info__kv";
  kv2.append(...createKeyValueRow("Best supply", `#${best.materialIndex + 1}`));
  kv2.append(
    ...createKeyValueRow(
      "Supply",
      `${supplyW.toFixed(2)}×${supplyH.toFixed(2)}×${best.l.toFixed(2)} m`,
    ),
  );
  kv2.append(...createKeyValueRow("Supply vol", `${supplyVol.toFixed(3)} m³`));
  kv2.append(...createKeyValueRow("Δvol", `${best.volDiffPct.toFixed(1)}%`));
  kv2.append(...createKeyValueRow("Score", `${best.similarityScore.toFixed(1)}%`));
  card.appendChild(kv2);

  const geo = document.createElement("div");
  geo.className = "hover-geo";
  const maxW = Math.max(demand.w, supplyW) || 1;
  const maxH = Math.max(demand.h, supplyH) || 1;
  const sizeW = 104;
  const sizeH = 56;

  const supplyRect = document.createElement("div");
  supplyRect.className = "hover-geo__supply";
  supplyRect.style.width = `${Math.max(10, (supplyW / maxW) * sizeW)}px`;
  supplyRect.style.height = `${Math.max(8, (supplyH / maxH) * sizeH)}px`;

  const demandRect = document.createElement("div");
  demandRect.className = "hover-geo__demand";
  demandRect.style.width = `${Math.max(10, (demand.w / maxW) * sizeW)}px`;
  demandRect.style.height = `${Math.max(8, (demand.h / maxH) * sizeH)}px`;

  geo.appendChild(supplyRect);
  geo.appendChild(demandRect);
  card.appendChild(geo);

  const legend = document.createElement("div");
  legend.className = "hover-geo__legend";
  legend.textContent = "Blue=supply cross-section, Yellow=demand";
  card.appendChild(legend);

  return card;
}

function countMatchableDemandObjects() {
  if (demandSizes.length === 0 || ifcDemandSizes.length === 0) return 0;
  let count = 0;
  for (const demand of ifcDemandSizes) {
    let found = false;
    for (let i = 0; i < demandSizes.length; i++) {
      if (materialFitsDemand(demandSizes[i], demand).fits) {
        found = true;
        break;
      }
    }
    if (found) count += 1;
  }
  return count;
}

function selectDemandForMatching(objectId) {
  const demand = ifcDemandSizes.find((row) => row.id === objectId);
  if (!demand) {
    setStatus("Clicked object is not in demand bank.");
    return;
  }
  const entity = viewer.scene.objects?.[objectId];
  if (!entity) {
    setStatus(`IFC object not found in scene: ${objectId}`);
    return;
  }

  clearMaterialHighlight();
  clearIfcHighlight();
  selectedIfcObjectId = objectId;
  entity.highlighted = true;
  flyToAABB(entity.aabb);

  activeMatch = {
    demandId: objectId,
    demand,
    candidates: buildCandidatesForDemand(demand),
  };
  renderMatchingPanel();
}

function buildIfcDemandFromModel() {
  if (!ifcModel) return [];

  const metaModel = viewer.metaScene.metaModels[ifcModel.id];
  if (!metaModel) return [];

  const objectIds = new Set();
  for (const root of metaModel.rootMetaObjects || []) {
    for (const id of viewer.metaScene.getObjectIDsInSubtree(root.id)) {
      objectIds.add(id);
    }
  }

  if (objectIds.size === 0) {
    for (const id of Object.keys(metaModel.metaObjects || {})) {
      objectIds.add(id);
    }
  }

  const skipTypes = new Set([
    "IfcProject",
    "IfcSite",
    "IfcBuilding",
    "IfcBuildingStorey",
    "IfcSpace",
  ]);

  ifcInfoById.clear();
  bestCandidateByDemandId.clear();

  /** @type {{ id: string; type: string; name: string; w: number; h: number; l: number; vol: number }[]} */
  const rows = [];
  for (const id of objectIds) {
    const meta = metaModel.metaObjects[id];
    const type = meta?.type || "IfcElement";
    if (skipTypes.has(type)) continue;
    if (meta?.children?.length) continue;

    const entity = viewer.scene.objects?.[id];
    if (!entity) continue;

    const aabb = entity.aabb;
    if (!aabb || aabb.length < 6) continue;

    const w = aabb[3] - aabb[0];
    const h = aabb[4] - aabb[1];
    const l = aabb[5] - aabb[2];
    if (![w, h, l].every(Number.isFinite)) continue;
    if (w <= 0 || h <= 0 || l <= 0) continue;

    const row = {
      id,
      type,
      name: meta?.name || id,
      w,
      h,
      l,
      vol: w * h * l,
    };
    rows.push(row);
    ifcInfoById.set(id, row);
  }

  rows.sort((a, b) => b.vol - a.vol);
  return rows;
}

/**
 * @returns
 *   | { type: "status"; status: string }
 *   | { type: "rows"; summary: [string, string][]; rows: typeof ifcDemandSizes }
 */
function getIfcDemandBankPresentation() {
  if (!ifcModel) {
    return { type: "status", status: "no ifc loaded" };
  }
  if (ifcDemandSizes.length === 0) {
    return { type: "status", status: "no measurable objects found" };
  }

  const shown = Math.min(IFC_DEMAND_TABLE_MAX_ROWS, ifcDemandSizes.length);
  const slice = ifcDemandSizes.slice(0, shown);

  let totalVolume = 0;
  let sumLength = 0;
  let minL = Infinity;
  let maxL = -Infinity;
  let minVol = Infinity;
  let maxVol = -Infinity;
  for (const row of slice) {
    totalVolume += row.vol;
    sumLength += row.l;
    minL = Math.min(minL, row.l);
    maxL = Math.max(maxL, row.l);
    minVol = Math.min(minVol, row.vol);
    maxVol = Math.max(maxVol, row.vol);
  }

  const summary = /** @type {[string, string][]} */ ([
    ["elements (visible)", `${shown} / ${ifcDemandSizes.length}`],
    ["total volume", fmtM3(totalVolume)],
    ["Σ length (L)", fmtM(sumLength)],
    ["length L (min → max)", `${fmtM(minL)} → ${fmtM(maxL)}`],
    ["single volume (min → max)", `${fmtM3(minVol)} → ${fmtM3(maxVol)}`],
    ["row span (X incl. gaps)", "—"],
  ]);

  return { type: "rows", summary, rows: slice };
}

/**
 * @param {HTMLElement} summaryEl
 * @param {HTMLTableSectionElement} tbodyEl
 * @param {ReturnType<typeof getIfcDemandBankPresentation>} presentation
 */
function renderDemandBankIntoDom(summaryEl, tbodyEl, presentation) {
  clearEl(summaryEl);
  tbodyEl.replaceChildren();

  if (presentation.type === "status") {
    appendSummaryRow(summaryEl, "status", presentation.status);
    return;
  }

  for (const [label, value] of presentation.summary) {
    appendSummaryRow(summaryEl, label, value);
  }

  for (let i = 0; i < presentation.rows.length; i++) {
    const { id, type, name, w, h, l, vol } = presentation.rows[i];
    const tr = document.createElement("tr");
    tr.dataset.objectId = id;
    tr.title = `${type} · ${name}`;
    for (const text of [
      String(i + 1),
      w.toFixed(2),
      h.toFixed(2),
      l.toFixed(2),
      vol.toFixed(3),
    ]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    tbodyEl.appendChild(tr);
  }
}

function syncDemandBankPopup() {
  if (!demandBankPopupWindow || demandBankPopupWindow.closed) {
    demandBankPopupWindow = null;
    return;
  }
  const doc = demandBankPopupWindow.document;
  const summaryEl = doc.getElementById("demand-bank-summary");
  const tbodyEl = doc.getElementById("demand-bank-tbody");
  if (!summaryEl || !tbodyEl) return;
  renderDemandBankIntoDom(summaryEl, tbodyEl, getIfcDemandBankPresentation());
}

function openDemandBankWindow() {
  if (demandBankPopupWindow && !demandBankPopupWindow.closed) {
    demandBankPopupWindow.focus();
    syncDemandBankPopup();
    return;
  }

  demandBankPopupWindow = window.open(
    "",
    "xeokitDemandBank",
    "width=560,height=720,scrollbars=yes,resizable=yes",
  );
  if (!demandBankPopupWindow) {
    setStatus("Could not open window (popup blocked?).");
    return;
  }

  const popupCss = `
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; }
    body {
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      color: #e8eef5;
      background: #0f1419;
      padding: 1rem 1.1rem 1.25rem;
    }
    .title {
      margin: 0 0 0.35rem;
      font-size: 0.95rem;
      font-weight: 600;
      text-transform: lowercase;
      letter-spacing: 0.02em;
      color: #d4e4f4;
    }
    .hint {
      margin: 0 0 0.75rem;
      font-size: 0.72rem;
      line-height: 1.35;
      color: #8aa8c4;
    }
    .summary {
      margin: 0 0 0.75rem;
      display: grid;
      gap: 0.45rem 0.75rem;
      grid-template-columns: auto 1fr;
      font-size: 0.78rem;
    }
    .summary dt { margin: 0; color: #8aa8c4; text-transform: lowercase; }
    .summary dd {
      margin: 0;
      font-variant-numeric: tabular-nums;
      color: #e8eef5;
      text-align: right;
    }
    .table-head {
      font-size: 0.72rem;
      text-transform: lowercase;
      color: #8aa8c4;
      margin-bottom: 0.35rem;
    }
    .table-wrap {
      max-height: calc(100vh - 12rem);
      overflow: auto;
      border: 1px solid #2a3a50;
      border-radius: 6px;
      background: #0f1419;
    }
    .tbl {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.72rem;
      font-variant-numeric: tabular-nums;
    }
    .tbl th, .tbl td {
      padding: 0.35rem 0.45rem;
      text-align: right;
      border-bottom: 1px solid #243044;
    }
    .tbl th:first-child, .tbl td:first-child {
      text-align: left;
      position: sticky;
      left: 0;
      background: #0f1419;
      box-shadow: 1px 0 0 #243044;
    }
    .tbl thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #1a2332;
      color: #9ecae8;
      font-weight: 600;
      text-transform: lowercase;
    }
    .tbl tbody tr:hover td { background: #151d2a; }
    .tbl tbody tr { cursor: pointer; }
  `;

  const w = demandBankPopupWindow;
  const d = w.document;
  d.open();
  d.write(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Demand bank (ifc)</title>
<style>${popupCss}</style>
</head>
<body>
  <h1 class="title">demand bank (ifc)</h1>
  <p class="hint">same summary and columns as material bank · click a row to select that object in the main viewer</p>
  <dl id="demand-bank-summary" class="summary"></dl>
  <div class="table-head">visible elements · W, H, L in m · vol in m³</div>
  <div class="table-wrap">
    <table class="tbl">
      <thead><tr><th>#</th><th>W</th><th>H</th><th>L</th><th>vol</th></tr></thead>
      <tbody id="demand-bank-tbody"></tbody>
    </table>
  </div>
  <script>
  (function () {
    var tbody = document.getElementById("demand-bank-tbody");
    if (!tbody) return;
    tbody.addEventListener("click", function (e) {
      var tr = e.target.closest("tr");
      var id = tr && tr.getAttribute("data-object-id");
      if (id && window.opener && typeof window.opener.__xeokitActivateDemandRow === "function") {
        window.opener.__xeokitActivateDemandRow(id);
      }
    });
  })();
  <\/script>
</body>
</html>`);
  d.close();
  syncDemandBankPopup();
}

function onIfcDemandRowActivate(objectId) {
  if (!objectId) return;

  if (previewMode === "matching") {
    selectDemandForMatching(objectId);
    return;
  }

  if (ifcModel) setPreviewMode("building");
  clearMaterialHighlight();
  clearIfcHighlight();

  const entity = viewer.scene.objects?.[objectId];
  if (!entity) {
    setStatus(`IFC object not found in scene: ${objectId}`);
    return;
  }
  selectedIfcObjectId = objectId;
  entity.highlighted = true;
  flyToAABB(entity.aabb);
}

window.__xeokitActivateDemandRow = onIfcDemandRowActivate;

function renderMaterialTable() {
  if (!materialSummary || !materialTableBody) return;
  clearEl(materialSummary);
  materialTableBody.replaceChildren();

  if (demandSizes.length === 0) {
    appendSummaryRow(materialSummary, "status", "no csv loaded");
    return;
  }

  const spacing = parseFloat(spacingSlider.value);
  const visible = selectedCount();
  const slice = demandSizes.slice(0, visible);

  let totalVolume = 0;
  let sumLength = 0;
  let minL = Infinity;
  let maxL = -Infinity;
  let minVol = Infinity;
  let maxVol = -Infinity;

  for (const { w, h, l } of slice) {
    const vol = w * h * l;
    totalVolume += vol;
    sumLength += l;
    minL = Math.min(minL, l);
    maxL = Math.max(maxL, l);
    minVol = Math.min(minVol, vol);
    maxVol = Math.max(maxVol, vol);
  }

  const rowSpanX =
    slice.reduce((acc, { w }) => acc + w, 0) + spacing * Math.max(0, slice.length - 1);

  appendSummaryRow(
    materialSummary,
    "elements (visible)",
    `${visible} / ${demandSizes.length}`,
  );
  appendSummaryRow(materialSummary, "total volume", fmtM3(totalVolume));
  appendSummaryRow(materialSummary, "Σ length (L)", fmtM(sumLength));
  appendSummaryRow(
    materialSummary,
    "length L (min → max)",
    `${fmtM(minL)} → ${fmtM(maxL)}`,
  );
  appendSummaryRow(
    materialSummary,
    "single volume (min → max)",
    `${fmtM3(minVol)} → ${fmtM3(maxVol)}`,
  );
  appendSummaryRow(materialSummary, "row span (X incl. gaps)", fmtM(rowSpanX));

  for (let i = 0; i < slice.length; i++) {
    const { w, h, l } = slice[i];
    const vol = w * h * l;
    const tr = document.createElement("tr");
    tr.dataset.rowIndex = String(i);
    for (const text of [
      String(i + 1),
      w.toFixed(2),
      h.toFixed(2),
      l.toFixed(2),
      vol.toFixed(3),
    ]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    materialTableBody.appendChild(tr);
  }
}

function renderIfcTable() {
  if (!ifcSummary || !ifcTableBody) return;
  renderDemandBankIntoDom(ifcSummary, ifcTableBody, getIfcDemandBankPresentation());
  syncDemandBankPopup();
}

function renderDashboard() {
  if (!dashboardContext) return;

  if (previewMode === "building") {
    dashboardContext.textContent = ifcModel
      ? "Canvas: building (IFC). Material table shows supply bank slice; IFC table shows demand bank objects from the loaded model."
      : "Canvas: building (IFC). Load an IFC file to populate the demand bank table.";
  } else if (previewMode === "matching") {
    dashboardContext.textContent =
      "Canvas: matching mode. Click IFC elements in the viewer to see all usable material-bank candidates on the right.";
  } else {
    dashboardContext.textContent =
      "Canvas: material bank. Left table shows material elements; second table shows IFC demand bank when loaded.";
  }

  renderMaterialTable();
  renderIfcTable();
  renderMatchingPanel();
}

function renderMatchingPanel() {
  if (!matchingPanel || !matchingStatus || !matchingSummary || !matchingTableBody || !matchingViz) return;

  matchingPanel.classList.toggle("matching-panel--active", previewMode === "matching");
  clearEl(matchingSummary);
  matchingTableBody.replaceChildren();
  matchingViz.replaceChildren();

  const matchableCount = countMatchableDemandObjects();
  const demandCount = ifcDemandSizes.length;
  const coverage = demandCount > 0 ? (100 * matchableCount) / demandCount : 0;
  appendSummaryRow(matchingSummary, "material elements", String(demandSizes.length));
  appendSummaryRow(matchingSummary, "demand elements", String(demandCount));
  appendSummaryRow(matchingSummary, "matchable demand", `${matchableCount} (${coverage.toFixed(1)}%)`);

  if (!ifcModel) {
    matchingStatus.textContent = "Load an IFC file to start matching.";
    return;
  }
  if (demandSizes.length === 0) {
    matchingStatus.textContent = "Load material bank CSV to calculate candidates.";
    return;
  }
  if (!activeMatch) {
    matchingStatus.textContent =
      "Click an IFC element in the viewer or in the demand bank table to show possible material-bank elements.";
    return;
  }

  const { demand, candidates } = activeMatch;
  matchingStatus.textContent =
    `${demand.type} · ${demand.name} · target ${demand.w.toFixed(2)}×${demand.h.toFixed(2)}×${demand.l.toFixed(2)} m`;
  appendSummaryRow(
    matchingSummary,
    "selected demand",
    `${demand.w.toFixed(2)} × ${demand.h.toFixed(2)} × ${demand.l.toFixed(2)} m`,
  );
  appendSummaryRow(matchingSummary, "candidates", String(candidates.length));

  const shown = Math.min(candidates.length, 300);
  const vizShown = Math.min(candidates.length, 60);
  appendSummaryRow(matchingSummary, "shown", `${shown} / ${candidates.length}`);
  appendSummaryRow(matchingSummary, "sorted by", "score (highest first)");

  const maxLength = Math.max(
    demand.l,
    ...candidates.slice(0, vizShown).map((row) => row.l),
  );
  for (let i = 0; i < vizShown; i++) {
    const row = candidates[i];
    const line = document.createElement("div");
    line.className = "match-line";

    const rank = document.createElement("div");
    rank.className = "match-line__rank";
    rank.textContent = `#${i + 1}`;

    const track = document.createElement("div");
    track.className = "match-line__track";

    const bar = document.createElement("div");
    bar.className = "match-line__bar";
    const widthPct = maxLength > 0 ? (100 * row.l) / maxLength : 0;
    bar.style.width = `${Math.max(3, widthPct)}%`;
    bar.style.opacity = `${0.45 + 0.55 * Math.max(0, Math.min(1, row.similarityScore / 100))}`;

    const demandMarker = document.createElement("div");
    demandMarker.className = "match-line__demand";

    track.appendChild(bar);
    track.appendChild(demandMarker);
    line.appendChild(rank);
    line.appendChild(track);
    line.title = `mat#${row.materialIndex + 1} · L=${row.l.toFixed(2)}m · score ${row.similarityScore.toFixed(1)}%`;
    matchingViz.appendChild(line);
  }

  for (let i = 0; i < shown; i++) {
    const row = candidates[i];
    const tr = document.createElement("tr");
    tr.title = row.rotated ? "width/height matched via 90-degree swap" : "direct width/height match";
    for (const text of [
      String(i + 1),
      String(row.materialIndex + 1),
      row.w.toFixed(2),
      row.h.toFixed(2),
      row.l.toFixed(2),
      `${row.similarityScore.toFixed(1)}%`,
      `${row.volDiffPct.toFixed(1)}%`,
      row.waste.toFixed(3),
    ]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    matchingTableBody.appendChild(tr);
  }
}

function syncSceneVisibility() {
  const showBoxes = previewMode === "materials";
  for (const mesh of demandMeshes) {
    mesh.visible = showBoxes;
  }
  if (ifcModel) {
    ifcModel.visible = previewMode === "building" || previewMode === "matching";
  }
}

/**
 * @param {"materials" | "building" | "matching"} mode
 */
function setPreviewMode(mode) {
  if ((mode === "building" || mode === "matching") && !ifcModel) {
    setStatus("Load an IFC file first, then switch to building preview.");
    return;
  }

  previewMode = mode;
  hideHoverInfo();
  previewMaterialsBtn.classList.toggle("toggle-btn--active", mode === "materials");
  previewBuildingBtn.classList.toggle("toggle-btn--active", mode === "building");
  previewMatchingBtn.classList.toggle("toggle-btn--active", mode === "matching");

  syncSceneVisibility();

  if (mode === "materials") {
    if (sceneBounds) fitZoomExtents();
  } else {
    jumpCameraToIfc();
  }

  renderDashboard();
}

function rebuildLayout() {
  const spacing = parseFloat(spacingSlider.value);
  const visibleCount = selectedCount();

  destroyDemandMeshesOnly();
  materialInfoById.clear();
  bestCandidateByDemandId.clear();
  selectedMaterialIndex = -1;
  activeMatch = null;
  sceneBounds = null;

  if (demandSizes.length === 0 || visibleCount === 0) {
    setStatus("No CSV data loaded.");
    syncSceneVisibility();
    renderDashboard();
    return;
  }

  let penX = 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;

  demandSizes.slice(0, visibleCount).forEach(({ w, h, l }, index) => {
    const cx = penX + w / 2;
    const cz = l / 2;
    const cy = h / 2;

    const mesh = new Mesh(viewer.scene, {
      geometry: sharedGeometry,
      material: new PhongMaterial(viewer.scene, {
        diffuse: boxColor(index),
        specular: [0.12, 0.12, 0.18],
        shininess: 40,
      }),
      position: [cx, cy, cz],
      scale: [w / 2, h / 2, l / 2],
    });
    demandMeshes.push(mesh);
    materialInfoById.set(mesh.id, { w, h, l, vol: w * h * l, rowIndex: index + 1 });

    minX = Math.min(minX, cx - w / 2);
    maxX = Math.max(maxX, cx + w / 2);
    minY = Math.min(minY, 0);
    maxY = Math.max(maxY, h);
    minZ = Math.min(minZ, cz - l / 2);
    maxZ = Math.max(maxZ, cz + l / 2);

    penX += w + spacing;
  });

  sceneBounds = { minX, maxX, minY, maxY, minZ, maxZ };
  syncSceneVisibility();

  if (previewMode === "materials") {
    fitZoomExtents();
  }

  setStatus(
    `${visibleCount}/${demandSizes.length} boxes in one row · spacing ${spacing.toFixed(1)} m`,
  );

  renderDashboard();
}

function setStatus(msg) {
  const el = document.getElementById("status");
  if (el) el.textContent = msg;
}

async function loadFromCsvFiles(files) {
  const csvFiles = files.filter((f) => f.name.toLowerCase().endsWith(".csv"));
  if (csvFiles.length === 0) {
    setStatus("No .csv files found.");
    return;
  }

  const merged = [];
  let parsedFiles = 0;

  for (const f of csvFiles) {
    try {
      const text = await f.text();
      merged.push(...parseDemandCsv(text));
      parsedFiles += 1;
    } catch {
      /* skip invalid */
    }
  }

  if (merged.length === 0) {
    setStatus("Could not parse any CSV (check columns).");
    return;
  }

  demandSizes = merged;
  bestCandidateByDemandId.clear();
  countSlider.min = "1";
  countSlider.max = String(demandSizes.length);
  countSlider.value = String(Math.min(demandSizes.length, 60));
  rebuildLayout();
  setStatus(
    `${demandSizes.length} rows · ${parsedFiles}/${csvFiles.length} csv file(s)`,
  );
}

async function ensureIfcLoader() {
  if (ifcLoader) return ifcLoader;
  if (!ifcLoaderPromise) {
    ifcLoaderPromise = (async () => {
      const WebIFC = await import(
        /* @vite-ignore */
        "https://cdn.jsdelivr.net/npm/web-ifc@0.0.51/web-ifc-api.js",
      );
      const IfcAPI = new WebIFC.IfcAPI();
      IfcAPI.SetWasmPath("https://cdn.jsdelivr.net/npm/web-ifc@0.0.51/");
      await IfcAPI.Init();
      ifcLoader = new WebIFCLoaderPlugin(viewer, {
        WebIFC,
        IfcAPI,
      });
      return ifcLoader;
    })().catch((error) => {
      ifcLoaderPromise = null;
      throw error;
    });
  }
  return ifcLoaderPromise;
}

function toErrorText(error) {
  if (error instanceof Error) return error.message;
  return String(error ?? "unknown error");
}

async function loadIfcFromFile(file) {
  if (!file?.name?.toLowerCase().endsWith(".ifc")) {
    setStatus("Please choose a .ifc file.");
    return;
  }

  try {
    setStatus("Loading IFC (first load may take a while)...");
    const loader = await ensureIfcLoader();

    if (ifcModel) {
      ifcModel.destroy();
      ifcModel = null;
      ifcDemandSizes = [];
      ifcInfoById.clear();
      bestCandidateByDemandId.clear();
      activeMatch = null;
    }

    const ifcArrayBuffer = await file.arrayBuffer();
    ifcModel = loader.load({
      id: "building-ifc",
      ifc: ifcArrayBuffer,
      excludeTypes: ["IfcSpace"],
      edges: true,
    });

    ifcModel.on("error", (err) => {
      const details = toErrorText(err);
      ifcDemandSizes = [];
      ifcInfoById.clear();
      bestCandidateByDemandId.clear();
      activeMatch = null;
      setStatus(`IFC failed: ${details}`);
      console.error("IFC load error:", err);
      renderDashboard();
    });

    ifcModel.on("loaded", () => {
      ifcDemandSizes = buildIfcDemandFromModel();
      activeMatch = null;
      syncSceneVisibility();
      setStatus(`IFC loaded: ${file.name} · ${ifcDemandSizes.length} demand objects`);
      if (previewMode === "building") {
        jumpCameraToIfc();
      }
      renderDashboard();
    });
  } catch (e) {
    const details = toErrorText(e);
    ifcDemandSizes = [];
    ifcInfoById.clear();
    bestCandidateByDemandId.clear();
    activeMatch = null;
    setStatus(`IFC load failed: ${details}`);
    console.error("IFC setup/load failed:", e);
    renderDashboard();
  }
}

async function loadBundledCsv() {
  try {
    const res = await fetch("/materialliste-biennale.csv");
    if (!res.ok) throw new Error(`Failed to load CSV (${res.status})`);
    const text = await res.text();

    demandSizes = parseDemandCsv(text);
    bestCandidateByDemandId.clear();
    countSlider.min = "1";
    countSlider.max = String(demandSizes.length);
    countSlider.value = String(Math.min(demandSizes.length, 40));
    rebuildLayout();
  } catch {
    setStatus("Drop CSV for materials, or add materialliste-biennale.csv to /public.");
    renderDashboard();
  }
}

pickFileBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

pickFolderBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  folderInput.click();
});

pickIfcBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  ifcInput.click();
});

previewMaterialsBtn.addEventListener("click", () => setPreviewMode("materials"));
previewBuildingBtn.addEventListener("click", () => setPreviewMode("building"));
previewMatchingBtn.addEventListener("click", () => setPreviewMode("matching"));

fileInput.addEventListener("change", () => {
  const list = fileInput.files ? Array.from(fileInput.files) : [];
  fileInput.value = "";
  if (list.length) loadFromCsvFiles(list);
});

folderInput.addEventListener("change", () => {
  const list = folderInput.files ? Array.from(folderInput.files) : [];
  folderInput.value = "";
  if (list.length) loadFromCsvFiles(list);
});

ifcInput.addEventListener("change", () => {
  const f = ifcInput.files?.[0];
  ifcInput.value = "";
  if (f) loadIfcFromFile(f);
});

canvasEl?.addEventListener("click", (event) => {
  if (previewMode !== "matching" || !ifcModel) return;
  const rect = canvasEl.getBoundingClientRect();
  const pickResult = viewer.scene.pick({
    pickSurface: true,
    canvasPos: [event.clientX - rect.left, event.clientY - rect.top],
  });
  const objectId = pickResult?.entity?.id;
  if (!objectId) return;
  selectDemandForMatching(objectId);
});

canvasEl?.addEventListener("mousemove", (event) => {
  const rect = canvasEl.getBoundingClientRect();
  const pickResult = viewer.scene.pick({
    pickSurface: false,
    canvasPos: [event.clientX - rect.left, event.clientY - rect.top],
  });
  const objectId = pickResult?.entity?.id;
  if (!objectId) {
    hideHoverInfo();
    return;
  }

  if (previewMode === "materials") {
    const info = materialInfoById.get(objectId);
    if (!info) {
      hideHoverInfo();
      return;
    }
    showHoverInfoText(
      event.clientX,
      event.clientY,
      `Material #${info.rowIndex}\nW ${info.w.toFixed(3)} m\nH ${info.h.toFixed(3)} m\nL ${info.l.toFixed(3)} m\nVol ${info.vol.toFixed(3)} m³`,
    );
    return;
  }

  const ifcInfo = ifcInfoById.get(objectId);
  if (!ifcInfo) {
    hideHoverInfo();
    return;
  }
  const best = getBestCandidateForDemand(ifcInfo);
  const card = createHoverMatchCard(ifcInfo, best);
  showHoverInfoNode(event.clientX, event.clientY, card);
});

canvasEl?.addEventListener("mouseleave", () => {
  hideHoverInfo();
});

materialTableBody?.addEventListener("click", (event) => {
  const row = event.target instanceof Element ? event.target.closest("tr") : null;
  if (!row) return;
  const idx = Number.parseInt(row.dataset.rowIndex || "", 10);
  if (!Number.isFinite(idx) || idx < 0 || idx >= demandMeshes.length) return;

  setPreviewMode("materials");
  clearIfcHighlight();
  clearMaterialHighlight();
  selectedMaterialIndex = idx;
  const mesh = demandMeshes[idx];
  mesh.highlighted = true;
  flyToAABB(mesh.aabb);
});

ifcTableBody?.addEventListener("click", (event) => {
  const row = event.target instanceof Element ? event.target.closest("tr") : null;
  if (!row) return;
  const objectId = row.dataset.objectId;
  onIfcDemandRowActivate(objectId);
});

openDemandBankWindowBtn?.addEventListener("click", () => openDemandBankWindow());

dropZoneCsv.addEventListener("dragover", (ev) => {
  ev.preventDefault();
  dropZoneCsv.classList.add("drop-zone--drag");
});

dropZoneCsv.addEventListener("dragleave", () => {
  dropZoneCsv.classList.remove("drop-zone--drag");
});

dropZoneCsv.addEventListener("drop", async (ev) => {
  ev.preventDefault();
  dropZoneCsv.classList.remove("drop-zone--drag");

  const items = ev.dataTransfer?.items;
  if (!items?.length) return;

  const files = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const entry = item.webkitGetAsEntry?.();
    if (entry) {
      try {
        files.push(...(await collectCsvFilesFromEntry(entry)));
      } catch {
        setStatus("Could not read dropped folder.");
        return;
      }
    } else if (item.kind === "file") {
      const f = item.getAsFile();
      if (f && f.name.toLowerCase().endsWith(".csv")) files.push(f);
    }
  }

  if (files.length) await loadFromCsvFiles(files);
  else setStatus("No CSV files in drop.");
});

dropZoneIfc.addEventListener("dragover", (ev) => {
  ev.preventDefault();
  dropZoneIfc.classList.add("drop-zone--drag");
});

dropZoneIfc.addEventListener("dragleave", () => {
  dropZoneIfc.classList.remove("drop-zone--drag");
});

dropZoneIfc.addEventListener("drop", (ev) => {
  ev.preventDefault();
  dropZoneIfc.classList.remove("drop-zone--drag");
  const f = ev.dataTransfer?.files?.[0];
  if (f && f.name.toLowerCase().endsWith(".ifc")) {
    loadIfcFromFile(f);
  } else {
    setStatus("Drop a single .ifc file here.");
  }
});

spacingSlider.addEventListener("input", () => {
  if (demandSizes.length) rebuildLayout();
});

countSlider.addEventListener("input", () => {
  if (demandSizes.length) rebuildLayout();
});

zoomExtentsButton.addEventListener("click", () => {
  if (previewMode === "building" && ifcModel) {
    jumpCameraToIfc();
  } else if (sceneBounds) {
    fitZoomExtents();
  }
});

resetButton.addEventListener("click", () => {
  if (previewMode === "building" && ifcModel) {
    jumpCameraToIfc();
  } else if (sceneBounds) {
    resetViewCamera();
  }
});

renderDashboard();
loadBundledCsv();
