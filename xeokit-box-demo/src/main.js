import {
  Viewer,
  Mesh,
  ReadableGeometry,
  buildBoxGeometry,
  PhongMaterial,
  DirLight,
  WebIFCLoaderPlugin,
  AnnotationsPlugin,
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
const previewResultsBtn = document.getElementById("preview-results");
const dashboardContext = document.getElementById("dashboard-context");
const materialSummary = document.getElementById("material-summary");
const materialTableBody = document.getElementById("material-table-body");
const ifcSummary = document.getElementById("ifc-summary");
const ifcTableBody = document.getElementById("ifc-table-body");
const matchingPanel = document.getElementById("matching-panel");
const matchingStatus = document.getElementById("matching-status");
const matchingSummary = document.getElementById("matching-summary");
const matchingTableBody = document.getElementById("matching-table-body");
const matchingViz = document.getElementById("matching-viz");
const canvasEl = document.getElementById("xeokit_canvas");
const resultsCanvas = document.getElementById("results-canvas");
const hoverInfo = document.getElementById("hover-info");
const leftPanelTitle = document.getElementById("left-panel-title");
const leftPanelDefault = document.getElementById("left-panel-default");
const leftPanelResults = document.getElementById("left-panel-results");
const rightPanelTitle = document.getElementById("right-panel-title");
const rightPanelMatching = document.getElementById("right-panel-matching");
const rightPanelResults = document.getElementById("right-panel-results");
const resultsAlgHint = document.getElementById("results-alg-hint");
const ifcLabelsToggle = document.getElementById("ifc-labels-toggle");

/** @type {"materials" | "building" | "matching" | "results"} */
let previewMode = "materials";

/** @type {"all_pairs" | "greedy_unique"} */
let resultsMatchingAlgorithm = "all_pairs";

/** @type {Mesh[]} */
let demandMeshes = [];
/** @type {{ w: number; h: number; l: number }[]} */
let demandSizes = [];
/** @type {{ id: string; type: string; name: string; material: string; w: number; h: number; l: number; vol: number; source: string }[]} */
let ifcDemandSizes = [];

const MATCHING_MATERIAL = "Timber";
const materialInfoById = new Map();
const ifcInfoById = new Map();
let selectedMaterialIndex = -1;
let selectedIfcObjectId = null;
/** @type {{ demandId: string; demand: { id: string; type: string; name: string; material: string; w: number; h: number; l: number; vol: number; source: string }; candidates: { materialIndex: number; w: number; h: number; l: number; waste: number; volDiffPct: number; similarityScore: number; rotated: boolean }[] } | null} */
let activeMatch = null;
const bestCandidateByDemandId = new Map();
/** Selected row in results diagram: demand index, or null */
let resultsSelectedDemandIdx = null;
let resultsCanvasListenersBound = false;

/** @type {{ minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number } | null} */
let sceneBounds = null;

/** @type {WebIFCLoaderPlugin | null} */
let ifcLoader = null;
/** @type {Promise<WebIFCLoaderPlugin> | null} */
let ifcLoaderPromise = null;
/** Loaded IFC scene model (xeokit SceneModel). */
let ifcModel = null;

const MAX_IFC_LABELS = 500;
let ifcLabelsVisible = false;
/** @type {AnnotationsPlugin | null} */
let annotationsPlugin = null;
/** @type {string[]} */
let ifcLabelAnnotationIds = [];

const IFC_SKIP_TYPES = new Set([
  "IfcProject",
  "IfcSite",
  "IfcBuilding",
  "IfcBuildingStorey",
  "IfcSpace",
]);

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

function getHoverPositionRoot() {
  return canvasEl?.parentElement || canvasEl;
}

function positionHoverInfo(clientX, clientY) {
  if (!hoverInfo) return;
  const root = getHoverPositionRoot();
  if (!root) return;
  const rect = root.getBoundingClientRect();
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

function parseIfcNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const cleaned = value.replace(",", ".").match(/-?\d+(\.\d+)?/);
  if (!cleaned) return null;
  const num = Number.parseFloat(cleaned[0]);
  return Number.isFinite(num) ? num : null;
}

function normalizeIfcDimMeters(value) {
  if (!Number.isFinite(value) || value <= 0) return null;
  if (value > 50) return value / 1000;
  return value;
}

function propertyNameMatches(name, aliases) {
  const n = String(name || "").toLowerCase().replace(/\s+/g, "");
  return aliases.some((alias) => n.includes(alias));
}

const materialPropertyAliases = ["material", "materialtype"];

function normalizeMaterialLabel(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase();
}

function isTimberMaterial(material) {
  return normalizeMaterialLabel(material) === normalizeMaterialLabel(MATCHING_MATERIAL);
}

function extractMaterialFromIfcProperties(metaObject) {
  if (!metaObject?.propertySets?.length) return "";
  for (const pset of metaObject.propertySets) {
    for (const prop of pset?.properties || []) {
      const propName = prop?.name || "";
      if (!propertyNameMatches(propName, materialPropertyAliases)) continue;
      const label = String(prop?.value ?? "").trim();
      if (label) return label;
    }
  }
  return "";
}

function formatIfcType(type) {
  const t = String(type || "IfcElement");
  return t.startsWith("Ifc") ? t.slice(3) : t;
}

function getIfcMetaModel() {
  if (!ifcModel) return null;
  return viewer.metaScene.metaModels[ifcModel.id] ?? null;
}

function collectIfcLeafObjectIds(metaModel) {
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
  return objectIds;
}

function aabbCenter(aabb) {
  return [
    (aabb[0] + aabb[3]) / 2,
    (aabb[1] + aabb[4]) / 2,
    (aabb[2] + aabb[5]) / 2,
  ];
}

function aabbVolume(aabb) {
  return (aabb[3] - aabb[0]) * (aabb[4] - aabb[1]) * (aabb[5] - aabb[2]);
}

function ensureAnnotationsPlugin() {
  if (annotationsPlugin) return annotationsPlugin;
  annotationsPlugin = new AnnotationsPlugin(viewer, {
    markerHTML: "",
    labelHTML:
      "<div class='ifc-annotation-label'>" +
      "<span class='ifc-annotation-type'>{{type}}</span>" +
      "<span class='ifc-annotation-material'>{{material}}</span>" +
      "</div>",
    values: {
      type: "Element",
      material: "—",
    },
  });
  return annotationsPlugin;
}

function clearIfcLabelAnnotations() {
  if (annotationsPlugin) {
    annotationsPlugin.clear();
  }
  ifcLabelAnnotationIds = [];
}

function setIfcLabelsVisible(visible) {
  ifcLabelsVisible = visible;
  if (!annotationsPlugin) return;
  for (const id of ifcLabelAnnotationIds) {
    const ann = annotationsPlugin.annotations[id];
    if (ann) {
      ann.markerShown = false;
      ann.labelShown = visible;
    }
  }
}

function syncIfcLabelsToggleUi() {
  if (!ifcLabelsToggle) return;
  const showControl =
    !!ifcModel && (previewMode === "building" || previewMode === "matching");
  ifcLabelsToggle.hidden = !showControl;
  if (!showControl) return;
  ifcLabelsToggle.classList.toggle("ifc-labels-toggle--on", ifcLabelsVisible);
  ifcLabelsToggle.setAttribute("aria-pressed", ifcLabelsVisible ? "true" : "false");
  ifcLabelsToggle.textContent = ifcLabelsVisible ? "labels on" : "labels off";
}

/**
 * @returns {{ id: string; type: string; material: string; vol: number; entity: object; worldPos: number[] }[]}
 */
function buildIfcLabelCandidates() {
  const metaModel = getIfcMetaModel();
  if (!metaModel) return [];

  /** @type {{ id: string; type: string; material: string; vol: number; entity: object; worldPos: number[] }[]} */
  const rows = [];
  for (const id of collectIfcLeafObjectIds(metaModel)) {
    const meta = metaModel.metaObjects[id];
    const type = meta?.type || "IfcElement";
    if (IFC_SKIP_TYPES.has(type)) continue;
    if (meta?.children?.length) continue;

    const entity = viewer.scene.objects?.[id];
    if (!entity) continue;
    const aabb = entity.aabb;
    if (!aabb || aabb.length < 6) continue;
    const vol = aabbVolume(aabb);
    if (!Number.isFinite(vol) || vol <= 0) continue;

    const material = extractMaterialFromIfcProperties(meta);
    rows.push({
      id,
      type: formatIfcType(type),
      material: material || "—",
      vol,
      entity,
      worldPos: aabbCenter(aabb),
    });
  }

  rows.sort((a, b) => b.vol - a.vol);
  return rows;
}

function rebuildIfcLabelAnnotations() {
  clearIfcLabelAnnotations();
  if (!ifcModel) {
    syncIfcLabelsToggleUi();
    return;
  }

  const plugin = ensureAnnotationsPlugin();
  const candidates = buildIfcLabelCandidates();
  const capped = candidates.slice(0, MAX_IFC_LABELS);

  for (const row of capped) {
    const annId = `ifc-label-${row.id}`;
    plugin.createAnnotation({
      id: annId,
      entity: row.entity,
      worldPos: row.worldPos,
      occludable: true,
      markerShown: false,
      labelShown: ifcLabelsVisible,
      values: {
        type: row.type,
        material: row.material,
      },
    });
    ifcLabelAnnotationIds.push(annId);
  }

  syncIfcLabelsToggleUi();
}

function getIfcMetaForObject(objectId) {
  const metaModel = getIfcMetaModel();
  return metaModel?.metaObjects?.[objectId] ?? null;
}

function createIfcElementHoverCard(objectId) {
  const meta = getIfcMetaForObject(objectId);
  const type = formatIfcType(meta?.type);
  const material = extractMaterialFromIfcProperties(meta) || "—";
  const name = meta?.name || objectId;
  const info = ifcInfoById.get(objectId);

  const card = document.createElement("div");
  const title = document.createElement("div");
  title.className = "hover-info__title";
  title.textContent = type;
  card.appendChild(title);

  const metaEl = document.createElement("div");
  metaEl.className = "hover-info__meta";
  metaEl.textContent = name;
  card.appendChild(metaEl);

  const kv = document.createElement("div");
  kv.className = "hover-info__kv";
  kv.append(...createKeyValueRow("Material", material));
  if (info) {
    kv.append(
      ...createKeyValueRow(
        "Size",
        `${info.w.toFixed(2)}×${info.h.toFixed(2)}×${info.l.toFixed(2)} m`,
      ),
    );
    kv.append(...createKeyValueRow("Volume", `${info.vol.toFixed(3)} m³`));
  }
  card.appendChild(kv);
  return card;
}

function extractDimsFromIfcProperties(metaObject) {
  if (!metaObject?.propertySets?.length) return null;

  const widthAliases = ["width", "overallwidth", "flangewidth", "b"];
  const heightAliases = ["height", "overallheight", "depth", "h"];
  const lengthAliases = ["length", "overalllength", "span", "l"];

  let w = null;
  let h = null;
  let l = null;

  for (const pset of metaObject.propertySets) {
    for (const prop of pset?.properties || []) {
      const raw = parseIfcNumber(prop?.value);
      if (raw == null) continue;
      const num = normalizeIfcDimMeters(raw);
      if (num == null) continue;

      const propName = prop?.name || "";
      if (w == null && propertyNameMatches(propName, widthAliases)) w = num;
      else if (h == null && propertyNameMatches(propName, heightAliases)) h = num;
      else if (l == null && propertyNameMatches(propName, lengthAliases)) l = num;
    }
  }

  if (w == null && h == null && l == null) return null;
  return { w, h, l };
}

function materialFitsDemand(material, demand) {
  const direct = material.w >= demand.w && material.h >= demand.h;
  const rotated = material.w >= demand.h && material.h >= demand.w;
  if (!direct && !rotated) return { fits: false, rotated: false };
  if (material.l < demand.l) return { fits: false, rotated: false };
  return { fits: true, rotated: !direct && rotated };
}

/**
 * @param {{ w: number; h: number; l: number; vol: number }} supply
 * @param {{ w: number; h: number; l: number; vol: number }} demand
 * @returns {{ volDiffPct: number; waste: number; supplyVol: number; similarityScore: number; rotated: boolean } | null}
 */
function linkStats(supply, demand) {
  const fit = materialFitsDemand(supply, demand);
  if (!fit.fits) return null;
  const supplyVol = supply.w * supply.h * supply.l;
  const waste = supplyVol - demand.vol;
  const volDiffPct = demand.vol > 0 ? (100 * waste) / demand.vol : 0;
  return {
    volDiffPct,
    waste,
    supplyVol,
    similarityScore: Math.max(0, 100 - volDiffPct),
    rotated: fit.rotated,
  };
}

/**
 * @param {{ demandIdx: number; supplyIdx: number }[]} allLinks
 * @param {{ w: number; h: number; l: number; vol: number }[]} demands
 * @param {{ w: number; h: number; l: number }[]} supplies
 */
function filterLinksByAlgorithm(allLinks, demands, supplies) {
  if (resultsMatchingAlgorithm !== "greedy_unique") return allLinks;

  /** @type {{ demandIdx: number; supplyIdx: number; similarityScore: number; waste: number }[]} */
  const scored = [];
  for (const link of allLinks) {
    const d = demands[link.demandIdx];
    const s = supplies[link.supplyIdx];
    const st = linkStats(s, d);
    if (!st) continue;
    scored.push({
      demandIdx: link.demandIdx,
      supplyIdx: link.supplyIdx,
      similarityScore: st.similarityScore,
      waste: st.waste,
    });
  }
  scored.sort((a, b) => {
    if (b.similarityScore !== a.similarityScore) return b.similarityScore - a.similarityScore;
    return a.waste - b.waste;
  });
  const usedD = new Set();
  const usedS = new Set();
  /** @type {{ demandIdx: number; supplyIdx: number }[]} */
  const out = [];
  for (const e of scored) {
    if (usedD.has(e.demandIdx) || usedS.has(e.supplyIdx)) continue;
    usedD.add(e.demandIdx);
    usedS.add(e.supplyIdx);
    out.push({ demandIdx: e.demandIdx, supplyIdx: e.supplyIdx });
  }
  return out;
}

function syncResultsAlgorithmButtons() {
  if (!rightPanelResults) return;
  for (const btn of rightPanelResults.querySelectorAll("[data-results-alg]")) {
    btn.classList.toggle(
      "results-alg__btn--active",
      btn.getAttribute("data-results-alg") === resultsMatchingAlgorithm,
    );
  }
}

function updateResultsAlgorithmHint() {
  if (!resultsAlgHint) return;
  if (resultsMatchingAlgorithm === "greedy_unique") {
    resultsAlgHint.textContent =
      "Greedy one-to-one: at most one connection per demand and per supply, preferring highest score (lower waste). Not a global optimum.";
  } else {
    resultsAlgHint.textContent =
      "All feasible pairs: every supply that can cover a demand gets a link; one supply may connect to many demands.";
  }
}

function syncLayoutPanels() {
  if (leftPanelTitle) {
    leftPanelTitle.textContent = previewMode === "results" ? "matching properties" : "analysis";
  }
  if (leftPanelDefault) leftPanelDefault.hidden = previewMode === "results";
  if (leftPanelResults) leftPanelResults.hidden = previewMode !== "results";
  if (rightPanelTitle) {
    rightPanelTitle.textContent = previewMode === "results" ? "matching properties" : "matching v0.3";
  }
  if (rightPanelMatching) rightPanelMatching.hidden = previewMode !== "matching";
  if (rightPanelResults) rightPanelResults.hidden = previewMode !== "results";
  syncResultsAlgorithmButtons();
  updateResultsAlgorithmHint();
}

function ensureResultsAlgorithmClick() {
  if (!rightPanelResults || rightPanelResults.dataset.bound) return;
  rightPanelResults.dataset.bound = "1";
  rightPanelResults.addEventListener("click", (e) => {
    const btn = e.target instanceof Element ? e.target.closest("[data-results-alg]") : null;
    if (!btn) return;
    const alg = btn.getAttribute("data-results-alg");
    if (alg !== "all_pairs" && alg !== "greedy_unique") return;
    resultsMatchingAlgorithm = alg;
    syncResultsAlgorithmButtons();
    updateResultsAlgorithmHint();
    renderResultsCanvas();
  });
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
  if (demand.material) {
    kv.append(...createKeyValueRow("Material", demand.material));
  }
  kv.append(
    ...createKeyValueRow(
      "Demand",
      `${demand.w.toFixed(2)}×${demand.h.toFixed(2)}×${demand.l.toFixed(2)} m`,
    ),
  );
  kv.append(...createKeyValueRow("Demand vol", `${demand.vol.toFixed(3)} m³`));
  kv.append(...createKeyValueRow("Source", demand.source));
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
    const metaModel = ifcModel ? viewer.metaScene.metaModels[ifcModel.id] : null;
    const meta = metaModel?.metaObjects?.[objectId];
    const material = meta ? extractMaterialFromIfcProperties(meta) : "";
    if (material && !isTimberMaterial(material)) {
      setStatus(
        `Matching is limited to ${MATCHING_MATERIAL} elements (this object: ${material}).`,
      );
    } else {
      setStatus(
        `Clicked object is not in the ${MATCHING_MATERIAL} demand bank (missing or unknown Material property).`,
      );
    }
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

  const metaModel = getIfcMetaModel();
  if (!metaModel) return [];

  ifcInfoById.clear();
  bestCandidateByDemandId.clear();

  /** @type {{ id: string; type: string; name: string; material: string; w: number; h: number; l: number; vol: number; source: string }[]} */
  const rows = [];
  for (const id of collectIfcLeafObjectIds(metaModel)) {
    const meta = metaModel.metaObjects[id];
    const type = meta?.type || "IfcElement";
    if (IFC_SKIP_TYPES.has(type)) continue;
    if (meta?.children?.length) continue;

    const material = extractMaterialFromIfcProperties(meta);
    if (!isTimberMaterial(material)) continue;

    const entity = viewer.scene.objects?.[id];
    if (!entity) continue;

    const aabb = entity.aabb;
    if (!aabb || aabb.length < 6) continue;

    const aabbW = aabb[3] - aabb[0];
    const aabbH = aabb[4] - aabb[1];
    const aabbL = aabb[5] - aabb[2];
    const fromIfc = extractDimsFromIfcProperties(meta);
    const w = fromIfc?.w ?? aabbW;
    const h = fromIfc?.h ?? aabbH;
    const l = fromIfc?.l ?? aabbL;
    if (![w, h, l].every(Number.isFinite)) continue;
    if (w <= 0 || h <= 0 || l <= 0) continue;

    const row = {
      id,
      type,
      name: meta?.name || id,
      material,
      w,
      h,
      l,
      vol: w * h * l,
      source: fromIfc ? "ifc-properties+fallback" : "aabb-fallback",
    };
    rows.push(row);
    ifcInfoById.set(id, row);
  }

  rows.sort((a, b) => b.vol - a.vol);
  return rows;
}

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
  clearEl(ifcSummary);
  ifcTableBody.replaceChildren();

  if (!ifcModel) {
    appendSummaryRow(ifcSummary, "status", "no ifc loaded");
    return;
  }

  if (ifcDemandSizes.length === 0) {
    appendSummaryRow(
      ifcSummary,
      "status",
      `no measurable ${MATCHING_MATERIAL} objects found`,
    );
    return;
  }

  appendSummaryRow(ifcSummary, "material filter", MATCHING_MATERIAL);

  let totalVolume = 0;
  let sumLength = 0;
  let minL = Infinity;
  let maxL = -Infinity;
  for (const row of ifcDemandSizes) {
    totalVolume += row.vol;
    sumLength += row.l;
    minL = Math.min(minL, row.l);
    maxL = Math.max(maxL, row.l);
  }

  appendSummaryRow(ifcSummary, "objects", String(ifcDemandSizes.length));
  appendSummaryRow(ifcSummary, "total volume", fmtM3(totalVolume));
  appendSummaryRow(ifcSummary, "Σ length (L)", fmtM(sumLength));
  appendSummaryRow(ifcSummary, "length L (min → max)", `${fmtM(minL)} → ${fmtM(maxL)}`);
  const maxRows = 800;
  const shownRows = Math.min(maxRows, ifcDemandSizes.length);
  appendSummaryRow(ifcSummary, "shown in table", `${shownRows} / ${ifcDemandSizes.length}`);

  for (let i = 0; i < shownRows; i++) {
    const { id, type, name, w, h, l, vol } = ifcDemandSizes[i];
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
    ifcTableBody.appendChild(tr);
  }
}

function renderDashboard() {
  if (!dashboardContext) return;
  ensureResultsAlgorithmClick();
  syncLayoutPanels();

  if (previewMode === "building") {
    dashboardContext.textContent = ifcModel
      ? "Canvas: building (IFC). Use labels in the viewer to show each element's type and material (up to 500 largest elements)."
      : "Canvas: building (IFC). Load an IFC file to populate the demand bank table.";
  } else if (previewMode === "matching") {
    dashboardContext.textContent =
      `Canvas: matching mode. Only IFC elements with Material=${MATCHING_MATERIAL} are in the demand bank. Click one to see material-bank candidates on the right.`;
  } else if (previewMode === "results") {
    dashboardContext.textContent =
      "Results: graph in the canvas. Pick how connections are built in the right-hand matching properties panel (not the lists below in other modes).";
  } else {
    dashboardContext.textContent =
      "Canvas: material bank. Left table shows material elements; second table shows IFC demand bank when loaded.";
  }

  if (previewMode !== "results") {
    renderMaterialTable();
    renderIfcTable();
  } else {
    materialSummary && clearEl(materialSummary);
    ifcSummary && clearEl(ifcSummary);
    materialTableBody && materialTableBody.replaceChildren();
    ifcTableBody && ifcTableBody.replaceChildren();
  }
  renderMatchingPanel();
  renderResultsCanvas();
  syncIfcLabelsToggleUi();
}

function renderMatchingPanel() {
  if (!matchingPanel || !matchingStatus || !matchingSummary || !matchingTableBody || !matchingViz) return;

  matchingPanel.classList.toggle(
    "matching-panel--active",
    previewMode === "matching" || previewMode === "results",
  );
  if (previewMode !== "matching") {
    return;
  }
  clearEl(matchingSummary);
  matchingTableBody.replaceChildren();
  matchingViz.replaceChildren();

  const matchableCount = countMatchableDemandObjects();
  const demandCount = ifcDemandSizes.length;
  const coverage = demandCount > 0 ? (100 * matchableCount) / demandCount : 0;
  appendSummaryRow(matchingSummary, "material elements", String(demandSizes.length));
  appendSummaryRow(matchingSummary, "demand elements", `${demandCount} (${MATCHING_MATERIAL} only)`);
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
  const vizShown = Math.min(candidates.length, 14);
  appendSummaryRow(matchingSummary, "shown", `${shown} / ${candidates.length}`);
  appendSummaryRow(matchingSummary, "sorted by", "score (highest first)");
  appendSummaryRow(matchingSummary, "dimension source", demand.source);

  const svgNS = "http://www.w3.org/2000/svg";
  const svgWidth = 320;
  const rowGap = 28;
  const margin = 14;
  const svgHeight = Math.max(120, margin * 2 + vizShown * rowGap);
  const leftX = 86;
  const rightX = 236;
  const demandY = svgHeight / 2;
  const glyphW = 40;
  const glyphH = 20;

  const maxW = Math.max(
    demand.w,
    ...candidates.slice(0, vizShown).map((r) => (r.rotated ? r.h : r.w)),
  );
  const maxH = Math.max(
    demand.h,
    ...candidates.slice(0, vizShown).map((r) => (r.rotated ? r.w : r.h)),
  );

  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${svgWidth} ${svgHeight}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", String(svgHeight));
  svg.classList.add("matching-diagram");

  const demandRect = document.createElementNS(svgNS, "rect");
  const dW = Math.max(8, (demand.w / maxW) * glyphW);
  const dH = Math.max(6, (demand.h / maxH) * glyphH);
  demandRect.setAttribute("x", String(leftX - dW / 2));
  demandRect.setAttribute("y", String(demandY - dH / 2));
  demandRect.setAttribute("width", String(dW));
  demandRect.setAttribute("height", String(dH));
  demandRect.setAttribute("class", "matching-diagram__demand");
  svg.appendChild(demandRect);

  for (let i = 0; i < vizShown; i++) {
    const c = candidates[i];
    const y = margin + i * rowGap + rowGap / 2;
    const cWRaw = c.rotated ? c.h : c.w;
    const cHRaw = c.rotated ? c.w : c.h;
    const cW = Math.max(8, (cWRaw / maxW) * glyphW);
    const cH = Math.max(6, (cHRaw / maxH) * glyphH);

    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", String(leftX + dW / 2));
    line.setAttribute("y1", String(demandY));
    line.setAttribute("x2", String(rightX - cW / 2));
    line.setAttribute("y2", String(y));
    line.setAttribute("class", "matching-diagram__link");
    svg.appendChild(line);

    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", String(rightX - cW / 2));
    rect.setAttribute("y", String(y - cH / 2));
    rect.setAttribute("width", String(cW));
    rect.setAttribute("height", String(cH));
    rect.setAttribute("class", "matching-diagram__supply");
    svg.appendChild(rect);

    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", String(rightX + 26));
    label.setAttribute("y", String(y + 3));
    label.setAttribute("class", "matching-diagram__label");
    label.textContent = `#${c.materialIndex + 1}`;
    svg.appendChild(label);
  }
  matchingViz.appendChild(svg);

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

function bindResultsCanvasEvents() {
  if (resultsCanvasListenersBound || !resultsCanvas) return;
  resultsCanvasListenersBound = true;
  resultsCanvas.addEventListener("click", onResultsCanvasClick);
  resultsCanvas.addEventListener("pointermove", onResultsCanvasPointerMove);
  resultsCanvas.addEventListener("pointerleave", () => hideHoverInfo());
}

/**
 * @param {MouseEvent} e
 */
function onResultsCanvasClick(e) {
  if (previewMode !== "results") return;
  if (!(e.target instanceof Element) || !e.target.closest?.(".results-svg")) return;
  const demandG = e.target.closest?.("[data-results-demand]");
  if (!demandG) return;
  const idx = parseInt(demandG.getAttribute("data-results-demand") || "-1", 10);
  if (!Number.isFinite(idx) || idx < 0) return;
  resultsSelectedDemandIdx = resultsSelectedDemandIdx === idx ? null : idx;
  renderResultsCanvas();
  e.stopPropagation();
}

/**
 * @param {PointerEvent} e
 */
function onResultsCanvasPointerMove(e) {
  if (previewMode !== "results" || !resultsCanvas) return;
  if (!(e.target instanceof Element) || !e.target.closest?.(".results-svg")) {
    hideHoverInfo();
    return;
  }
  const dem = e.target.closest?.("[data-results-demand]");
  if (dem) {
    const d = parseInt(dem.getAttribute("data-results-demand") || "-1", 10);
    if (Number.isFinite(d) && d >= 0) showResultsDemandTooltip(e.clientX, e.clientY, d);
    return;
  }
  const sup = e.target.closest?.("[data-results-supply]");
  if (sup) {
    const s = parseInt(sup.getAttribute("data-results-supply") || "-1", 10);
    if (Number.isFinite(s) && s >= 0) showResultsSupplyTooltip(e.clientX, e.clientY, s);
    return;
  }
  hideHoverInfo();
}

const MAX_RESULT_TOOLTIP_LINES = 16;

function showResultsDemandTooltip(clientX, clientY, dIdx) {
  const demand = ifcDemandSizes[dIdx];
  if (!demand) return;
  const card = document.createElement("div");
  const t = document.createElement("div");
  t.className = "hover-info__title";
  t.textContent = `Demand #${dIdx + 1} ${demand.type}`;
  card.appendChild(t);
  const meta = document.createElement("div");
  meta.className = "hover-info__meta";
  meta.textContent = `${demand.w.toFixed(2)}×${demand.h.toFixed(2)}×${demand.l.toFixed(2)} m · vol ${demand.vol.toFixed(3)} m³ · ${demand.source}`;
  card.appendChild(meta);
  const div = document.createElement("div");
  div.className = "hover-info__divider";
  card.appendChild(div);
  const sub = document.createElement("div");
  sub.className = "hover-info__meta";
  sub.textContent = "Feasible supply (sorted by least waste %)";
  card.appendChild(sub);
  /** @type {{ s: number; volDiffPct: number; waste: number; similarityScore: number }[]} */
  const list = [];
  for (let s = 0; s < demandSizes.length; s++) {
    const st = linkStats(demandSizes[s], demand);
    if (st) list.push({ s, ...st });
  }
  list.sort((a, b) => a.volDiffPct - b.volDiffPct);
  const show = list.slice(0, MAX_RESULT_TOOLTIP_LINES);
  for (const row of show) {
    const line = document.createElement("div");
    line.className = "hover-info__meta";
    line.textContent = `Supply #${row.s + 1}: score ${row.similarityScore.toFixed(1)}% · Δvol ${row.volDiffPct.toFixed(1)}% · waste ${row.waste.toFixed(3)} m³${row.rotated ? " · 90°" : ""}`;
    card.appendChild(line);
  }
  if (list.length > show.length) {
    const more = document.createElement("div");
    more.className = "hover-info__meta";
    more.textContent = `+ ${list.length - show.length} more…`;
    card.appendChild(more);
  }
  if (list.length === 0) {
    const none = document.createElement("div");
    none.className = "hover-info__meta";
    none.textContent = "No supply element fits this demand.";
    card.appendChild(none);
  }
  showHoverInfoNode(clientX, clientY, card);
}

function showResultsSupplyTooltip(clientX, clientY, sIdx) {
  const mat = demandSizes[sIdx];
  if (!mat) return;
  const vol = mat.w * mat.h * mat.l;
  const card = document.createElement("div");
  const t = document.createElement("div");
  t.className = "hover-info__title";
  t.textContent = `Supply #${sIdx + 1}`;
  card.appendChild(t);
  const meta = document.createElement("div");
  meta.className = "hover-info__meta";
  meta.textContent = `${mat.w.toFixed(2)}×${mat.h.toFixed(2)}×${mat.l.toFixed(2)} m · vol ${vol.toFixed(3)} m³`;
  card.appendChild(meta);
  const div = document.createElement("div");
  div.className = "hover-info__divider";
  card.appendChild(div);
  const sub = document.createElement("div");
  sub.className = "hover-info__meta";
  sub.textContent = "This supply can cover these demands (least waste first)";
  card.appendChild(sub);
  /** @type {{ d: number; volDiffPct: number; waste: number; similarityScore: number; demandVol: number }[]} */
  const list = [];
  for (let d = 0; d < ifcDemandSizes.length; d++) {
    const st = linkStats(mat, ifcDemandSizes[d]);
    if (st) list.push({ d, demandVol: ifcDemandSizes[d].vol, ...st });
  }
  list.sort((a, b) => a.volDiffPct - b.volDiffPct);
  const show = list.slice(0, MAX_RESULT_TOOLTIP_LINES);
  for (const row of show) {
    const line = document.createElement("div");
    line.className = "hover-info__meta";
    const wastefrac = row.supplyVol > 0 ? (100 * row.waste) / row.supplyVol : 0;
    line.textContent = `Demand #${row.d + 1}: score ${row.similarityScore.toFixed(1)}% · Δvol vs demand ${row.volDiffPct.toFixed(1)}% · waste of supply ${wastefrac.toFixed(1)}%${row.rotated ? " · 90°" : ""}`;
    card.appendChild(line);
  }
  if (list.length > show.length) {
    const more = document.createElement("div");
    more.className = "hover-info__meta";
    more.textContent = `+ ${list.length - show.length} more…`;
    card.appendChild(more);
  }
  if (list.length === 0) {
    const none = document.createElement("div");
    none.className = "hover-info__meta";
    none.textContent = "No demand element can be cut from this supply.";
    card.appendChild(none);
  }
  showHoverInfoNode(clientX, clientY, card);
}

function renderResultsCanvas() {
  if (!resultsCanvas) return;
  resultsCanvas.replaceChildren();
  bindResultsCanvasEvents();

  if (previewMode !== "results") {
    resultsCanvas.hidden = true;
    return;
  }

  resultsCanvas.hidden = false;

  if (
    resultsSelectedDemandIdx != null &&
    resultsSelectedDemandIdx >= (ifcDemandSizes?.length ?? 0)
  ) {
    resultsSelectedDemandIdx = null;
  }

  const status = document.createElement("p");
  status.className = "results-canvas__status";

  if (!ifcModel) {
    status.textContent = "Load IFC to populate demand bank before viewing results.";
    resultsCanvas.appendChild(status);
    return;
  }
  if (ifcDemandSizes.length === 0) {
    status.textContent = `No ${MATCHING_MATERIAL} demand elements found in IFC.`;
    resultsCanvas.appendChild(status);
    return;
  }
  if (demandSizes.length === 0) {
    status.textContent = "Load material bank CSV to populate supply elements.";
    resultsCanvas.appendChild(status);
    return;
  }

  const demands = ifcDemandSizes;
  const supplies = demandSizes.map((s, i) => ({ id: i + 1, w: s.w, h: s.h, l: s.l }));
  /** @type {{ demandIdx: number; supplyIdx: number }[]} */
  const allPairLinks = [];
  for (let d = 0; d < demands.length; d++) {
    for (let s = 0; s < supplies.length; s++) {
      if (materialFitsDemand(supplies[s], demands[d]).fits) {
        allPairLinks.push({ demandIdx: d, supplyIdx: s });
      }
    }
  }

  const links = filterLinksByAlgorithm(allPairLinks, demands, supplies);
  const algLabel =
    resultsMatchingAlgorithm === "greedy_unique"
      ? "greedy one-to-one"
      : "all feasible pairs";

  const matchedDemandSet = new Set(links.map((l) => l.demandIdx));
  const matchedSupplySet = new Set(links.map((l) => l.supplyIdx));
  const activeSupplySet = new Set();
  if (resultsSelectedDemandIdx != null) {
    for (const link of links) {
      if (link.demandIdx === resultsSelectedDemandIdx) {
        activeSupplySet.add(link.supplyIdx);
      }
    }
  }

  let hint = "Click a demand row to highlight its supply matches.";
  if (resultsSelectedDemandIdx != null) {
    hint = `Selected demand #${resultsSelectedDemandIdx + 1} — ${activeSupplySet.size} supply link(s) highlighted.`;
  }
  status.textContent = `Algorithm: ${algLabel} · shown edges ${links.length} (all feasible pair edges ${allPairLinks.length}) · Demand ${demands.length} · Supply ${supplies.length} · Matchable demand ${matchedDemandSet.size}/${demands.length} · ${hint}`;
  resultsCanvas.appendChild(status);

  const svgNS = "http://www.w3.org/2000/svg";
  const rowGap = 18;
  const marginTop = 30;
  const marginBottom = 20;
  const height = Math.max(
    260,
    marginTop + Math.max(demands.length, supplies.length) * rowGap + marginBottom,
  );
  const width = 1080;
  const demandX = 70;
  const supplyX = 620;
  const nodeW = 260;
  const nodeH = 12;

  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "results-svg");

  const titleDemand = document.createElementNS(svgNS, "text");
  titleDemand.setAttribute("x", String(demandX));
  titleDemand.setAttribute("y", "16");
  titleDemand.setAttribute("class", "results-title");
  titleDemand.textContent = "Demand bank (IFC)";
  svg.appendChild(titleDemand);

  const titleSupply = document.createElementNS(svgNS, "text");
  titleSupply.setAttribute("x", String(supplyX));
  titleSupply.setAttribute("y", "16");
  titleSupply.setAttribute("class", "results-title");
  titleSupply.textContent = "Supply bank (CSV)";
  svg.appendChild(titleSupply);

  for (const link of links) {
    if (
      resultsSelectedDemandIdx != null &&
      link.demandIdx === resultsSelectedDemandIdx
    ) {
      continue;
    }
    const y1 = marginTop + link.demandIdx * rowGap;
    const y2 = marginTop + link.supplyIdx * rowGap;
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", String(demandX + nodeW));
    line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(supplyX));
    line.setAttribute("y2", String(y2));
    line.setAttribute("class", "results-link");
    svg.appendChild(line);
  }

  if (resultsSelectedDemandIdx != null) {
    for (const link of links) {
      if (link.demandIdx !== resultsSelectedDemandIdx) continue;
      const y1 = marginTop + link.demandIdx * rowGap;
      const y2 = marginTop + link.supplyIdx * rowGap;
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", String(demandX + nodeW));
      line.setAttribute("y1", String(y1));
      line.setAttribute("x2", String(supplyX));
      line.setAttribute("y2", String(y2));
      line.setAttribute("class", "results-link results-link--active");
      svg.appendChild(line);
    }
  }

  for (let i = 0; i < demands.length; i++) {
    const demand = demands[i];
    const y = marginTop + i * rowGap;
    const g = document.createElementNS(svgNS, "g");
    g.setAttribute("data-results-demand", String(i));

    const nodeClass = `results-node results-node--demand${
      resultsSelectedDemandIdx === i ? " results-node--selected" : ""
    }${matchedDemandSet.has(i) ? "" : ""}`;
    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", String(demandX));
    rect.setAttribute("y", String(y - nodeH / 2));
    rect.setAttribute("width", String(nodeW));
    rect.setAttribute("height", String(nodeH));
    rect.setAttribute("class", nodeClass);
    g.appendChild(rect);

    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", String(demandX + 4));
    label.setAttribute("y", String(y));
    label.setAttribute("class", "results-label");
    label.textContent = `${i + 1}. ${demand.type} | ${demand.w.toFixed(2)}×${demand.h.toFixed(2)}×${demand.l.toFixed(2)}`;
    g.appendChild(label);

    const hit = document.createElementNS(svgNS, "rect");
    hit.setAttribute("class", "results-hit");
    hit.setAttribute("x", String(demandX - 2));
    hit.setAttribute("y", String(y - 9));
    hit.setAttribute("width", String(nodeW + 4));
    hit.setAttribute("height", "18");
    g.appendChild(hit);
    svg.appendChild(g);
  }

  if (resultsSelectedDemandIdx != null) {
    for (const sIdx of activeSupplySet) {
      const y = marginTop + sIdx * rowGap;
      const ell = document.createElementNS(svgNS, "ellipse");
      ell.setAttribute("cx", String(supplyX + nodeW / 2));
      ell.setAttribute("cy", String(y));
      ell.setAttribute("rx", String(nodeW / 2 + 6));
      ell.setAttribute("ry", "10");
      ell.setAttribute("class", "results-oblong");
      svg.appendChild(ell);
    }
  }

  for (let i = 0; i < supplies.length; i++) {
    const supply = supplies[i];
    const y = marginTop + i * rowGap;
    const g = document.createElementNS(svgNS, "g");
    g.setAttribute("data-results-supply", String(i));
    const linked = activeSupplySet.has(i);
    const sc = `results-node results-node--supply${
      linked ? " results-supply--linked" : ""
    }`;
    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", String(supplyX));
    rect.setAttribute("y", String(y - nodeH / 2));
    rect.setAttribute("width", String(nodeW));
    rect.setAttribute("height", String(nodeH));
    rect.setAttribute("class", sc);
    g.appendChild(rect);

    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", String(supplyX + 4));
    label.setAttribute("y", String(y));
    label.setAttribute("class", "results-label");
    label.textContent = `${supply.id}. ${supply.w.toFixed(2)}×${supply.h.toFixed(2)}×${supply.l.toFixed(2)}`;
    g.appendChild(label);

    const hit = document.createElementNS(svgNS, "rect");
    hit.setAttribute("class", "results-hit");
    hit.setAttribute("x", String(supplyX - 2));
    hit.setAttribute("y", String(y - 9));
    hit.setAttribute("width", String(nodeW + 4));
    hit.setAttribute("height", "18");
    g.appendChild(hit);
    svg.appendChild(g);
  }

  resultsCanvas.appendChild(svg);
}

function syncSceneVisibility() {
  const showBoxes = previewMode === "materials";
  for (const mesh of demandMeshes) {
    mesh.visible = showBoxes;
  }
  if (ifcModel) {
    ifcModel.visible =
      previewMode === "building" || previewMode === "matching";
  }
}

/**
 * @param {"materials" | "building" | "matching" | "results"} mode
 */
function setPreviewMode(mode) {
  if ((mode === "building" || mode === "matching" || mode === "results") && !ifcModel) {
    setStatus("Load an IFC file first, then switch to building preview.");
    return;
  }

  previewMode = mode;
  hideHoverInfo();
  if (mode !== "results") {
    resultsSelectedDemandIdx = null;
  }
  previewMaterialsBtn.classList.toggle("toggle-btn--active", mode === "materials");
  previewBuildingBtn.classList.toggle("toggle-btn--active", mode === "building");
  previewMatchingBtn.classList.toggle("toggle-btn--active", mode === "matching");
  previewResultsBtn.classList.toggle("toggle-btn--active", mode === "results");

  syncSceneVisibility();

  if (mode === "materials") {
    if (sceneBounds) fitZoomExtents();
  } else if (mode !== "results") {
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
      resultsSelectedDemandIdx = null;
      clearIfcLabelAnnotations();
      syncIfcLabelsToggleUi();
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
      rebuildIfcLabelAnnotations();
      activeMatch = null;
      syncSceneVisibility();
      const labelTotal = buildIfcLabelCandidates().length;
      const labelNote =
        labelTotal > MAX_IFC_LABELS
          ? ` · labels for ${MAX_IFC_LABELS}/${labelTotal} elements`
          : labelTotal > 0
            ? ` · ${labelTotal} label targets`
            : "";
      setStatus(
        `IFC loaded: ${file.name} · ${ifcDemandSizes.length} ${MATCHING_MATERIAL} demand objects${labelNote}`,
      );
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
previewResultsBtn.addEventListener("click", () => setPreviewMode("results"));

ifcLabelsToggle?.addEventListener("click", () => {
  if (!ifcModel) return;
  setIfcLabelsVisible(!ifcLabelsVisible);
  syncIfcLabelsToggleUi();
  const n = ifcLabelAnnotationIds.length;
  setStatus(
    ifcLabelsVisible
      ? `IFC labels on (${n} element${n === 1 ? "" : "s"})`
      : "IFC labels off",
  );
});

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
  if (previewMode === "results") {
    hideHoverInfo();
    return;
  }
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

  if (!ifcModel || !getIfcMetaForObject(objectId)) {
    hideHoverInfo();
    return;
  }

  const ifcInfo = ifcInfoById.get(objectId);
  if (ifcInfo && demandSizes.length > 0) {
    const best = getBestCandidateForDemand(ifcInfo);
    const card = createHoverMatchCard(ifcInfo, best);
    showHoverInfoNode(event.clientX, event.clientY, card);
    return;
  }

  showHoverInfoNode(
    event.clientX,
    event.clientY,
    createIfcElementHoverCard(objectId),
  );
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
});

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
