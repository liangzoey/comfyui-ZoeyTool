// Zoey - MiniMax H3 参考转视频 (@) 前端扩展
// prompt 为 contenteditable 富文本：@P1/@V1/@A1 渲染成带缩略图的 chip（图片/音频/视频），
// 输入 @ 弹出选择器选择已连接参考素材；串行化回 @P1 文本交给后端。
import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import { ComfyWidgets } from "../../../../scripts/widgets.js";
import { addWidget, DOMWidgetImpl } from "../../../../scripts/domWidget.js";
import { $el } from "../../../../scripts/ui.js";

const NODE_TYPE = "ZoeyMiniMaxH3ReferenceToVideo";
const EXT_NAME = "Zoey.MiniMaxH3.RefPicker";

const STYLE = `
.zoey-ref-picker {
  position: fixed;
  z-index: 99999;
  background: var(--comfy-menu-bg, #202020);
  border: 1px solid var(--border-color, #333);
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(0,0,0,.5);
  min-width: 240px;
  max-height: 300px;
  overflow-y: auto;
  font-family: var(--font-family, sans-serif);
  font-size: 13px;
  color: var(--descrip-text, #ccc);
}
.zoey-ref-header {
  padding: 6px 10px;
  font-size: 12px;
  opacity: .7;
  border-bottom: 1px solid var(--border-color, #333);
  white-space: nowrap;
}
.zoey-ref-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  cursor: pointer;
  user-select: none;
}
.zoey-ref-item:hover,
.zoey-ref-item--selected {
  background: rgba(255,255,255,.08);
}
.zoey-ref-thumb {
  width: 40px;
  height: 40px;
  flex: 0 0 40px;
  border-radius: 4px;
  object-fit: cover;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}
.zoey-ref-meta {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}
.zoey-ref-label {
  font-weight: 600;
}
.zoey-ref-hint {
  font-size: 11px;
  opacity: .6;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 220px;
}
.zoey-ref-tag {
  font-size: 11px;
  font-family: monospace;
  color: #7ec8ff;
}
.zoey-ref-empty {
  padding: 10px;
  text-align: center;
  opacity: .7;
}
.zoey-ref-strip {
  flex: 0 0 64px;
  display: flex;
  align-items: center;
  gap: 6px;
  overflow-x: auto;
  padding: 4px 6px;
  box-sizing: border-box;
  border-top: 1px solid rgba(255,255,255,.06);
}
.zoey-ref-preview-item {
  position: relative;
  flex: 0 0 auto;
  width: 52px;
  height: 52px;
  border-radius: 5px;
  overflow: hidden;
  background: #111;
  border: 1px solid #333;
}
.zoey-ref-preview-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.zoey-ref-preview-tag {
  position: absolute;
  left: 2px;
  bottom: 2px;
  font-size: 9px;
  font-family: monospace;
  color: #fff;
  background: rgba(0, 0, 0, .65);
  border-radius: 3px;
  padding: 0 3px;
  line-height: 12px;
}
.zoey-ref-preview-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  font-size: 22px;
}
.zoey-ref-preview-hint {
  font-size: 11px;
  opacity: .6;
  padding: 4px 6px;
  white-space: nowrap;
}
/* ---- 富文本 prompt 编辑器（含下方预览条，合并为一个控件避免 widgets_values 错位） ---- */
.zoey-prompt-container {
  --comfy-widget-height: 170px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-sizing: border-box;
}
.zoey-prompt-editor {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  padding: 4px 6px;
  box-sizing: border-box;
  outline: none;
  font-size: var(--comfy-textarea-font-size, 12px);
  line-height: 1.5;
  color: var(--input-text, #ddd);
}
.zoey-prompt-editor:empty::before {
  content: attr(data-placeholder);
  color: #888;
  pointer-events: none;
}
.zoey-prompt-chip {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  vertical-align: middle;
  border-radius: 4px;
  padding: 1px 4px 1px 2px;
  margin: 0 1px;
  background: rgba(0, 0, 0, .45);
  border: 1px solid #444;
  font-size: 11px;
  line-height: 1;
  cursor: default;
}
.zoey-prompt-chip:hover {
  border-color: #7ec8ff;
}
.zoey-prompt-chip-media {
  width: 18px;
  height: 18px;
  border-radius: 3px;
  overflow: hidden;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  background: #000;
}
.zoey-prompt-chip-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.zoey-prompt-chip-tag {
  color: #7ec8ff;
  font-family: monospace;
}
/* ---- 时长滑动条（含快捷预设与状态行） ---- */
.zoey-duration {
  --comfy-widget-height: 96px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 4px 6px;
  box-sizing: border-box;
}
.zoey-duration-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.zoey-duration-row input[type="range"] {
  flex: 1;
  min-width: 0;
}
.zoey-duration-value {
  font-family: monospace;
  font-size: 12px;
  min-width: 40px;
  text-align: right;
  color: var(--input-text, #ddd);
}
.zoey-duration-presets {
  display: flex;
  gap: 4px;
}
.zoey-duration-presets button {
  flex: 1;
  font-size: 11px;
  padding: 2px 0;
  border-radius: 4px;
  border: 1px solid #444;
  background: rgba(255, 255, 255, .05);
  color: var(--input-text, #ddd);
  cursor: pointer;
}
.zoey-duration-presets button:hover {
  border-color: #7ec8ff;
}
.zoey-duration-status {
  font-size: 11px;
  opacity: .75;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--input-text, #ddd);
}
/* ---- 导演台（分镜列表） ---- */
.zoey-director {
  --comfy-widget-height: 260px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 4px 6px;
  box-sizing: border-box;
}
.zoey-director-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  font-weight: 600;
}
.zoey-director-header button {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid #444;
  background: rgba(255, 255, 255, .06);
  color: var(--input-text, #ddd);
  cursor: pointer;
}
.zoey-director-header button:hover {
  border-color: #7ec8ff;
}
.zoey-director-decl {
  width: 100%;
  min-height: 34px;
  max-height: 80px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  box-sizing: border-box;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .04);
  color: var(--input-text, #ddd);
  font-size: 11px;
  line-height: 1.5;
  padding: 3px 5px;
  outline: none;
}
.zoey-director-decl:empty::before {
  content: attr(data-placeholder);
  color: #888;
  pointer-events: none;
}
.zoey-director-list {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.zoey-director-shot {
  border: 1px solid #3a3a3a;
  border-radius: 5px;
  padding: 4px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  background: rgba(0, 0, 0, .25);
}
.zoey-director-shot-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.zoey-director-shot-head span {
  font-family: monospace;
  font-size: 11px;
  color: #7ec8ff;
}
.zoey-director-shot-head button {
  font-size: 11px;
  border: none;
  background: none;
  color: #f88;
  cursor: pointer;
  padding: 0 4px;
}
.zoey-director-shot-editor {
  width: 100%;
  min-height: 56px;
  max-height: 120px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  box-sizing: border-box;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .04);
  color: var(--input-text, #ddd);
  font-size: 12px;
  line-height: 1.5;
  padding: 3px 5px;
  outline: none;
}
.zoey-director-shot-editor:empty::before {
  content: attr(data-placeholder);
  color: #888;
  pointer-events: none;
}
.zoey-director-shot-dur {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
}
.zoey-director-shot-dur input {
  width: 52px;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .04);
  color: var(--input-text, #ddd);
  font-size: 12px;
  padding: 1px 4px;
}
.zoey-director-shot-trans {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .04);
  color: var(--input-text, #ddd);
  font-size: 11px;
  padding: 2px 5px;
}
.zoey-director-total {
  font-size: 11px;
  opacity: .8;
  color: var(--input-text, #ddd);
}
`;

// ---- 收集已连接的参考素材 ----
function getSourceInfo(input) {
  // input.link 是链接 ID（数字），链接对象在 graph.links 里
  const linkId = input?.link;
  if (linkId == null) return null;
  const link = app.graph?.links?.[linkId];
  if (!link) return null;
  const src = app.graph?.getNodeById(link.origin_id);
  if (!src) return null;
  let filename = null;
  for (const w of src.widgets || []) {
    const n = (w.name || "").toLowerCase();
    if (["image", "video", "audio", "file", "filename", "filename_abs"].includes(n)
        && typeof w.value === "string" && w.value) {
      filename = w.value;
      break;
    }
  }
  return { filename, srcType: src.comfyClass, src };
}

function buildViewUrl(filename) {
  if (!filename) return null;
  let type = "input", subfolder = "", name = filename.trim();
  const m = name.match(/^(.*)\s*\[(\w+)\]$/);
  if (m) { name = m[1].trim(); type = m[2]; }
  const idx = Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\"));
  if (idx >= 0) { subfolder = name.slice(0, idx); name = name.slice(idx + 1); }
  const params = new URLSearchParams({ filename: name, type });
  if (subfolder) params.set("subfolder", subfolder);
  params.set("t", String(Date.now()));
  return api.apiURL(`/view?${params.toString()}`);
}

function resolveThumbSrc(entry) {
  const imgs = entry?.src?.imgs;
  if (imgs?.length) {
    // 优先 src（加载未完成时 currentSrc 可能为空）
    const src = imgs[0]?.src || imgs[0]?.currentSrc;
    if (src) return src;
  }
  return buildViewUrl(entry?.filename);
}

function collectEntries(node) {
  const images = [], videos = [], audios = [], soundtracks = [];
  for (const input of node.inputs || []) {
    if (input.link == null) continue;
    const m = input.name?.match(/^ref_image_(\d+)$/);
    if (m) { images.push({ slot: +m[1], input }); continue; }
    const mv = input.name?.match(/^ref_video_(\d+)$/);
    if (mv) { videos.push({ slot: +mv[1], input }); continue; }
    const ms = input.name?.match(/^ref_video_audio_(\d+)$/);
    if (ms) { soundtracks.push({ slot: +ms[1], input }); continue; }
    const ma = input.name?.match(/^ref_audio_(\d+)$/);
    if (ma) { audios.push({ slot: +ma[1], input }); continue; }
  }
  const bySlot = (a, b) => a.slot - b.slot;
  images.sort(bySlot);
  videos.sort(bySlot);
  audios.sort(bySlot);
  soundtracks.sort(bySlot);

  const videoSlots = new Set(videos.map((v) => v.slot));
  const paired = soundtracks.filter((s) => videoSlots.has(s.slot));

  const entries = [];
  images.forEach((r, i) => entries.push(makeEntry(r, "图", "image", `@P${i + 1}`, null, i + 1)));
  videos.forEach((r, i) => entries.push(makeEntry(r, "视频", "video", `@V${i + 1}`, null, i + 1)));
  paired.forEach((r, i) => entries.push(makeEntry(r, "音频", "audio", `@A${i + 1}`, `视频${r.slot + 1} 音轨`, i + 1)));
  const audioStart = paired.length + 1;
  audios.forEach((r, i) => entries.push(makeEntry(r, "音频", "audio", `@A${audioStart + i}`, "独立音频", audioStart + i)));
  return entries;
}

function makeEntry(ref, kind, mediaType, tag, hint, num) {
  const src = getSourceInfo(ref.input);
  return { ref, kind, mediaType, tag, hint, num, filename: src?.filename ?? null, srcType: src?.srcType ?? null, src: src?.src ?? null };
}

// ---- 分辨率/时长计算（与后端 zoey_minimax_h3.py 保持一致） ----
const RESOLUTION_TABLE = {
  "608*352": [608, 352], "736*416": [736, 416], "864*480": [864, 480],
  "960*544": [960, 544], "1056*608": [1056, 608], "1152*640": [1152, 640],
  "1216*672": [1216, 672], "1280*736": [1280, 736], "1344*768": [1344, 768],
  "1376*768": [1376, 768], "1504*832": [1504, 832], "1664*928": [1664, 928],
  "1824*1024": [1824, 1024], "1920*1088": [1920, 1088],
};
const ASPECT_RATIO = { "16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1, "4:3": 4 / 3, "3:4": 3 / 4, "2:3": 2 / 3, "3:2": 3 / 2 };

function roundMultiple(v, m) {
  return Math.max(m, Math.round(v / m) * m);
}

function alignFrames(n) {
  while (n % 17 !== 5) n += 1;
  return n;
}

function frameCount(duration) {
  return alignFrames(Math.max(5, Math.round(duration * 24)));
}

function getFirstRefInfo(node) {
  for (const input of node.inputs || []) {
    if (input.link == null) continue;
    if (!/^ref_image_\d+$/.test(input.name || "")) continue;
    const info = getSourceInfo(input);
    const img = info?.src?.imgs?.[0];
    if (img && img.naturalWidth && img.naturalHeight) {
      const w = img.naturalWidth, h = img.naturalHeight;
      return { ratio: w / h, short: Math.min(w, h) };
    }
    return null;
  }
  return null;
}

function computeCanvas(resolution, aspect, refInfo) {
  const ratio = (aspect === "自动" || !ASPECT_RATIO[aspect])
    ? (refInfo?.ratio || 16 / 9)
    : ASPECT_RATIO[aspect];
  let cw, ch;
  if (RESOLUTION_TABLE[resolution]) {
    // 像素档：16:9 时精确用参考尺寸，其他比例按面积换算
    const [baseW, baseH] = RESOLUTION_TABLE[resolution];
    if (Math.abs(ratio - 16 / 9) < 1e-6) {
      cw = baseW; ch = baseH;
    } else {
      const area = baseW * baseH;
      cw = Math.sqrt(area * ratio);
      ch = Math.sqrt(area / ratio);
    }
  } else { // 自动
    const short = refInfo?.short || 720;
    cw = ratio >= 1 ? short * ratio : short;
    ch = ratio >= 1 ? short : short / ratio;
  }
  return [roundMultiple(cw, 32), roundMultiple(ch, 32)];
}

function countRefs(node) {
  let images = 0, videos = 0, audios = 0;
  for (const input of node.inputs || []) {
    if (input.link == null) continue;
    if (/^ref_image_\d+$/.test(input.name || "")) images += 1;
    else if (/^ref_video_\d+$/.test(input.name || "")) videos += 1;
    else if (/^ref_audio_\d+$/.test(input.name || "")) audios += 1;
  }
  return { images, videos, audios };
}

function entryThumb(entry) {
  const box = $el("div", { className: "zoey-ref-thumb" });
  const imgStyle = { width: "100%", height: "100%", objectFit: "cover", borderRadius: "4px", display: "block" };
  if (entry.mediaType === "image" || entry.mediaType === "video") {
    const url = resolveThumbSrc(entry);
    const fallbackIcon = entry.mediaType === "video" ? "🎬" : "🖼";
    if (url) {
      const img = $el("img", { style: imgStyle });
      img.onerror = () => { box.textContent = fallbackIcon; };
      img.src = url;
      box.appendChild(img);
    } else {
      box.textContent = fallbackIcon;
    }
  } else {
    box.textContent = "🔊";
  }
  return box;
}

// ---- 富文本 prompt 的 chip 渲染 / 串行化 ----
function makeChip(tag, entry) {
  const chip = document.createElement("span");
  chip.className = "zoey-prompt-chip";
  chip.contentEditable = "false";
  chip.dataset.tag = tag;
  chip.title = entry?.hint || tag;

  const media = document.createElement("span");
  media.className = "zoey-prompt-chip-media";
  if (entry?.mediaType === "audio") {
    media.textContent = "🔊";
  } else {
    const url = resolveThumbSrc(entry);
    if (url) {
      const img = document.createElement("img");
      img.className = "zoey-prompt-chip-img";
      img.onerror = () => { img.remove(); media.textContent = "🖼"; };
      img.src = url;
      media.appendChild(img);
    } else {
      media.textContent = entry?.mediaType === "video" ? "🎬" : "🖼";
    }
  }
  const label = document.createElement("span");
  label.className = "zoey-prompt-chip-tag";
  label.textContent = tag;
  chip.append(media, label);
  return chip;
}

function renderPrompt(el, text, node) {
  const entries = collectEntries(node);
  const byTag = new Map(entries.map((e) => [e.tag, e]));
  const frag = document.createDocumentFragment();
  const re = /(@[PpVvAa]\d+)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
    const tag = m[0].toUpperCase();
    frag.appendChild(makeChip(tag, byTag.get(tag)));
    last = m.index + m[0].length;
  }
  if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
  el.replaceChildren(frag);
}

function serializePrompt(el) {
  let out = "";
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    // 跳过 chip 里纯展示的媒体图标（🔊/🖼/🎬），只保留标签文本
    const p = n.parentElement;
    if (p && p.classList?.contains("zoey-prompt-chip-media")) continue;
    out += n.textContent;
  }
  return out;
}

function caretTexts(el) {
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount) return null;
  const range = sel.getRangeAt(0).cloneRange();
  if (!el.contains(range.commonAncestorContainer)) return null;
  const pre = document.createRange();
  pre.selectNodeContents(el);
  pre.setEnd(range.startContainer, range.startOffset);
  const post = document.createRange();
  post.selectNodeContents(el);
  post.setStart(range.endContainer, range.endOffset);
  return { before: pre.toString(), after: post.toString(), range };
}

// ---- 创建 contenteditable prompt 控件（含下方预览条，合并为一个控件） ----
function createPromptEditor(node, name, inputData) {
  const container = document.createElement("div");
  container.className = "zoey-prompt-container";

  const editor = document.createElement("div");
  editor.className = "zoey-prompt-editor";
  editor.contentEditable = "true";
  editor.spellcheck = false;
  editor.setAttribute("data-placeholder", "输入 @ 引用参考素材，如 @P1 的男人…");
  container.appendChild(editor);

  const strip = document.createElement("div");
  strip.className = "zoey-ref-strip";
  container.appendChild(strip);

  const widget = new DOMWidgetImpl({
    node,
    name,
    type: "customtext",
    element: container,
    options: {
      hideOnZoom: true,
      getValue: () => serializePrompt(editor),
      setValue: (v) => renderPrompt(editor, v ?? "", node),
    },
  });
  widget.inputEl = editor;
  widget.strip = strip;
  addWidget(node, widget);

  editor.addEventListener("paste", (e) => {
    e.preventDefault();
    const text = e.clipboardData?.getData("text/plain") ?? "";
    insertTextAtCaret(editor, text);
  });
  editor.addEventListener("drop", (e) => e.preventDefault());
  // Enter 的换行逻辑统一放在 RefPicker.#onKeyDown，避免重复插入

  return widget;
}

function insertTextAtCaret(editor, text) {
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount) return;
  const range = sel.getRangeAt(0);
  range.deleteContents();
  const node = document.createTextNode(text);
  range.insertNode(node);
  range.setStartAfter(node);
  range.collapse(true);
  sel.removeAllRanges();
  sel.addRange(range);
  editor.dispatchEvent(new Event("input", { bubbles: true }));
}

// ---- 时长滑动条控件（滑动条 + 快捷预设 + 状态行） ----
const DURATION_PRESETS = [3, 5, 8, 10, 15];

function createDurationWidget(node, name, inputData) {
  const container = document.createElement("div");
  container.className = "zoey-duration";

  const row = document.createElement("div");
  row.className = "zoey-duration-row";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "1";
  slider.max = "15";
  slider.step = "0.5";
  const valueLabel = document.createElement("span");
  valueLabel.className = "zoey-duration-value";
  row.append(slider, valueLabel);

  const presets = document.createElement("div");
  presets.className = "zoey-duration-presets";
  for (const s of DURATION_PRESETS) {
    const b = document.createElement("button");
    b.textContent = `${s}s`;
    b.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); setDuration(s); });
    presets.appendChild(b);
  }

  const status = document.createElement("div");
  status.className = "zoey-duration-status";

  container.append(row, presets, status);

  let current = parseFloat(inputData?.[1]?.default ?? 5) || 5;

  const clamp = (v) => (Number.isNaN(v) ? 5 : Math.min(15, Math.max(1, v)));

  const widget = new DOMWidgetImpl({
    node,
    name,
    type: "customtext",
    element: container,
    options: {
      hideOnZoom: true,
      getValue: () => current,
      setValue: (v) => { current = clamp(parseFloat(v)); syncUI(); refreshStatus(); },
    },
  });
  widget.inputEl = slider;
  addWidget(node, widget);

  function syncUI() {
    slider.value = String(current);
    valueLabel.textContent = `${current}s`;
  }

  function setDuration(v) {
    current = clamp(v);
    syncUI();
    refreshStatus();
  }

  slider.addEventListener("input", () => {
    current = clamp(parseFloat(slider.value));
    syncUI();
    refreshStatus();
  });

  function refreshStatus() {
    try {
      const res = node.widgets?.find((w) => w.name === "resolution");
      const asp = node.widgets?.find((w) => w.name === "aspect");
      const resolution = typeof res?.value === "string" ? res.value : "自动";
      const aspect = typeof asp?.value === "string" ? asp.value : "自动";
      const refInfo = getFirstRefInfo(node);
      const [w, h] = computeCanvas(resolution, aspect, refInfo);
      const frames = frameCount(current);
      const c = countRefs(node);
      status.textContent = `图×${c.images} 视频×${c.videos} 音频×${c.audios} ｜ ${w}×${h} ｜ ${current}s (${frames}帧)`;
    } catch (e) {
      console.error("[Zoey MiniMax H3] refreshStatus:", e);
    }
  }

  // 分辨率/比例变化时更新状态
  for (const w of node.widgets || []) {
    if (w.name === "resolution" || w.name === "aspect") {
      const prevCb = w.callback;
      w.callback = (value) => {
        try { prevCb?.(value); } catch (e) { console.error("[Zoey MiniMax H3] widget callback:", e); }
        refreshStatus();
      };
    }
  }

  // 连接变化（参考素材增删）时更新状态；用 .call(node) 保留 this
  const prevConn = node.onConnectionsChange;
  node.onConnectionsChange = (type, slot, isConnected, ...rest) => {
    try {
      prevConn?.call(node, type, slot, isConnected, ...rest);
    } catch (e) {
      console.error("[Zoey MiniMax H3] onConnectionsChange:", e);
    }
    refreshStatus();
  };

  syncUI();
  setTimeout(refreshStatus, 200);
  return widget;
}

// ---- 导演台（分镜列表）控件 ----
const MAX_TOTAL_SECONDS = 15;

function createDirectorPanel(node, name, inputData) {
  const container = document.createElement("div");
  container.className = "zoey-director";

  const header = document.createElement("div");
  header.className = "zoey-director-header";
  const title = document.createElement("span");
  title.textContent = "🎬 导演台 · 分镜";
  const addBtn = document.createElement("button");
  addBtn.textContent = "＋ 添加镜头";
  header.append(title, addBtn);

  const list = document.createElement("div");
  list.className = "zoey-director-list";

  const total = document.createElement("div");
  total.className = "zoey-director-total";

  container.append(header, list, total);

  // 参考素材说明：contenteditable 富文本，支持 @ 缩略图 chip 与下拉选择
  const declEditor = document.createElement("div");
  declEditor.className = "zoey-director-decl";
  declEditor.contentEditable = "true";
  declEditor.spellcheck = false;
  declEditor.setAttribute("data-placeholder", "参考素材说明（可空）：@P1 是人物参考，@P2 是场景参考…");
  header.after(declEditor);
  declEditor.addEventListener("paste", (e) => {
    e.preventDefault();
    insertTextAtCaret(declEditor, e.clipboardData?.getData("text/plain") ?? "");
  });
  declEditor.addEventListener("drop", (e) => e.preventDefault());

  let refDecl = "";
  let shots = [];
  let pickers = [];
  let declPicker = null;

  const widget = new DOMWidgetImpl({
    node,
    name,
    type: "customtext",
    element: container,
    options: {
      hideOnZoom: true,
      getValue: () => JSON.stringify({ ref_decl: refDecl, shots }),
      setValue: (v) => { const d = parseData(v); refDecl = d.refDecl; shots = d.shots; render(); },
    },
  });
  widget.inputEl = list;
  addWidget(node, widget);

  function parseData(v) {
    try {
      const d = JSON.parse(v || "");
      if (Array.isArray(d)) return { refDecl: "", shots: d }; // 兼容旧格式：纯镜头数组
      if (d && typeof d === "object") {
        return {
          refDecl: typeof d.ref_decl === "string" ? d.ref_decl : "",
          shots: Array.isArray(d.shots) ? d.shots : [],
        };
      }
    } catch (e) {}
    return { refDecl: "", shots: [] };
  }

  declPicker = new RefPicker(node, null, declEditor);
  declPicker.onChange = () => { refDecl = serializePrompt(declEditor); };

  function render() {
    pickers.forEach((p) => p.hide());
    declPicker?.hide();
    pickers = [];
    renderPrompt(declEditor, refDecl, node);
    list.replaceChildren();
    shots.forEach((shot, i) => list.appendChild(shotCard(shot, i)));
    updateTotal();
  }

  function shotCard(shot, i) {
    const card = document.createElement("div");
    card.className = "zoey-director-shot";

    const head = document.createElement("div");
    head.className = "zoey-director-shot-head";
    const label = document.createElement("span");
    label.textContent = `CUT ${i + 1}`;
    const del = document.createElement("button");
    del.textContent = "✕";
    del.title = "删除镜头";
    del.addEventListener("pointerdown", (e) => {
      e.preventDefault(); e.stopPropagation();
      shots.splice(i, 1);
      render();
    });
    head.append(label, del);

    // 镜头提示词：contenteditable 富文本，支持 @ 缩略图 chip 与下拉选择
    const editor = document.createElement("div");
    editor.className = "zoey-director-shot-editor";
    editor.contentEditable = "true";
    editor.spellcheck = false;
    editor.setAttribute("data-placeholder", "本镜头提示词，输入 @ 引用参考素材");
    renderPrompt(editor, shot.prompt || "", node);
    editor.addEventListener("paste", (e) => {
      e.preventDefault();
      insertTextAtCaret(editor, e.clipboardData?.getData("text/plain") ?? "");
    });
    editor.addEventListener("drop", (e) => e.preventDefault());

    const picker = new RefPicker(node, null, editor);
    picker.onChange = () => { shot.prompt = serializePrompt(editor); };
    pickers.push(picker);

    const durRow = document.createElement("div");
    durRow.className = "zoey-director-shot-dur";
    const durLbl = document.createElement("span");
    durLbl.textContent = "时长";
    const durInput = document.createElement("input");
    durInput.type = "number";
    durInput.min = "1"; durInput.max = String(MAX_TOTAL_SECONDS); durInput.step = "0.5";
    durInput.value = String(shot.duration ?? 5);
    durInput.addEventListener("input", () => {
      const v = parseFloat(durInput.value);
      shot.duration = Number.isNaN(v) ? 5 : Math.min(MAX_TOTAL_SECONDS, Math.max(1, v));
      updateTotal();
    });
    const durUnit = document.createElement("span");
    durUnit.textContent = "s";
    durRow.append(durLbl, durInput, durUnit);

    const trans = document.createElement("input");
    trans.className = "zoey-director-shot-trans";
    trans.value = shot.transition || "";
    trans.placeholder = "进入本镜头的转场（如 WHIP PAN，可留空）";
    trans.addEventListener("input", () => { shot.transition = trans.value; });

    card.append(head, editor, durRow, trans);
    return card;
  }

  function updateTotal() {
    const sum = shots.reduce((s, x) => s + (parseFloat(x.duration) || 0), 0);
    const capped = Math.min(sum, MAX_TOTAL_SECONDS);
    const frames = frameCount(capped);
    total.textContent = `总时长 ${sum}s → ${capped}s (${frames}帧)` + (sum > MAX_TOTAL_SECONDS ? "  ⚠ 超15s将截断" : "");
  }

  addBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault(); e.stopPropagation();
    shots.push({ prompt: "", duration: 5, transition: "" });
    render();
  });

  render();
  return widget;
}

function setupDirectorToggle(node, dirWidget) {
  const modeW = node.widgets?.find((w) => w.name === "director_mode");
  const promptW = node.widgets?.find((w) => w.name === "prompt");
  const durW = node.widgets?.find((w) => w.name === "duration");
  if (!modeW || !dirWidget || !promptW || !durW) return;

  const apply = () => {
    const on = !!modeW.value;
    [promptW, durW].forEach((w) => {
      w.hidden = on;
      if (w.element) w.element.style.display = on ? "none" : "";
    });
    dirWidget.hidden = !on;
    if (dirWidget.element) dirWidget.element.style.display = on ? "" : "none";
    try {
      const h = node.computeSize ? node.computeSize()[1] : node.size[1];
      node.setSize([node.size[0], h]);
      app.graph?.setDirtyCanvas(true);
    } catch (e) { /* 尺寸重算失败不影响功能 */ }
  };

  const prevCb = modeW.callback;
  modeW.callback = (v) => {
    try { prevCb?.(v); } catch (e) { console.error("[Zoey MiniMax H3] director_mode callback:", e); }
    apply();
  };

  apply();
  setTimeout(apply, 300); // 加载工作流后值可能被覆盖，再校正一次
}

// ---- @ 选择器 ----
class RefPicker {
  constructor(node, widget, editor) {
    this.node = node;
    this.widget = widget;
    this.editor = editor;
    this.dropdown = null;
    this.selected = 0;
    this.strip = null;
    this.onChange = null; // 内容变化回调（导演台镜头同步用）

    editor.addEventListener("keydown", (e) => this.#onKeyDown(e));
    editor.addEventListener("keyup", (e) => this.#onKeyUp(e));
    editor.addEventListener("input", () => { this.#update(); this.refreshPreview(); this.onChange?.(); });
    editor.addEventListener("click", () => this.#update());
    editor.addEventListener("blur", () => setTimeout(() => this.hide(), 150));
    document.addEventListener("pointerdown", (e) => {
      if (this.dropdown && !this.dropdown.contains(e.target)) this.hide();
    });
  }

  #currentToken() {
    const ct = caretTexts(this.editor);
    if (!ct || ct.range.startContainer.nodeType !== Node.TEXT_NODE) return null;
    const m = ct.before.match(/(@[\p{L}\p{N}]*)$/u);
    if (!m) return null;
    // 确认 token 末尾确实落在当前文本节点里（避免在 chip 之后误触发）
    const cur = ct.range.startContainer.textContent.slice(0, ct.range.startOffset);
    if (!cur.endsWith(m[1])) return null;
    return { text: m[1], before: ct.before, after: ct.after, startOffset: ct.range.startOffset - m[1].length };
  }

  #onKeyDown(e) {
    if (this.dropdown && this.entries?.length) {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          this.selected = Math.min(this.selected + 1, this.entries.length - 1);
          this.#refreshSelection();
          return;
        case "ArrowUp":
          e.preventDefault();
          this.selected = Math.max(this.selected - 1, 0);
          this.#refreshSelection();
          return;
        case "Enter":
        case "Tab":
          e.preventDefault();
          this.#select(this.entries[this.selected]);
          return;
        case "Escape":
          e.preventDefault();
          this.hide();
          return;
      }
    }
    if (e.key === "Enter") {
      // 未走选择逻辑时，回车=换行（统一在此处理，避免重复插入）
      e.preventDefault();
      this.hide();
      insertTextAtCaret(this.editor, "\n");
    }
  }

  #onKeyUp(e) {
    if (this.dropdown && e.key === "Escape") this.hide();
  }

  #update() {
    const token = this.#currentToken();
    if (!token) { this.hide(); return; }
    const typed = token.text.slice(1).toLowerCase();
    const all = collectEntries(this.node);
    // 类型过滤：@P/@图 -> 图片，@V/@视频 -> 视频，@A/@音频 -> 音频；也按标签/提示匹配
    const filtered = typed ? all.filter((en) =>
      en.tag.toLowerCase().includes(typed) ||
      en.kind.includes(typed) ||
      (en.hint ?? "").toLowerCase().includes(typed)
    ) : all;
    this.entries = filtered;
    this.selected = 0;
    this.#show();
  }

  #show() {
    if (!this.dropdown) {
      this.dropdown = $el("div", { className: "zoey-ref-picker" });
      this.dropdown.addEventListener("pointerdown", (e) => e.stopPropagation());
      document.body.appendChild(this.dropdown);
    }
    this.#render();
  }

  #render() {
    if (!this.dropdown) return;
    this.#position();
    if (!this.entries?.length) {
      const hasAny = (this.node?.inputs || []).some((i) => i.link != null);
      this.dropdown.replaceChildren($el("div", {
        className: "zoey-ref-empty",
        textContent: hasAny ? "没有匹配的参考素材" : "未连接参考素材，请先在节点上连接图片/视频/音频",
      }));
      return;
    }
    const header = $el("div", {
      className: "zoey-ref-header",
      textContent: "@ 选择参考素材（@P 图 / @V 视频 / @A 音频）",
    });
    const items = this.entries.map((en, i) => {
      const meta = $el("div", { className: "zoey-ref-meta" }, [
        $el("div", { className: "zoey-ref-label", textContent: `${en.kind} ${en.num}` }),
        en.hint ? $el("div", { className: "zoey-ref-hint", textContent: en.hint }) : null,
        $el("div", { className: "zoey-ref-tag", textContent: en.tag }),
      ]);
      const item = $el("div", {
        className: "zoey-ref-item" + (i === this.selected ? " zoey-ref-item--selected" : ""),
        onclick: () => this.#select(en),
        onmouseenter: () => {
          if (this.selected !== i) {
            this.selected = i;
            this.#refreshSelection();
          }
        },
      }, [entryThumb(en), meta]);
      return item;
    });
    this.dropdown.replaceChildren(header, ...items);
  }

  #refreshSelection() {
    if (!this.dropdown) return;
    this.dropdown.querySelectorAll(".zoey-ref-item").forEach((el, i) => {
      el.classList.toggle("zoey-ref-item--selected", i === this.selected);
    });
  }

  #position() {
    const sel = window.getSelection();
    const rect = sel?.rangeCount ? sel.getRangeAt(0).getBoundingClientRect() : null;
    if (!rect || (!rect.width && !rect.height)) return;
    this.dropdown.style.left = `${rect.left}px`;
    this.dropdown.style.top = `${rect.bottom + 4}px`;
  }

  #select(entry) {
    const token = this.#currentToken();
    this.hide();
    if (!token) return;
    const sel = window.getSelection();
    if (!sel?.rangeCount) return;
    // 重新解析该标签对应的最新条目，避免用下拉打开瞬间收集的过期 src.imgs
    const fresh = collectEntries(this.node).find((e) => e.tag === entry.tag) || entry;
    const range = sel.getRangeAt(0);
    // 删除当前文本节点里的 @... token，原位插入 chip
    range.setStart(range.startContainer, token.startOffset);
    range.deleteContents();
    const chip = makeChip(fresh.tag, fresh);
    range.insertNode(chip);
    const caret = document.createRange();
    caret.setStartAfter(chip);
    caret.collapse(true);
    sel.removeAllRanges();
    sel.addRange(caret);
    this.editor.focus();
    this.refreshPreview();
    this.onChange?.();
  }

  attachStrip(strip) {
    this.strip = strip;
    this.refreshPreview();
  }

  refreshPreview() {
    if (!this.strip) return;
    const text = this.widget?.value ?? "";
    const used = new Set();
    const re = /@[PpVvAa]\d+/g;
    let m;
    while ((m = re.exec(text))) used.add(m[0].toUpperCase());
    const entries = collectEntries(this.node);
    const byTag = new Map(entries.map((e) => [e.tag, e]));
    const items = [...used].map((tag) => byTag.get(tag)).filter(Boolean);
    if (!items.length) {
      const hasAny = (this.node?.inputs || []).some((i) => i.link != null);
      this.strip.replaceChildren($el("div", {
        className: "zoey-ref-preview-hint",
        textContent: hasAny ? "提示词里输入 @ 后选择参考素材，预览会显示在这里" : "未连接参考素材",
      }));
      return;
    }
    this.strip.replaceChildren(...items.map((it) => this.#previewItem(it)));
  }

  #previewItem(entry) {
    const tag = $el("span", { className: "zoey-ref-preview-tag", textContent: entry.tag });
    if (entry.mediaType === "audio") {
      return $el("div", { className: "zoey-ref-preview-item", title: entry.hint || entry.tag }, [
        $el("span", { className: "zoey-ref-preview-icon", textContent: "🔊" }),
        tag,
      ]);
    }
    const url = resolveThumbSrc(entry);
    let media;
    if (url) {
      const img = $el("img", { className: "zoey-ref-preview-img" });
      img.onerror = () => {
        const icon = $el("span", { className: "zoey-ref-preview-icon", textContent: entry.mediaType === "video" ? "🎬" : "🖼" });
        img.replaceWith(icon);
      };
      img.src = url;
      media = img;
    } else {
      media = $el("span", { className: "zoey-ref-preview-icon", textContent: entry.mediaType === "video" ? "🎬" : "🖼" });
    }
    return $el("div", { className: "zoey-ref-preview-item", title: entry.hint || entry.tag }, [media, tag]);
  }

  hide() {
    if (this.dropdown) {
      this.dropdown.remove();
      this.dropdown = null;
    }
  }
}

// ---- 包装 STRING/FLOAT：prompt 换成富文本编辑器（含预览条），duration 换成秒计滑动条 ----
function patchWidgets() {
  if (ComfyWidgets.STRING.__zoeyRefPickerWrapped) return;
  ComfyWidgets.STRING.__zoeyRefPickerWrapped = true;

  const origString = ComfyWidgets.STRING;
  ComfyWidgets.STRING = function (node, inputName, inputData, opts) {
    if (node?.comfyClass === NODE_TYPE && inputName === "prompt") {
      const editorWidget = createPromptEditor(node, inputName, inputData);
      attachPicker(node, editorWidget);
      return { widget: editorWidget };
    }
    if (node?.comfyClass === NODE_TYPE && inputName === "director_shots") {
      const dirWidget = createDirectorPanel(node, inputName, inputData);
      setupDirectorToggle(node, dirWidget);
      return { widget: dirWidget };
    }
    return origString.apply(this, arguments);
  };

  const origFloat = ComfyWidgets.FLOAT;
  ComfyWidgets.FLOAT = function (node, inputName, inputData, opts) {
    if (node?.comfyClass === NODE_TYPE && inputName === "duration") {
      return { widget: createDurationWidget(node, inputName, inputData) };
    }
    return origFloat.apply(this, arguments);
  };
}

function attachPicker(node, widget) {
  const tryAttach = () => {
    const el = widget.inputEl || widget.element;
    if (!el) return false;
    const picker = new RefPicker(node, widget, el);
    if (widget.strip) picker.attachStrip(widget.strip);

    const refresh = () => { try { picker.refreshPreview(); } catch (e) { console.error("[Zoey MiniMax H3] refreshPreview:", e); } };
    // 注意：前端 onConfigure/onConnectionsChange 内部依赖 this === node，
    // 必须用 .call(node, ...) 保留 this，否则会因 this 丢失而崩溃（如读 this.has_errors）
    const prevConn = node.onConnectionsChange;
    node.onConnectionsChange = (type, slot, isConnected, ...rest) => {
      try {
        prevConn?.call(node, type, slot, isConnected, ...rest);
      } catch (e) {
        console.error("[Zoey MiniMax H3] onConnectionsChange:", e);
      }
      refresh();
    };
    const prevConfigure = node.onConfigure;
    node.onConfigure = (...args) => {
      try {
        prevConfigure?.call(node, ...args);
      } catch (e) {
        console.error("[Zoey MiniMax H3] onConfigure:", e);
      }
      refresh();
    };
    setTimeout(refresh, 200);

    const prevOnRemoved = node.onRemoved;
    node.onRemoved = () => { picker.hide(); prevOnRemoved?.call(node); };
    return true;
  };
  if (tryAttach()) return;
  let tries = 0;
  const iv = setInterval(() => {
    tries += 1;
    if (tryAttach() || tries > 30) clearInterval(iv);
  }, 100);
}

app.registerExtension({
  name: EXT_NAME,
  init() {
    $el("style", { textContent: STYLE, parent: document.head });
    patchWidgets();
  },
});
