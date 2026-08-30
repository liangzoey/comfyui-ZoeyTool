// Zoey - MiniMax H3 参考转视频 (@) 前端扩展
// prompt 为 contenteditable 富文本：@P1/@V1/@A1 渲染成带缩略图的 chip（图片/音频/视频），
// 输入 @ 弹出选择器选择已连接参考素材；串行化回 @P1 文本交给后端。
import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import { ComfyWidgets } from "../../../../scripts/widgets.js";
import { addWidget, DOMWidgetImpl } from "../../../../scripts/domWidget.js";
import { $el } from "../../../../scripts/ui.js";

// ---- 永久全局素材库（跨工作流，存 <ComfyUI>/input/zoey_library/） ----
let globalLibrary = [];
let globalLibraryLoaded = false;
let globalLibraryLoading = null;

async function loadGlobalLibrary() {
  if (globalLibraryLoading) return globalLibraryLoading;
  globalLibraryLoading = (async () => {
    try {
      const r = await api.fetchApi("/zoey/library");
      const d = await r.json();
      globalLibrary = Array.isArray(d.entries) ? d.entries : [];
    } catch (e) {
      console.error("[Zoey MiniMax H3] loadGlobalLibrary:", e);
      globalLibrary = [];
    }
    globalLibraryLoaded = true;
    globalLibraryLoading = null;
    document.dispatchEvent(new CustomEvent("zoey:library-loaded"));
    return globalLibrary;
  })();
  return globalLibraryLoading;
}

function libraryMediaUrl(file) {
  if (!file) return null;
  const p = new URLSearchParams({ filename: file, type: "input", subfolder: "zoey_library" });
  p.set("t", String(Date.now()));
  return api.apiURL(`/view?${p.toString()}`);
}

let saveLibraryTimer = null;
async function saveLibrary() {
  try {
    await api.fetchApi("/zoey/library/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entries: globalLibrary }),
    });
  } catch (e) {
    console.error("[Zoey MiniMax H3] saveLibrary:", e);
  }
}
function scheduleSaveLibrary() {
  if (saveLibraryTimer) clearTimeout(saveLibraryTimer);
  saveLibraryTimer = setTimeout(saveLibrary, 500);
}

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
  flex: 0 0 auto;
  display: flex;
  align-items: flex-start;
  gap: 6px;
  overflow-x: auto;
  padding: 4px 6px;
  box-sizing: border-box;
  border-top: 1px solid rgba(255,255,255,.06);
}
.zoey-ref-preview-wrap {
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1px;
}
.zoey-ref-purpose {
  width: 52px;
}
/* 自定义下拉按钮（替代原生 select，Chrome 深色主题下原生选项看不清） */
.zoey-pop-select {
  font-size: 10px;
  font-weight: 600;
  border: 1px solid #8a8a8a;
  border-radius: 4px;
  background: #2e2e2e;
  color: #f0f0f0;
  cursor: pointer;
  padding: 1px 4px;
  box-sizing: border-box;
  white-space: nowrap;
  transition: border-color .15s ease, color .15s ease;
}
.zoey-pop-select:hover {
  border-color: #7ec8ff;
  color: #fff;
}
/* ---- 简单模式编译预览 ---- */
.zoey-preview-row {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  padding: 2px 4px 0;
}
.zoey-preview-btn {
  font-size: 10px;
  padding: 1px 8px;
  border-radius: 4px;
  border: 1px solid #444;
  background: rgba(255, 255, 255, .05);
  color: var(--input-text, #ddd);
  cursor: pointer;
  transition: all .15s ease;
}
.zoey-preview-btn:hover,
.zoey-preview-btn.active {
  border-color: #7ec8ff;
  color: #7ec8ff;
}
.zoey-simple-preview {
  flex: 0 0 auto;
  width: 100%;
  height: 88px;
  box-sizing: border-box;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(0, 0, 0, .35);
  color: #9ec7ff;
  font-size: 10px;
  font-family: monospace;
  line-height: 1.4;
  padding: 3px 5px;
}
.zoey-preview-meta {
  flex: 0 0 auto;
  font-size: 10px;
  opacity: .8;
  color: var(--input-text, #ddd);
  padding: 2px 4px;
}
.zoey-preview-meta.warn {
  color: #f88;
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
/* ---- @ 缩略图悬停大图预览 ---- */
.zoey-ref-hover {
  position: fixed;
  z-index: 999999; /* 高于 @ 选择下拉(99999) */
  pointer-events: none;
  display: none;
  background: rgba(15, 15, 15, .96);
  border: 1px solid var(--border-color, #555);
  border-radius: 6px;
  box-shadow: 0 6px 24px rgba(0, 0, 0, .6);
  max-width: min(46vw, 420px);
  max-height: min(56vh, 440px);
  overflow: hidden;
}
.zoey-ref-hover-img {
  display: block;
  max-width: min(46vw, 420px);
  max-height: min(56vh, 440px);
  object-fit: contain;
}
.zoey-ref-hover-audio {
  display: none;
  align-items: center;
  justify-content: center;
  width: 240px;
  height: 140px;
  font-size: 40px;
  color: #aaa;
  user-select: none;
}
.zoey-ref-hover-video {
  display: none;
  max-width: min(46vw, 420px);
  max-height: min(56vh, 440px);
  object-fit: contain;
  pointer-events: auto;
  cursor: pointer;
}
.zoey-ref-hover-audio-el {
  display: none;
  width: 300px;
  max-width: min(46vw, 420px);
  margin: 12px 14px;
  pointer-events: auto;
}
/* ---- 富文本 prompt 编辑器（创作区卡片：顶部模式条 + 编辑框 + 预览条，合并为一个控件避免 widgets_values 错位） ---- */
.zoey-prompt-container {
  --comfy-widget-height: 280px;
  height: var(--comfy-widget-height);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-sizing: border-box;
  background: rgba(126, 200, 255, .045);
  border: 1px solid rgba(126, 200, 255, .16);
  border-radius: 6px;
}
/* 模式切换：参考 / T2V / I2V / 自动（创作区顶部分段开关） */
.zoey-mode-bar {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 3px;
  padding: 4px 6px;
  border-bottom: 1px solid rgba(126, 200, 255, .10);
  background: rgba(126, 200, 255, .05);
}
.zoey-mode-bar .mode-label {
  font-size: 10px;
  font-weight: 600;
  color: #7ec8ff;
  letter-spacing: 1px;
  margin-right: 4px;
  flex: 0 0 auto;
}
.zoey-mode-btn {
  flex: 1;
  min-width: 0;
  font-size: 11px;
  padding: 2px 0;
  border-radius: 4px;
  border: 1px solid #3a3a3a;
  background: rgba(255, 255, 255, .04);
  color: #bbb;
  cursor: pointer;
  text-align: center;
  user-select: none;
  transition: all .15s ease;
}
.zoey-mode-btn:hover {
  border-color: #7ec8ff;
  color: #fff;
}
.zoey-mode-btn.active {
  border-color: #7ec8ff;
  background: rgba(126, 200, 255, .18);
  color: #7ec8ff;
  font-weight: 600;
  box-shadow: 0 0 0 1px rgba(126, 200, 255, .25);
}
/* mode 值持有控件：隐藏（UI 顶部模式条控制），仅用于后端取值的序列化 */
.zoey-mode-holder {
  --comfy-widget-height: 0px;
  --comfy-widget-min-height: 0px;
  --comfy-widget-max-height: 0px;
  display: none;
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
  border: 1px solid #666;
  border-radius: 4px;
  background: rgba(0, 0, 0, .30);
  font-size: var(--comfy-textarea-font-size, 12px);
  line-height: 1.5;
  color: var(--input-text, #ddd);
}
.zoey-prompt-editor:focus {
  border-color: #7ec8ff;
  box-shadow: 0 0 0 1px rgba(126, 200, 255, .30);
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
/* ---- 时长滑动条（滑杆行内嵌状态小字 + 快捷预设） ---- */
.zoey-duration {
  --comfy-widget-height: 62px;
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
  flex: 1;
  min-width: 0;
  font-size: 10px;
  opacity: .7;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: right;
  color: var(--input-text, #ddd);
}
/* ---- 导演台（分镜列表） ---- */
.zoey-director {
  --comfy-widget-height: 380px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 4px 6px;
  box-sizing: border-box;
  overflow-y: auto;
  background: rgba(255, 255, 255, .025);
  border: 1px solid rgba(255, 255, 255, .07);
  border-radius: 6px;
}
.zoey-director-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  font-weight: 600;
  color: #7ec8ff;
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
.zoey-loras-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 2px 0;
}
.zoey-lora-row {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
}
.zoey-lora-row select {
  flex: 1 1 auto;
  min-width: 0;
  font-size: 11px;
  padding: 1px 4px;
  background: var(--comfy-input-bg, #222);
  color: var(--input-text, #ddd);
  border: 1px solid #444;
  border-radius: 4px;
}
.zoey-lora-row label {
  white-space: nowrap;
  opacity: .75;
}
.zoey-lora-row input[type=number] {
  width: 58px;
  font-size: 11px;
  padding: 1px 3px;
  background: var(--comfy-input-bg, #222);
  color: var(--input-text, #ddd);
  border: 1px solid #444;
  border-radius: 4px;
}
.zoey-lora-row .zoey-lora-del {
  flex: 0 0 auto;
  font-size: 11px;
  padding: 1px 5px;
  border-radius: 4px;
  border: 1px solid #444;
  background: rgba(255, 255, 255, .06);
  color: var(--input-text, #ddd);
  cursor: pointer;
}
.zoey-lora-del:hover {
  border-color: #ff7e7e;
  color: #ff7e7e;
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
  flex: 0 0 auto;
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
/* ---- 导演台增强：景别/运镜按钮、角色槽、说话人、对白、音效配乐、编译预览 ---- */
.zoey-director-camrow {
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
}
.zoey-director-camrow .cam-label {
  font-size: 10px;
  opacity: .6;
  align-self: center;
  margin-right: 2px;
}
.zoey-director-camrow button {
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 3px;
  border: 1px solid #3a3a3a;
  background: rgba(255, 255, 255, .05);
  color: var(--input-text, #ddd);
  cursor: pointer;
}
.zoey-director-camrow button:hover {
  border-color: #7ec8ff;
}
.zoey-director-chars {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.zoey-director-char {
  display: flex;
  align-items: center;
  gap: 4px;
}
.zoey-director-char-btn {
  width: 34px;
  height: 34px;
  border-radius: 5px;
  overflow: hidden;
  border: 1px solid #3a3a3a;
  background: rgba(255, 255, 255, .05);
  color: var(--input-text, #ddd);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-family: monospace;
  flex: 0 0 34px;
  box-sizing: border-box;
  padding: 0;
}
.zoey-director-char-btn img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.zoey-director-char-btn:hover {
  border-color: #7ec8ff;
}
.zoey-director-char input {
  width: 72px;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .04);
  color: var(--input-text, #ddd);
  font-size: 11px;
  padding: 1px 4px;
}
.zoey-director-mini-btn {
  font-size: 10px;
  padding: 0 4px;
  border-radius: 3px;
  border: 1px solid #444;
  background: rgba(255, 255, 255, .05);
  color: var(--input-text, #ddd);
  cursor: pointer;
  line-height: 14px;
}
.zoey-director-mini-btn:hover {
  border-color: #7ec8ff;
}
.zoey-director-collapse {
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .03);
}
.zoey-director-collapse-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11px;
  padding: 2px 4px;
  cursor: pointer;
  user-select: none;
}
.zoey-director-collapse-head .caret {
  color: #7ec8ff;
}
.zoey-director-collapse-body {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 2px 4px 4px;
}
.zoey-director-speaker {
  display: flex;
  align-items: center;
  gap: 4px;
}
.zoey-director-speaker .zoey-director-spk-id {
  font-family: monospace;
  font-size: 11px;
  color: #7ec8ff;
  width: 28px;
  flex: 0 0 28px;
}
.zoey-director-speaker input {
  flex: 1;
  min-width: 0;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .04);
  color: var(--input-text, #ddd);
  font-size: 11px;
  padding: 1px 4px;
}
.zoey-director-dlg {
  display: flex;
  align-items: center;
  gap: 3px;
}
.zoey-director-dlg select,
.zoey-director-dlg input {
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .04);
  color: var(--input-text, #ddd);
  font-size: 11px;
  padding: 1px 3px;
}
.zoey-director-dlg .spk {
  width: 50px;
}
.zoey-director-dlg .lang {
  width: 66px;
}
.zoey-director-dlg .txt {
  flex: 1;
  min-width: 0;
}
.zoey-director-dlg .del {
  color: #f88;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 11px;
}
.zoey-director-preview {
  width: 100%;
  min-height: 56px;
  max-height: 120px;
  box-sizing: border-box;
  resize: vertical;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(0, 0, 0, .35);
  color: #9ec7ff;
  font-size: 10px;
  font-family: monospace;
  line-height: 1.4;
  padding: 3px 5px;
}
.zoey-director-field {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.zoey-director-field label {
  font-size: 10px;
  opacity: .7;
}
.zoey-director-field input {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .04);
  color: var(--input-text, #ddd);
  font-size: 11px;
  padding: 1px 5px;
}
.zoey-director-dlgadd {
  align-self: flex-start;
  font-size: 10px;
  padding: 0 5px;
  border-radius: 3px;
  border: 1px dashed #555;
  background: none;
  color: var(--input-text, #ddd);
  cursor: pointer;
}
.zoey-director-dlgadd:hover {
  border-color: #7ec8ff;
}
.zoey-director-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
}
.zoey-director-toolbar .tool-label {
  font-size: 10px;
  opacity: .6;
}
.zoey-director-toolbar button {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  border: 1px solid #3a3a3a;
  background: rgba(255, 255, 255, .05);
  color: var(--input-text, #ddd);
  cursor: pointer;
  line-height: 14px;
}
.zoey-director-toolbar button:hover {
  border-color: #7ec8ff;
}
.zoey-director-toggle {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 11px;
  color: var(--input-text, #ddd);
  cursor: pointer;
  user-select: none;
}
.zoey-director-toggle input {
  accent-color: #7ec8ff;
  margin: 0;
}
/* ---- 素材库（角色/道具/场景/音频） ---- */
.zoey-library {
  --comfy-widget-height: 150px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 4px 6px;
  box-sizing: border-box;
  background: rgba(255, 255, 255, .025);
  border: 1px solid rgba(255, 255, 255, .07);
  border-radius: 6px;
}
.zoey-library-list {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.zoey-library-row {
  display: flex;
  align-items: center;
  gap: 3px;
}
.zoey-library-row select {
  width: 44px;
  border: 1px solid #8a8a8a;
  border-radius: 4px;
  background: #2e2e2e;
  color: #f0f0f0;
  font-size: 12px;
  font-weight: 600;
  padding: 1px 2px;
}
.zoey-library-slot {
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #3a3a3a;
  background: rgba(255, 255, 255, .05);
  color: var(--input-text, #ddd);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-family: monospace;
  padding: 0;
  box-sizing: border-box;
}
.zoey-library-slot img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.zoey-library-slot:hover {
  border-color: #7ec8ff;
}
.zoey-library-row input {
  flex: 1;
  min-width: 0;
  border: 1px solid #333;
  border-radius: 4px;
  background: rgba(255, 255, 255, .04);
  color: var(--input-text, #ddd);
  font-size: 11px;
  padding: 1px 4px;
}
.zoey-library-row input.lib-desc {
  flex: 1.2;
}
.zoey-library-hint {
  font-size: 11px;
  opacity: .6;
  padding: 4px 2px;
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
  // 视频文件本身不能当 <img> 渲染（会裂图），留给播放/图标兜底
  if (entry?.mediaType === "video") return null;
  return buildViewUrl(entry?.filename);
}

// 媒体播放 URL（真正的视频/音频文件，供 <video>/<audio> 播放）
function mediaFileUrl(entry) {
  if (!entry || !entry.filename) return null;
  return buildViewUrl(entry.filename);
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

  // 角色槽 @C（导演台定义）：条目指向其分配的参考图，供 chip/下拉/悬停预览使用
  const chars = node._zoeyDirector?.characters || [];
  chars.forEach((ch, i) => {
    if (!ch || ch.slot == null) return;
    const refEntry = entries.find((e) => e.slot === ch.slot && e.mediaType === "image");
    const name = (ch.name || "").trim();
    if (refEntry) {
      entries.push({ ...refEntry, kind: "角色", mediaType: "image", tag: `@C${i + 1}`, hint: name || `角色 ${i + 1}`, num: i + 1 });
    } else {
      entries.push({ ref: null, kind: "角色", mediaType: "image", tag: `@C${i + 1}`, hint: (name || `角色 ${i + 1}`) + "（未连接图片）", num: i + 1, slot: null, filename: null, srcType: null, src: null });
    }
  });

  // 素材库 @L（永久全局：角色/道具/场景/音频）
  const lib = globalLibrary;
  lib.forEach((entry, i) => {
    if (!entry || typeof entry !== "object") return;
    const isAudio = entry.kind === "audio";
    const name = (entry.name || "").trim();
    const desc = (entry.desc || "").trim();
    const file = isAudio ? (entry.audio_file || entry.file) : entry.file;
    if (!file) {
      entries.push({
        ref: null, kind: "素材", mediaType: isAudio ? "audio" : "image", tag: `@L${i + 1}`,
        hint: (name || `素材 ${i + 1}`) + "（未上传）", num: i + 1,
        slot: null, filename: null, srcType: null, src: null,
      });
      return;
    }
    entries.push({
      ref: null, kind: "素材", mediaType: isAudio ? "audio" : "image", tag: `@L${i + 1}`,
      hint: (name || `素材 ${i + 1}`) + (desc ? ` · ${desc}` : ""), num: i + 1,
      slot: null, filename: `zoey_library/${file}`, srcType: null, src: null,
    });
  });
  return entries;
}

function makeEntry(ref, kind, mediaType, tag, hint, num) {
  const src = getSourceInfo(ref.input);
  const m = (ref.input?.name || "").match(/^(ref_image|ref_video|ref_audio|ref_video_audio)_(\d+)$/);
  const slot = m ? +m[2] : null;
  return { ref, kind, mediaType, tag, hint, num, slot, filename: src?.filename ?? null, srcType: src?.srcType ?? null, src: src?.src ?? null };
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
  attachHover(box, entry);
  return box;
}

// ---- @ 缩略图悬停预览：图片 / 可播放视频 / 可播放音频 ----
// 框一次定位不再跟随光标（否则光标永远追不上框，无法点播放控件）；
// 离开源元素后延迟关闭，光标移入框内则保持打开。
const HoverPreview = (() => {
  let el, img, videoEl, audioEl, audioBox, visible = false;
  let lastX = 0, lastY = 0, hideTimer = null;
  // 所有绑定悬停预览的源元素。mousemove 用它判断光标是否还"在素材/预览上"，
  // 而不是只依赖源元素的 mouseleave——后者在光标滑到预览本体(pointer-events:none)
  // 或源被替换时不会触发，导致预览卡住不消失。
  const hoverSources = new WeakSet();
  function ensure() {
    if (el) return;
    el = $el("div", { className: "zoey-ref-hover" });
    img = $el("img", { className: "zoey-ref-hover-img" });
    videoEl = $el("video", { className: "zoey-ref-hover-video", muted: true, loop: true, autoplay: true, playsinline: true, controls: true });
    audioEl = $el("audio", { className: "zoey-ref-hover-audio-el", controls: true });
    audioBox = $el("div", { className: "zoey-ref-hover-audio", textContent: "🔊" });
    el.append(img, videoEl, audioEl, audioBox);
    document.body.appendChild(el);
    document.addEventListener("mousemove", (e) => {
      lastX = e.clientX; lastY = e.clientY;
      if (!visible) return;
      // 光标是否在预览本体上（pointer-events:none 的覆盖层 / 可交互的视频·音频控件）
      const r = el.getBoundingClientRect();
      const overOverlay = e.clientX >= r.left - 4 && e.clientX <= r.right + 4
        && e.clientY >= r.top - 4 && e.clientY <= r.bottom + 4;
      // 光标是否在某个悬停源（或其子元素）上——elementFromPoint 不受 pointer-events:none 影响
      let overSource = false;
      const t = document.elementFromPoint(e.clientX, e.clientY);
      for (let n = t; n; n = n.parentElement) {
        if (hoverSources.has(n)) { overSource = true; break; }
      }
      if (overOverlay || overSource) cancelHide();
      else leave(); // 光标既不在预览也不在素材上：安排关闭，别再只 cancelHide 而永不隐藏
    });
    document.addEventListener("pointerdown", () => hide());
  }
  function place() {
    const pad = 14;
    let x = lastX + pad, y = lastY + pad;
    const r = el.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = lastX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = lastY - r.height - pad;
    el.style.left = `${Math.max(8, x)}px`;
    el.style.top = `${Math.max(8, y)}px`;
  }
  function cancelHide() { if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; } }
  function stopPlayback() {
    try { videoEl?.pause(); if (videoEl) videoEl.currentTime = 0; } catch (e) {}
    try { audioEl?.pause(); if (audioEl) audioEl.currentTime = 0; } catch (e) {}
  }
  function hide() {
    cancelHide();
    visible = false;
    stopPlayback();
    if (el) el.style.display = "none";
  }
  function leave() { if (!visible) return; cancelHide(); hideTimer = setTimeout(hide, 300); }
  return {
    show(entry, x, y) {
      ensure();
      cancelHide();
      lastX = x ?? lastX; lastY = y ?? lastY;
      visible = true;
      stopPlayback();
      img.style.display = "none";
      videoEl.style.display = "none";
      audioEl.style.display = "none";
      audioBox.style.display = "none";
      const url = mediaFileUrl(entry);
      if (entry?.mediaType === "audio") {
        if (url) {
          audioEl.src = url;
          audioEl.style.display = "block";
        } else {
          audioBox.style.display = "flex";
        }
      } else if (entry?.mediaType === "video") {
        if (url) {
          videoEl.src = url;
          videoEl.style.display = "block";
          videoEl.play().catch(() => {});
        } else {
          const turl = resolveThumbSrc(entry);
          if (turl) { img.src = turl; img.style.display = "block"; }
        }
      } else {
        const turl = resolveThumbSrc(entry);
        if (turl) { img.src = turl; img.style.display = "block"; }
      }
      el.style.display = "block";
      place();
    },
    hide,
    leave,
    trackSource(elNode) { hoverSources.add(elNode); },
  };
})();

// 给任意缩略图元素绑定悬停预览（chip / 预览条 / @ 下拉条目）
function attachHover(elNode, entry) {
  HoverPreview.trackSource(elNode);
  elNode.addEventListener("mouseenter", (e) => {
    HoverPreview.show(entry, e.clientX, e.clientY);
  });
  elNode.addEventListener("mouseleave", () => HoverPreview.leave());
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
  if (entry) attachHover(chip, entry);
  return chip;
}

function renderPrompt(el, text, node) {
  const entries = collectEntries(node);
  const byTag = new Map(entries.map((e) => [e.tag, e]));
  const frag = document.createDocumentFragment();
  const re = /(@[PpVvAaCcLl]\d+)/g;
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
const MODE_OPTIONS = ["参考", "T2V", "I2V", "自动"];
const MODE_HINTS = {
  "参考": "参考转视频：用 @P1/@V1/@A1 引用素材",
  "T2V": "纯文生视频（忽略参考素材）",
  "I2V": "图生视频：第一张已连接参考图作首帧；≥2 张时最后一张作尾帧",
  "自动": "自动：有视频/音频→参考；单图→I2V；多图→参考；无素材→T2V",
};

// 参考图用途标注（官方手册：参考图必须标注用途，否则保主体不保背景）
const REF_PURPOSE_OPTIONS = [
  { key: "", label: "用途…" },
  { key: "character", label: "人物", line: (k) => `<Picture ${k}> 是人物参考（锁定脸和服装）` },
  { key: "scene", label: "场景", line: (k) => `<Picture ${k}> 是场景参考（背景完全一致）` },
  { key: "style", label: "风格", line: (k) => `<Picture ${k}> 是风格参考（匹配这种美术风格）` },
  { key: "composition", label: "构图", line: (k) => `<Picture ${k}> 是构图参考（匹配这个取景）` },
  { key: "object", label: "物体", line: (k) => `<Picture ${k}> 是物体参考（保持这件物品原样）` },
  { key: "first_frame", label: "首帧", line: (k) => `<Picture ${k}> 是首帧参考` },
  { key: "last_frame", label: "尾帧", line: (k) => `<Picture ${k}> 是尾帧参考` },
  { key: "motion", label: "动作", line: (k) => `<Picture ${k}> 是动作参考（沿用它的动作）` },
];

// 自定义下拉：原生 <select> 在 Chrome 深色主题下选项看不清，改为按钮 + 弹出列表
// options: [{value, label, hint?}]，get 返回当前 value，set(value) 处理选择
function buildPopSelect({ title, options, get, set }) {
  const btn = document.createElement("button");
  btn.className = "zoey-pop-select";
  btn.title = title || "";
  const sync = () => {
    const cur = get();
    const opt = options.find((o) => o.value === cur);
    btn.textContent = opt ? opt.label : "…";
  };
  sync();
  btn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const box = $el("div", { className: "zoey-ref-picker" });
    const r = btn.getBoundingClientRect();
    const items = options.map((o) =>
      $el("div", {
        className: "zoey-ref-item" + (o.value === get() ? " zoey-ref-item--selected" : ""),
        onclick: () => {
          box.remove();
          set(o.value);
          sync();
        },
      }, [$el("div", { className: "zoey-ref-meta" }, [
        $el("div", { className: "zoey-ref-label", textContent: o.label }),
        o.hint ? $el("div", { className: "zoey-ref-hint", textContent: o.hint }) : null,
      ])])
    );
    box.replaceChildren($el("div", { className: "zoey-ref-header", textContent: title || "" }), ...items);
    document.body.appendChild(box);
    const bw = box.offsetWidth;
    const bh = box.offsetHeight;
    box.style.left = `${Math.max(4, Math.min(r.left, window.innerWidth - bw - 8))}px`;
    box.style.top = (r.bottom + 4 + bh > window.innerHeight && r.top - 4 - bh > 0)
      ? `${r.top - bh - 4}px`
      : `${r.bottom + 4}px`;
    const dismiss = (ev) => {
      if (!box.contains(ev.target)) {
        box.remove();
        document.removeEventListener("pointerdown", dismiss);
      }
    };
    setTimeout(() => document.addEventListener("pointerdown", dismiss), 0);
  });
  return btn;
}

// 与后端 _build_purpose_lines 对齐：读 ref_purposes JSON 生成用途标注行
function purposeLinesJs(node) {
  const w = node.widgets?.find((x) => x.name === "ref_purposes");
  let map = {};
  try { map = JSON.parse(w?.value || "{}") || {}; } catch (e) { map = {}; }
  const slots = [];
  for (const input of node.inputs || []) {
    if (input.link == null) continue;
    const m = input.name?.match(/^ref_image_(\d+)$/);
    if (m) slots.push(+m[1]);
  }
  slots.sort((a, b) => a - b);
  const lines = [];
  for (const [slotStr, key] of Object.entries(map)) {
    const slot = +slotStr;
    const idx = slots.indexOf(slot);
    if (idx < 0) continue;
    const opt = REF_PURPOSE_OPTIONS.find((o) => o.key === key);
    if (opt && opt.line) lines.push(opt.line(idx + 1));
  }
  return lines;
}

// 顶部模式切换条：挂在 prompt 容器里；通过隐藏的 mode 控件把值同步给后端
function buildModeBar(node) {
  const bar = document.createElement("div");
  bar.className = "zoey-mode-bar";
  const label = document.createElement("span");
  label.className = "mode-label";
  label.textContent = "模式";
  bar.appendChild(label);
  const buttons = {};
  for (const m of MODE_OPTIONS) {
    const b = document.createElement("button");
    b.className = "zoey-mode-btn";
    b.textContent = m;
    b.title = MODE_HINTS[m] || "";
    b.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (node._zoeyModeWidget) {
        node._zoeyModeWidget.value = m;
        app.graph?.setDirtyCanvas(true);
      }
    });
    bar.appendChild(b);
    buttons[m] = b;
  }
  const setActive = (m) => {
    const active = MODE_OPTIONS.includes(m) ? m : MODE_OPTIONS[0];
    for (const k of MODE_OPTIONS) buttons[k].classList.toggle("active", k === active);
  };
  bar.setActive = setActive;
  node._zoeyModeBar = { setActive };
  setActive(MODE_OPTIONS[0]);
  return bar;
}

function createPromptEditor(node, name, inputData) {
  const container = document.createElement("div");
  container.className = "zoey-prompt-container";

  container.appendChild(buildModeBar(node));

  const editor = document.createElement("div");
  editor.className = "zoey-prompt-editor";
  editor.contentEditable = "true";
  editor.spellcheck = false;
  editor.setAttribute("data-placeholder", "输入 @ 引用参考素材，如 @P1 的男人…");
  container.appendChild(editor);

  const strip = document.createElement("div");
  strip.className = "zoey-ref-strip";
  container.appendChild(strip);

  // 简单模式编译预览：展示最终发给模型的提示词（@展开+声明+用途+引号转对白）
  const previewRow = document.createElement("div");
  previewRow.className = "zoey-preview-row";
  const previewBtn = document.createElement("button");
  previewBtn.className = "zoey-preview-btn";
  previewBtn.textContent = "👁 预览最终提示词";
  previewRow.appendChild(previewBtn);
  const previewBox = document.createElement("textarea");
  previewBox.className = "zoey-simple-preview";
  previewBox.readOnly = true;
  previewBox.spellcheck = false;
  previewBox.style.display = "none";
  const previewMeta = document.createElement("div");
  previewMeta.className = "zoey-preview-meta";
  previewMeta.style.display = "none";
  container.append(previewRow, previewBox, previewMeta);

  function updatePreview() {
    if (previewBox.style.display === "none") return;
    try {
      previewBox.value = composeSimplePreview(node, serializePrompt(editor));
    } catch (err) {
      console.error("[Zoey MiniMax H3] preview:", err);
      previewBox.value = String(err?.message || err);
    }
    const text = serializePrompt(editor);
    const chars = text.length;
    const over = chars > 7000;
    const slots = [];
    for (const input of node.inputs || []) {
      if (input.link == null) continue;
      const m = input.name?.match(/^ref_image_(\d+)$/);
      if (m) slots.push(+m[1]);
    }
    slots.sort((a, b) => a - b);
    const used = new Set();
    for (const m of text.matchAll(/@[Pp]\d+/g)) used.add(m[0].toUpperCase());
    const unused = slots.map((s, i) => ({ s, tag: `@P${i + 1}` })).filter((x) => !used.has(x.tag)).map((x) => x.tag);
    const parts = [`${chars} 字符（上限7000）`];
    if (over) parts.push("⚠ 超出官方上限，可能被截断");
    if (unused.length) parts.push(`已连接未用: ${unused.join(", ")}`);
    previewMeta.textContent = parts.join(" ｜ ");
    previewMeta.classList.toggle("warn", over || unused.length > 0);
  }

  previewBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const show = previewBox.style.display === "none";
    strip.style.display = show ? "none" : "";
    previewBox.style.display = show ? "" : "none";
    previewMeta.style.display = show ? "" : "none";
    previewBtn.classList.toggle("active", show);
    if (show) updatePreview();
  });
  // 素材库异步加载完成后，若预览已展开则刷新 @L 展开结果
  document.addEventListener("zoey:library-loaded", () => {
    if (previewBox.style.display !== "none") updatePreview();
  });

  const widget = new DOMWidgetImpl({
    node,
    name,
    type: "customtext",
    element: container,
    options: {
      hideOnZoom: true,
      getValue: () => serializePrompt(editor),
      setValue: (v) => renderPrompt(editor, v ?? "", node),
      // 不固定：提示词编辑区高度 = 节点总高 − 其余控件高，节点拉大它就变大（280~820px）
      afterResize() { syncPromptHeight(); },
    },
  });
  widget.inputEl = editor;
  widget.strip = strip;
  addWidget(node, widget);

  // 计算并应用提示词区高度（按节点高度比例，节点拉大/缩小都跟随；280~900px）
  function syncPromptHeight() {
    try {
      const h = node.size?.[1];
      if (!h) return;
      const ph = Math.max(280, Math.min(900, Math.round(h * 0.35)));
      if (container.style.getPropertyValue("--comfy-widget-height") !== `${ph}px`) {
        container.style.setProperty("--comfy-widget-height", `${ph}px`);
        app.graph?.setDirtyCanvas(true);
      }
    } catch (e) {
      console.error("[Zoey MiniMax H3] syncPromptHeight:", e);
    }
  }
  // 暴露给 attachPicker：加载工作流后也同步一次（不只拖拽时才动）
  node._zoeySyncPromptHeight = syncPromptHeight;
  // 轮询节点高度：节点被拖拽变大小时可靠同步（不依赖 afterResize 是否触发）
  let _lastNodeH = node.size?.[1] || 0;
  node._zoeySizePoller = setInterval(() => {
    try {
      const h = node.size?.[1];
      if (h && Math.abs(h - _lastNodeH) > 2) {
        _lastNodeH = h;
        syncPromptHeight();
      }
    } catch (e) { /* 轮询错误忽略 */ }
  }, 400);

  editor.addEventListener("paste", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const text = e.clipboardData?.getData("text/plain") ?? "";
    insertTextAtCaret(editor, text);
  });
  editor.addEventListener("drop", (e) => e.preventDefault());
  editor.addEventListener("input", () => {
    if (previewBox.style.display !== "none") updatePreview();
  });
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
const DURATION_PRESETS = [1, 2, 4, 5, 8, 10, 15]; // 输出范围 1-15s

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
  // 状态行并入滑杆行：图×n 视频×n 音频×n ｜ 分辨率 ｜ 帧数，右侧小字（占位无独立整行）
  const status = document.createElement("span");
  status.className = "zoey-duration-status";
  row.append(slider, valueLabel, status);

  const presets = document.createElement("div");
  presets.className = "zoey-duration-presets";
  for (const s of DURATION_PRESETS) {
    const b = document.createElement("button");
    b.textContent = `${s}s`;
    b.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); setDuration(s); });
    presets.appendChild(b);
  }

  container.append(row, presets);

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

// ---- 模式值持有控件（隐藏，仅用于序列化；顶部模式条控制它） ----
function createModeWidget(node, name, inputData) {
  const holder = document.createElement("div");
  holder.className = "zoey-mode-holder";
  const options = (inputData && inputData[1]) || {};
  let current = MODE_OPTIONS.includes(options.default) ? options.default : MODE_OPTIONS[0];
  const widget = new DOMWidgetImpl({
    node,
    name,
    type: "customtext",
    element: holder,
    options: {
      hideOnZoom: true,
      getValue: () => current,
      setValue: (v) => {
        const m = MODE_OPTIONS.includes(v) ? v : MODE_OPTIONS[0];
        current = m;
        if (node._zoeyModeBar) node._zoeyModeBar.setActive(m);
        node._zoeyDirector?.composePreview?.();
      },
    },
  });
  widget.inputEl = holder;
  widget.hidden = true; // 值持有控件：不占位、不可见，仅参与序列化
  addWidget(node, widget);
  node._zoeyModeWidget = widget;
  if (node._zoeyModeBar) node._zoeyModeBar.setActive(current);
  return widget;
}

// ---- 参考图用途标注值持有控件（隐藏；预览条用途下拉控制它，序列化 {槽位: 用途key}） ----
function createRefPurposesWidget(node, name, inputData) {
  const holder = document.createElement("div");
  holder.className = "zoey-mode-holder";
  if (!node._zoeyRefPurposes) node._zoeyRefPurposes = {};
  const widget = new DOMWidgetImpl({
    node,
    name,
    type: "customtext",
    element: holder,
    options: {
      hideOnZoom: true,
      getValue: () => JSON.stringify(node._zoeyRefPurposes || {}),
      setValue: (v) => {
        try {
          node._zoeyRefPurposes = JSON.parse(v || "{}") || {};
        } catch (e) {
          node._zoeyRefPurposes = {};
        }
        if (node._zoeyRefreshRefStrip) node._zoeyRefreshRefStrip();
      },
    },
  });
  widget.inputEl = holder;
  widget.hidden = true;
  addWidget(node, widget);
  return widget;
}

// ---- 导演台（分镜列表）控件 ----
const MAX_TOTAL_SECONDS = 15;
const MAX_CHARACTERS = 4;
const MAX_SPEAKERS = 5;
const SPEAKER_IDS = ["S1", "S2", "S3", "S4", "S5"];

// 景别/运镜/转场预设（语汇来自 MiniMax H3 官方运镜词表）
const SHOT_SIZE_PRESETS = [
  { label: "特写", text: "a close-up shot" },
  { label: "近景", text: "a medium close-up shot" },
  { label: "中景", text: "a medium shot" },
  { label: "全景", text: "a wide shot" },
  { label: "远景", text: "a long shot" },
];
const CAMERA_PRESETS = [
  { label: "推", text: "The camera pushes in with small amplitude at slow speed." },
  { label: "拉", text: "The camera pulls out with small amplitude at slow speed." },
  { label: "摇左", text: "The camera pans left with small amplitude at slow speed." },
  { label: "摇右", text: "The camera pans right with small amplitude at slow speed." },
  { label: "横移左", text: "The camera trucks left with small amplitude at slow speed." },
  { label: "横移右", text: "The camera trucks right with small amplitude at slow speed." },
  { label: "升降", text: "The camera pedestals up with small amplitude at slow speed." },
  { label: "环绕", text: "The camera arcs around the subject with small amplitude at slow speed." },
  { label: "跟随", text: "A tracking shot follows the moving subject." },
  { label: "俯拍", text: "The camera looks down at the subject from above." },
  { label: "仰拍", text: "A low-angle shot frames the subject from below." },
  { label: "静态", text: "The camera holds a static shot." },
  { label: "POV", text: "A POV shot from the subject's perspective." },
  { label: "微抖", text: "The camera shakes slightly." },
];
const TRANSITION_PRESETS = ["", "cut", "hard cut", "WHIP PAN", "cross-dissolve", "fade", "wipe", "match cut"];

// 参考用途一键标注（语汇来自官方手册"参考用途清单"）
const REF_PURPOSES = [
  "是人物参考（锁定脸和服装）",
  "是场景参考（背景完全一致）",
  "是风格参考（匹配这种美术风格）",
  "是构图参考（匹配这个取景）",
  "是物体参考（保持这件物品原样）",
  "是首帧参考",
  "是尾帧参考",
  "是动作参考（沿用它的动作）",
];

// 镜头模板库：一键追加一组分镜
const SHOT_TEMPLATES = [
  { name: "产品广告", shots: [
    { prompt: "@P1 产品特写，缓慢推镜，金属质感与光影细节，桌面倒影", duration: 5, transition: "", dialogue: [] },
    { prompt: "产品在真实场景中使用，背景柔和虚化，光线自然，突出质感", duration: 5, transition: "cross-dissolve", dialogue: [] },
    { prompt: "产品与品牌元素同框收尾，构图工整，产品清晰定格", duration: 5, transition: "fade", dialogue: [] },
  ]},
  { name: "角色剧情", shots: [
    { prompt: "全景建立场景，@C1 走入画面，保持脸和服装一致", duration: 5, transition: "", dialogue: [] },
    { prompt: "中景 @C1 说话并做关键动作，情绪到位", duration: 5, transition: "cut", dialogue: [{ speaker: "S1", lang: "English", text: "" }] },
    { prompt: "近景 @C1 反应特写，表情变化，安静收尾", duration: 5, transition: "cut", dialogue: [] },
  ]},
  { name: "转场节奏", shots: [
    { prompt: "静态开场，主体缓慢入画，构图留白", duration: 4, transition: "", dialogue: [] },
    { prompt: "动作爆发，快速运动，冲击力强", duration: 4, transition: "WHIP PAN", dialogue: [] },
    { prompt: "收束定格，主体稳定，画面安静", duration: 4, transition: "WHIP PAN", dialogue: [] },
  ]},
  { name: "音乐MV", shots: [
    { prompt: "@C1 特写，跟随节奏轻摆，霓虹光晕", duration: 5, transition: "", dialogue: [] },
    { prompt: "@C1 全景在场景中舞动，镜头环绕慢速", duration: 5, transition: "cut", dialogue: [] },
    { prompt: "@C1 近景对嘴型，情绪高潮，落幅定格", duration: 5, transition: "fade", dialogue: [] },
  ]},
  { name: "单镜头", shots: [
    { prompt: "", duration: 5, transition: "", dialogue: [] },
  ]},
];

// 参考用途按钮插入到当前（或上次）聚焦的富文本编辑器光标处
let lastRefEditor = null;
function getFocusedEditor() {
  let el = document.activeElement;
  while (el && !el.isContentEditable) el = el.parentElement;
  return el && el.isContentEditable ? el : lastRefEditor;
}
function applyPurpose(text) {
  const ed = getFocusedEditor();
  if (!ed) return;
  appendToEditor(ed, text);
}

// 角色槽 -> 已连接图片槽位对应的 <Picture K> 编号（图片按连接顺序编号，与后端一致）
function pictureNumberForSlot(node, slot) {
  const connected = [];
  for (const input of node.inputs || []) {
    if (input.link == null) continue;
    const m = input.name?.match(/^ref_image_(\d+)$/);
    if (m) connected.push(+m[1]);
  }
  connected.sort((a, b) => a - b);
  const idx = connected.indexOf(slot);
  return idx >= 0 ? idx + 1 : null;
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  const mm = String(Math.floor(s)).padStart(2, "0");
  const ms = Math.round((s - Math.floor(s)) * 1000);
  return `${String(m).padStart(2, "0")}:${mm}.${String(ms).padStart(3, "0")}`;
}

// 引用声明行（与后端 zoey_minimax_h3_tags.build_declaration 对齐）
function buildDeclarationJs(node) {
  const c = countRefs(node);
  const joinNames = (ns) => ns.length === 1 ? ns[0] : ns.slice(0, -1).join(", ") + " and " + ns[ns.length - 1];
  const parts = [];
  if (c.images) parts.push(`${joinNames(c.images ? [...Array(c.images)].map((_, i) => `<Picture ${i + 1}>`) : [])} as reference ${c.images === 1 ? "frame" : "frames"}`);
  if (c.videos) parts.push(`${joinNames([...Array(c.videos)].map((_, i) => `<Video ${i + 1}>`))} as reference motion`);
  if (c.audios) parts.push(`${joinNames([...Array(c.audios)].map((_, i) => `<Audio ${i + 1}>`))} exactly as it is`);
  if (!parts.length) return "";
  return "Use " + parts.join(", and ") + ".";
}

// 素材库槽位信息：已连接图片/独立音频槽位 + 带音轨视频数（与后端编号一致）
function libraryInfo(node) {
  const imageSlots = [], audioSlots = [];
  for (const input of node.inputs || []) {
    if (input.link == null) continue;
    const m = input.name?.match(/^ref_image_(\d+)$/);
    if (m) imageSlots.push(+m[1]);
    const a = input.name?.match(/^ref_audio_(\d+)$/);
    if (a) audioSlots.push(+a[1]);
  }
  imageSlots.sort((x, y) => x - y);
  audioSlots.sort((x, y) => x - y);
  let paired = 0;
  for (let i = 0; i < 3; i++) {
    const v = node.inputs?.find((x) => x.name === `ref_video_${i}`);
    const va = node.inputs?.find((x) => x.name === `ref_video_audio_${i}`);
    if (v?.link != null && va?.link != null) paired++;
  }
  return { lib: globalLibrary, imageSlots, audioSlots, paired };
}

// 与后端 _library_plan 一致：算被引用素材库条目的 <Picture K>/<Audio K> 编号与注入顺序
function libraryPlan(node, referenced) {
  const { lib, imageSlots, audioSlots, paired } = libraryInfo(node);
  const pic_of = {}, aud_of = {}, imgOrder = [], audOrder = [];
  let nImg = imageSlots.length;
  let nAud = audioSlots.length;
  const sorted = [...new Set(referenced)].filter((i) => i >= 0 && i < lib.length).sort((a, b) => a - b);
  for (const i of sorted) {
    const e = lib[i];
    if (!e || typeof e !== "object") continue;
    const kind = e.kind || "";
    if (["character", "prop", "scene"].includes(kind) && e.file) {
      nImg += 1;
      pic_of[i] = nImg;
      imgOrder.push([i, kind]);
    }
  }
  const libAudio = [], charVoice = [];
  for (const i of sorted) {
    const e = lib[i];
    if (!e || typeof e !== "object") continue;
    const kind = e.kind || "";
    if (kind === "audio" && e.file) libAudio.push(i);
    else if (kind === "character" && e.audio_file) charVoice.push(i);
  }
  libAudio.forEach((i, rank) => { aud_of[i] = paired + nAud + rank + 1; audOrder.push([i, "audio"]); });
  charVoice.forEach((i, rank) => { aud_of[i] = paired + nAud + libAudio.length + rank + 1; audOrder.push([i, "voice"]); });
  return { pic_of, aud_of, imgOrder, audOrder };
}

// 引号台词自动转 <d>[语种] 内容</d>（与后端 _convert_quoted_dialogue/_detect_lang 对齐）
function detectLangJs(text) {
  if (/[぀-ヿ]/.test(text)) return "Japanese";
  if (/[가-힣]/.test(text)) return "Korean";
  if (/[一-鿿]/.test(text)) return "Chinese";
  if (/[Ѐ-ӿ]/.test(text)) return "Russian";
  if (/[฀-๿]/.test(text)) return "Thai";
  if (/[؀-ۿ]/.test(text)) return "Arabic";
  return "English";
}
function convertDialogueJs(text) {
  if (!text) return text;
  return String(text).replace(/「([^」\n]*)」|『([^』\n]*)』|“([^”\n]*)”|"([^"\n]*)"/g, (m, a, b, c, d) => {
    const content = (a || b || c || d || "").trim();
    if (!content) return m;
    return `<d>[${detectLangJs(content)}] ${content}</d>`;
  });
}
function dialogueConvertEnabled(node) {
  const w = node.widgets?.find((x) => x.name === "dialogue_convert");
  return w ? !!w.value : true;
}

// 编译预览：把导演台数据拼成最终发给模型的提示词（与后端 _compose_director 对齐）
function composeDirector(node, d) {
  const chars = d.characters || [];
  const charPic = {};
  chars.forEach((ch, i) => {
    if (!ch) return;
    const k = pictureNumberForSlot(node, ch.slot);
    if (k != null) charPic[i + 1] = k;
  });
  const libInfo = libraryInfo(node);
  // 被引用的 @L 索引（ref_decl + 所有镜头提示词）→ 全局合并编号（与后端 _library_plan 一致）
  const refText = [String(d.ref_decl || ""), ...(d.shots || []).map((s) => String(s.prompt || ""))].join(" ");
  const refIdx = new Set();
  for (const m of refText.matchAll(/@[Ll](\d+)/g)) refIdx.add(+m[1] - 1);
  const libPlan = libraryPlan(node, [...refIdx]);
  const libAnnos = [];
  const libAnnoSeen = new Set();
  const expandLib = (text) => String(text || "").replace(/@[Ll](\d+)/g, (m, n) => {
    const i = +n - 1;
    const e = libInfo.lib[i];
    if (!e || typeof e !== "object") return m;
    let line = null;
    let tag = null;
    if (i in libPlan.pic_of) {
      tag = `<Picture ${libPlan.pic_of[i]}>`;
      const kind = e.kind || "";
      const name = (e.name || "").trim();
      if (kind === "character") {
        line = name ? `${tag} 是${name}的人物参考（锁定脸和服装）` : `${tag} 是人物参考（锁定脸和服装）`;
        const app = (e.appearance || "").trim();
        if (app) line += `。外貌：${app}`;
        if (i in libPlan.aud_of) line += `，音色参考 <Audio ${libPlan.aud_of[i]}>`;
      } else if (kind === "prop") {
        line = name ? `${tag} 是${name}的物体参考（保持原样）` : `${tag} 是物体参考（保持这件物品原样）`;
      } else {
        line = name ? `${tag} 是${name}的场景参考（背景完全一致）` : `${tag} 是场景参考（背景完全一致）`;
      }
    } else if (i in libPlan.aud_of) {
      tag = `<Audio ${libPlan.aud_of[i]}>`;
      line = `${tag} 原样复用这段音频`;
    }
    if (line && !libAnnoSeen.has(line)) { libAnnoSeen.add(line); libAnnos.push(line); }
    return tag || m;
  });
  const expand = (text) => {
    text = String(text || "").replace(/@[Cc](\d+)/g, (m, n) => {
      const k = charPic[+n];
      return k != null ? `<Picture ${k}>` : m;
    });
    text = expandLib(text);
    text = text.replace(/@([PpVv])(\d+)/g, (m, tag, num) => `<${tag.toUpperCase() === "P" ? "Picture" : "Video"} ${num}>`);
    text = text.replace(/@[Aa](\d+)/g, (m, n) => `<Audio ${n}>`);
    return text;
  };

  const decl = [];
  if ((d.ref_decl || "").trim()) decl.push(expand(d.ref_decl));
  chars.forEach((ch, i) => {
    if (!ch) return;
    const k = charPic[i + 1];
    if (k == null) return;
    const name = (ch.name || "").trim();
    decl.push(name
      ? `<Picture ${k}> 是${name}的人物参考（锁定脸和服装）`
      : `<Picture ${k}> 是人物参考（锁定脸和服装）`);
  });
  const speakers = {};
  (d.speakers || []).forEach((sp) => { if (sp && sp.id) speakers[sp.id] = sp.desc || ""; });

  const parts = [];
  const consistent = d.consistent !== false;
  const dialogueConvert = dialogueConvertEnabled(node);
  (d.shots || []).forEach((shot, i) => {
    if (i > 0) {
      const tr = (shot.transition || "").trim();
      if (tr) parts.push(`TRANSITION: ${tr}`);
    }
    let shotText = String(shot.prompt || "");
    if (dialogueConvert) shotText = convertDialogueJs(shotText);
    shotText = expand(shotText).trim();
    if (i > 0 && consistent) shotText += "\n保持与上一镜相同的角色、场景、服装与光线。";
    for (const dl of shot.dialogue || []) {
      const text = (dl.text || "").trim();
      if (!text) continue;
      const spk = (dl.speaker || "").trim() || "S1";
      const lang = (dl.lang || "").trim() || "English";
      const desc = speakers[spk];
      shotText += desc
        ? `\n${desc} (${spk}) says: <d>[${lang}] ${text}.</d>`
        : `\n(${spk}) says: <d>[${lang}] ${text}.</d>`;
    }
    parts.push(`CUT ${i + 1}: ${shotText}`);
  });
  if ((d.soundscape || "").trim()) parts.push(`overall_soundscape: ${d.soundscape.trim()}`);
  if ((d.music || "").trim()) parts.push(`non_diegetic_music: ${d.music.trim()}`);

  if (libAnnos.length) decl.push(...libAnnos);
  if (decl.length) parts.unshift(decl.join("\n"));
  const autoW = node.widgets?.find((w) => w.name === "auto_declaration");
  const auto = autoW ? !!autoW.value : true;
  if (!(d.ref_decl || "").trim() && auto) {
    const bd = buildDeclarationJs(node);
    if (bd) parts.unshift(bd);
  }
  const purposeLines = purposeLinesJs(node);
  if (purposeLines.length) parts.unshift(purposeLines.join("\n"));
  return parts.join("\n");
}

// T2V/I2V 编译预览：与后端 _compose_plain_director 对齐（不做 @ 展开）
function composePlainDirector(node, d) {
  const speakers = {};
  (d.speakers || []).forEach((sp) => { if (sp && sp.id) speakers[sp.id] = sp.desc || ""; });
  const parts = [];
  const consistent = d.consistent !== false;
  const dialogueConvert = dialogueConvertEnabled(node);
  (d.shots || []).forEach((shot, i) => {
    if (i > 0) {
      const tr = (shot.transition || "").trim();
      if (tr) parts.push(`TRANSITION: ${tr}`);
    }
    let t = String(shot.prompt || "");
    if (dialogueConvert) t = convertDialogueJs(t);
    t = t.trim();
    if (i > 0 && consistent) t += "\n保持与上一镜相同的角色、场景、服装与光线。";
    for (const dl of shot.dialogue || []) {
      const text = (dl.text || "").trim();
      if (!text) continue;
      const spk = (dl.speaker || "").trim() || "S1";
      const lang = (dl.lang || "").trim() || "English";
      const desc = speakers[spk];
      t += desc
        ? `\n${desc} (${spk}) says: <d>[${lang}] ${text}.</d>`
        : `\n(${spk}) says: <d>[${lang}] ${text}.</d>`;
    }
    parts.push(`CUT ${i + 1}: ${t}`);
  });
  if ((d.soundscape || "").trim()) parts.push(`overall_soundscape: ${d.soundscape.trim()}`);
  if ((d.music || "").trim()) parts.push(`non_diegetic_music: ${d.music.trim()}`);
  return parts.join("\n");
}

// 简单模式编译预览：与后端 _run_ref2va 非导演台路径对齐
function composeSimplePreview(node, text) {
  const modeW = node.widgets?.find((w) => w.name === "mode");
  const mode = modeW ? modeW.value : "参考";
  let out = dialogueConvertEnabled(node) ? convertDialogueJs(text) : text;
  if (mode === "T2V" || mode === "I2V") return out; // 后端不展开 @、不加声明

  out = out.replace(/@([PpVv])(\d+)/g, (m, tag, num) => `<${tag.toUpperCase() === "P" ? "Picture" : "Video"} ${num}>`);
  out = out.replace(/@[Aa](\d+)/g, (m, n) => `<Audio ${n}>`);

  const libInfo = libraryInfo(node);
  const refIdx = new Set();
  for (const m of String(text).matchAll(/@[Ll](\d+)/g)) refIdx.add(+m[1] - 1);
  const libPlan = libraryPlan(node, [...refIdx]);
  const libAnnos = [];
  const seen = new Set();
  out = out.replace(/@[Ll](\d+)/g, (m, n) => {
    const i = +n - 1;
    const e = libInfo.lib[i];
    if (!e || typeof e !== "object") return m;
    let line = null, tag = null;
    if (i in libPlan.pic_of) {
      tag = `<Picture ${libPlan.pic_of[i]}>`;
      const kind = e.kind || "";
      const name = (e.name || "").trim();
      if (kind === "character") {
        line = name ? `${tag} 是${name}的人物参考（锁定脸和服装）` : `${tag} 是人物参考（锁定脸和服装）`;
        const app = (e.appearance || "").trim();
        if (app) line += `。外貌：${app}`;
        if (i in libPlan.aud_of) line += `，音色参考 <Audio ${libPlan.aud_of[i]}>`;
      } else if (kind === "prop") {
        line = name ? `${tag} 是${name}的物体参考（保持原样）` : `${tag} 是物体参考（保持这件物品原样）`;
      } else {
        line = name ? `${tag} 是${name}的场景参考（背景完全一致）` : `${tag} 是场景参考（背景完全一致）`;
      }
    } else if (i in libPlan.aud_of) {
      tag = `<Audio ${libPlan.aud_of[i]}>`;
      line = `${tag} 原样复用这段音频`;
    }
    if (line && !seen.has(line)) { seen.add(line); libAnnos.push(line); }
    return tag || m;
  });

  const decl = [];
  decl.push(...purposeLinesJs(node));
  const autoW = node.widgets?.find((w) => w.name === "auto_declaration");
  const auto = autoW ? !!autoW.value : true;
  if (auto) {
    const bd = buildDeclarationJs(node);
    if (bd) decl.push(bd);
  }
  decl.push(...libAnnos);
  if (decl.length) out = decl.join("\n") + "\n" + out;
  return out;
}

// 把文本追加到 editor 光标处（无光标时定位到末尾）
function appendToEditor(editor, text) {
  const sel = window.getSelection();
  let inEditor = sel && sel.rangeCount && editor.contains(sel.getRangeAt(0).commonAncestorContainer);
  if (!inEditor) {
    const range = document.createRange();
    range.selectNodeContents(editor);
    range.collapse(false);
    sel.removeAllRanges();
    sel.addRange(range);
    editor.focus();
  }
  insertTextAtCaret(editor, " " + text);
}

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

  // 参考素材说明：contenteditable 富文本，支持 @ 缩略图 chip 与下拉选择
  const declEditor = document.createElement("div");
  declEditor.className = "zoey-director-decl";
  declEditor.contentEditable = "true";
  declEditor.spellcheck = false;
  declEditor.setAttribute("data-placeholder", "参考素材说明（可空）：@P1 是人物参考，@P2 是场景参考…");
  declEditor.addEventListener("paste", (e) => {
    e.preventDefault();
    e.stopPropagation();
    insertTextAtCaret(declEditor, e.clipboardData?.getData("text/plain") ?? "");
  });
  declEditor.addEventListener("drop", (e) => e.preventDefault());

  const charsBox = document.createElement("div");
  charsBox.className = "zoey-director-chars";
  const spkBox = document.createElement("div");
  const sndBox = document.createElement("div");
  const list = document.createElement("div");
  list.className = "zoey-director-list";
  const total = document.createElement("div");
  total.className = "zoey-director-total";
  const preview = document.createElement("textarea");
  preview.className = "zoey-director-preview";
  preview.readOnly = true;
  preview.spellcheck = false;
  preview.placeholder = "编译预览：最终发给模型的完整提示词会显示在这里";

  const toolbar = document.createElement("div");
  toolbar.className = "zoey-director-toolbar";

  container.append(header, toolbar, declEditor, charsBox, spkBox, sndBox, list, total, preview);

  let refDecl = "";
  let shots = [];
  let speakers = [];
  let characters = [];
  let soundscape = "";
  let music = "";
  let consistent = true;
  let pickers = [];
  let declPicker = null;
  let startTimes = [];
  // collectEntries 通过 node._zoeyDirector.characters 读角色槽（getter 保证与闭包同步）
  node._zoeyDirector = { get characters() { return characters; }, composePreview };

  const widget = new DOMWidgetImpl({
    node,
    name,
    type: "customtext",
    element: container,
    options: {
      hideOnZoom: true,
      getValue: () => JSON.stringify({ ref_decl: refDecl, shots, speakers, characters, soundscape, music, consistent }),
      setValue: (v) => {
        const d = parseData(v);
        refDecl = d.refDecl;
        shots = d.shots;
        speakers = d.speakers;
        characters = d.characters;
        soundscape = d.soundscape;
        music = d.music;
        consistent = d.consistent;
        render();
      },
    },
  });
  widget.inputEl = list;
  addWidget(node, widget);

  function parseData(v) {
    try {
      const d = JSON.parse(v || "");
      if (Array.isArray(d)) return { refDecl: "", shots: d, speakers: [], characters: [], soundscape: "", music: "", consistent: true }; // 兼容旧格式：纯镜头数组
      if (d && typeof d === "object") {
        return {
          refDecl: typeof d.ref_decl === "string" ? d.ref_decl : "",
          shots: Array.isArray(d.shots) ? d.shots : [],
          speakers: Array.isArray(d.speakers) ? d.speakers : [],
          characters: Array.isArray(d.characters) ? d.characters : [],
          soundscape: typeof d.soundscape === "string" ? d.soundscape : "",
          music: typeof d.music === "string" ? d.music : "",
          consistent: typeof d.consistent === "boolean" ? d.consistent : true,
        };
      }
    } catch (e) {}
    return { refDecl: "", shots: [], speakers: [], characters: [], soundscape: "", music: "", consistent: true };
  }

  declPicker = new RefPicker(node, null, declEditor);
  declPicker.onChange = () => { refDecl = serializePrompt(declEditor); composePreview(); };

  function cloneShot(shot) {
    return {
      prompt: shot.prompt || "",
      duration: shot.duration ?? 5,
      transition: shot.transition || "",
      dialogue: (shot.dialogue || []).map((d) => ({ ...d })),
    };
  }

  function resolveCharEntry(i) {
    const ch = characters[i];
    if (!ch || ch.slot == null) return null;
    return collectEntries(node).find((e) => e.kind === "图" && e.slot === ch.slot) || null;
  }

  function pickImageForChar(anchor, i) {
    const entries = collectEntries(node).filter((e) => e.kind === "图");
    const box = $el("div", { className: "zoey-ref-picker" });
    const r = anchor.getBoundingClientRect();
    const items = entries.map((en) =>
      $el("div", {
        className: "zoey-ref-item",
        onclick: () => {
          characters[i] = { slot: en.slot, name: characters[i]?.name || "" };
          box.remove();
          render();
        },
      }, [entryThumb(en), $el("div", { className: "zoey-ref-meta" }, [
        $el("div", { className: "zoey-ref-label", textContent: `图 ${en.num}` }),
        $el("div", { className: "zoey-ref-tag", textContent: en.tag }),
      ])])
    );
    box.replaceChildren(
      $el("div", { className: "zoey-ref-header", textContent: `@C${i + 1} 选择角色参考图` }),
      ...(items.length ? items : [$el("div", { className: "zoey-ref-empty", textContent: "未连接参考图，请先在节点上连接图片" })]),
    );
    document.body.appendChild(box);
    box.style.left = `${r.left}px`;
    box.style.top = `${r.bottom + 4}px`;
    const dismiss = (e) => { if (!box.contains(e.target)) { box.remove(); document.removeEventListener("pointerdown", dismiss); } };
    setTimeout(() => document.addEventListener("pointerdown", dismiss), 0);
  }

  function charSlot(ch, i) {
    const wrap = document.createElement("div");
    wrap.className = "zoey-director-char";
    const btn = document.createElement("button");
    btn.className = "zoey-director-char-btn";
    if (ch && ch.slot != null) {
      const entry = resolveCharEntry(i);
      const url = entry ? resolveThumbSrc(entry) : null;
      if (url) {
        const img = document.createElement("img");
        img.onerror = () => { btn.textContent = `@C${i + 1}`; };
        img.src = url;
        btn.appendChild(img);
      } else {
        btn.textContent = `@C${i + 1}`;
      }
    } else {
      btn.textContent = "＋";
    }
    btn.title = ch && ch.slot != null ? "点击更换角色参考图" : `点击为 @C${i + 1} 分配参考图`;
    btn.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); pickImageForChar(btn, i); });

    const name = document.createElement("input");
    name.placeholder = `@C${i + 1} 名字`;
    name.value = (ch && ch.name) || "";
    name.addEventListener("input", () => {
      if (!characters[i]) characters[i] = { slot: null, name: "" };
      characters[i].name = name.value;
      composePreview();
    });

    const del = document.createElement("button");
    del.className = "zoey-director-mini-btn";
    del.textContent = "✕";
    del.title = "清空角色";
    del.addEventListener("pointerdown", (e) => {
      e.preventDefault(); e.stopPropagation();
      characters[i] = null;
      render();
    });

    wrap.append(btn, name, ch && ch.slot != null ? del : null);
    return wrap;
  }

  function camRow(editor) {
    const row = document.createElement("div");
    row.className = "zoey-director-camrow";
    const lbl1 = document.createElement("span");
    lbl1.className = "cam-label";
    lbl1.textContent = "景别";
    row.appendChild(lbl1);
    for (const p of SHOT_SIZE_PRESETS) row.appendChild(camBtn(editor, p));
    const lbl2 = document.createElement("span");
    lbl2.className = "cam-label";
    lbl2.textContent = "运镜";
    row.appendChild(lbl2);
    for (const p of CAMERA_PRESETS) row.appendChild(camBtn(editor, p));
    return row;
  }
  function camBtn(editor, p) {
    const b = document.createElement("button");
    b.textContent = p.label;
    b.title = p.text;
    b.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); appendToEditor(editor, p.text); });
    return b;
  }

  function addSpeaker() {
    if (speakers.length >= MAX_SPEAKERS) return;
    const used = new Set(speakers.map((s) => s.id));
    const id = SPEAKER_IDS.find((s) => !used.has(s)) || `S${speakers.length + 1}`;
    speakers.push({ id, desc: "" });
    render();
  }

  function buildSpeakerUI() {
    const box = document.createElement("div");
    box.className = "zoey-director-collapse";
    const head = document.createElement("div");
    head.className = "zoey-director-collapse-head";
    const t = document.createElement("span");
    t.textContent = `🗣 说话人定义（${speakers.length}/${MAX_SPEAKERS}）`;
    const add = document.createElement("button");
    add.className = "zoey-director-mini-btn";
    add.textContent = "＋ 说话人";
    add.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); addSpeaker(); });
    const caret = document.createElement("span");
    caret.className = "caret";
    caret.textContent = "▸";
    head.append(t, add);
    const body = document.createElement("div");
    body.className = "zoey-director-collapse-body";
    body.style.display = "none";
    head.addEventListener("click", (e) => {
      if (e.target === add) return;
      body.style.display = body.style.display === "none" ? "flex" : "none";
      caret.textContent = body.style.display === "none" ? "▸" : "▾";
    });
    speakers.forEach((sp, i) => {
      const row = document.createElement("div");
      row.className = "zoey-director-speaker";
      const id = document.createElement("span");
      id.className = "zoey-director-spk-id";
      id.textContent = sp.id;
      const desc = document.createElement("input");
      desc.value = sp.desc || "";
      desc.placeholder = "身份/音色，如 the young woman with a soft, breathy voice";
      desc.addEventListener("input", () => { sp.desc = desc.value; composePreview(); });
      const del = document.createElement("button");
      del.className = "zoey-director-mini-btn";
      del.textContent = "✕";
      del.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); speakers.splice(i, 1); render(); });
      row.append(id, desc, del);
      body.appendChild(row);
    });
    box.append(head, body);
    return box;
  }

  function buildSoundUI() {
    const box = document.createElement("div");
    box.className = "zoey-director-collapse";
    const head = document.createElement("div");
    head.className = "zoey-director-collapse-head";
    const t = document.createElement("span");
    t.textContent = "🎵 音效 / 配乐";
    const caret = document.createElement("span");
    caret.className = "caret";
    caret.textContent = "▸";
    head.append(t, caret);
    const body = document.createElement("div");
    body.className = "zoey-director-collapse-body";
    body.style.display = "none";
    head.addEventListener("click", () => {
      body.style.display = body.style.display === "none" ? "flex" : "none";
      caret.textContent = body.style.display === "none" ? "▸" : "▾";
    });
    const f1 = document.createElement("div");
    f1.className = "zoey-director-field";
    const l1 = document.createElement("label");
    l1.textContent = "环境声/音效 overall_soundscape";
    const sfx = document.createElement("input");
    sfx.value = soundscape;
    sfx.placeholder = "全片环境声与物理声，如 Rain on the window...";
    sfx.addEventListener("input", () => { soundscape = sfx.value; composePreview(); });
    f1.append(l1, sfx);
    const f2 = document.createElement("div");
    f2.className = "zoey-director-field";
    const l2 = document.createElement("label");
    l2.textContent = "配乐 non_diegetic_music";
    const bgm = document.createElement("input");
    bgm.value = music;
    bgm.placeholder = "仅观众可听的配乐，如 Sustained cello at slow tempo...";
    bgm.addEventListener("input", () => { music = bgm.value; composePreview(); });
    f2.append(l2, bgm);
    body.append(f1, f2);
    box.append(head, body);
    return box;
  }

  function applyTemplate(t) {
    t.shots.forEach((s) => shots.push(cloneShot(s)));
    render();
  }

  function toggleTemplateDropdown(btn) {
    document.querySelectorAll(".zoey-director-tmpl").forEach((d) => d.remove());
    const box = $el("div", { className: "zoey-ref-picker zoey-director-tmpl" });
    const r = btn.getBoundingClientRect();
    const items = SHOT_TEMPLATES.map((t) =>
      $el("div", { className: "zoey-ref-item", onclick: () => { box.remove(); applyTemplate(t); } }, [
        $el("div", { className: "zoey-ref-meta" }, [
          $el("div", { className: "zoey-ref-label", textContent: t.name }),
          $el("div", { className: "zoey-ref-hint", textContent: `${t.shots.length} 镜` }),
        ]),
      ])
    );
    box.replaceChildren($el("div", { className: "zoey-ref-header", textContent: "镜头模板（追加到分镜末尾）" }), ...items);
    document.body.appendChild(box);
    box.style.left = `${r.left}px`;
    box.style.top = `${r.bottom + 4}px`;
    const dismiss = (e) => { if (!box.contains(e.target)) { box.remove(); document.removeEventListener("pointerdown", dismiss); } };
    setTimeout(() => document.addEventListener("pointerdown", dismiss), 0);
  }

  // 按台词量分配每镜时长：对白多的镜头给更多时间，总量不超过 15s（对白~4字/秒）
  function autoAllocateDurations() {
    const needs = shots.map((s) => {
      const chars = (s.dialogue || []).reduce((a, d) => a + ((d.text || "").trim().length || 0), 0);
      return chars > 0 ? Math.min(MAX_TOTAL_SECONDS, Math.max(1.5, Math.ceil((chars / 4) * 10) / 10)) : 3;
    });
    const sum = needs.reduce((a, b) => a + b, 0);
    const factor = sum > MAX_TOTAL_SECONDS ? MAX_TOTAL_SECONDS / sum : 1;
    shots.forEach((s, i) => {
      s.duration = Math.round(Math.min(MAX_TOTAL_SECONDS, Math.max(1, needs[i] * factor)) * 10) / 10;
    });
    render();
  }

  function buildToolbar() {
    const tb = document.createElement("div");
    tb.className = "zoey-director-toolbar";

    const purposeLabel = document.createElement("span");
    purposeLabel.className = "tool-label";
    purposeLabel.textContent = "📌 用途";
    tb.appendChild(purposeLabel);
    for (const p of REF_PURPOSES) {
      const b = document.createElement("button");
      b.textContent = p.replace(/^是/, "").split("（")[0];
      b.title = p;
      b.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); applyPurpose(p); });
      tb.appendChild(b);
    }

    const tmplBtn = document.createElement("button");
    tmplBtn.textContent = "📋 模板";
    tmplBtn.title = "套用镜头模板";
    tmplBtn.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); toggleTemplateDropdown(tmplBtn); });
    tb.appendChild(tmplBtn);

    const allocBtn = document.createElement("button");
    allocBtn.textContent = "⏱ 按台词分配时长";
    allocBtn.title = "根据对白量自动分配每镜时长（总长≤15s）";
    allocBtn.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); autoAllocateDurations(); });
    tb.appendChild(allocBtn);

    const toggle = document.createElement("label");
    toggle.className = "zoey-director-toggle";
    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.checked = consistent;
    chk.addEventListener("change", () => { consistent = chk.checked; composePreview(); });
    toggle.append(chk, document.createTextNode("跨镜一致"));
    tb.appendChild(toggle);

    return tb;
  }

  function render() {
    pickers.forEach((p) => p.hide());
    declPicker?.hide();
    pickers = [];
    toolbar.replaceChildren(buildToolbar());
    renderPrompt(declEditor, refDecl, node);
    charsBox.replaceChildren();
    for (let i = 0; i < MAX_CHARACTERS; i++) charsBox.appendChild(charSlot(characters[i], i));
    spkBox.replaceChildren(buildSpeakerUI());
    sndBox.replaceChildren(buildSoundUI());
    startTimes = [];
    let acc = 0;
    shots.forEach((s) => { startTimes.push(acc); acc += parseFloat(s.duration) || 5; });
    list.replaceChildren();
    shots.forEach((shot, i) => list.appendChild(shotCard(shot, i)));
    composePreview();
  }

  function shotCard(shot, i) {
    const card = document.createElement("div");
    card.className = "zoey-director-shot";

    const head = document.createElement("div");
    head.className = "zoey-director-shot-head";
    const label = document.createElement("span");
    label.textContent = `CUT ${i + 1} · ${fmtTime(startTimes[i] ?? 0)}`;
    const btns = document.createElement("div");
    btns.style.display = "flex";
    btns.style.gap = "2px";
    const mkBtn = (t, title, fn) => {
      const b = document.createElement("button");
      b.className = "zoey-director-mini-btn";
      b.textContent = t;
      b.title = title;
      b.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); fn(); });
      return b;
    };
    btns.append(
      mkBtn("⇧", "上移", () => { if (i > 0) { [shots[i - 1], shots[i]] = [shots[i], shots[i - 1]]; render(); } }),
      mkBtn("⇩", "下移", () => { if (i < shots.length - 1) { [shots[i + 1], shots[i]] = [shots[i], shots[i + 1]]; render(); } }),
      mkBtn("⧉", "复制镜头", () => { shots.splice(i + 1, 0, cloneShot(shot)); render(); }),
      mkBtn("✕", "删除镜头", () => { shots.splice(i, 1); render(); }),
    );
    head.append(label, btns);

    // 镜头提示词：contenteditable 富文本，支持 @ 缩略图 chip、角色 @C1、下拉选择
    const editor = document.createElement("div");
    editor.className = "zoey-director-shot-editor";
    editor.contentEditable = "true";
    editor.spellcheck = false;
    editor.setAttribute("data-placeholder", "本镜头提示词，输入 @ 引用素材，角色用 @C1…");
    renderPrompt(editor, shot.prompt || "", node);
    editor.addEventListener("paste", (e) => {
      e.preventDefault();
      e.stopPropagation();
      insertTextAtCaret(editor, e.clipboardData?.getData("text/plain") ?? "");
    });
    editor.addEventListener("drop", (e) => e.preventDefault());

    const picker = new RefPicker(node, null, editor);
    picker.onChange = () => { shot.prompt = serializePrompt(editor); composePreview(); };
    pickers.push(picker);

    const cam = camRow(editor);

    // 对白（说话人 S1/S2… + 语言 + 台词原文）
    const dialogueBox = document.createElement("div");
    dialogueBox.style.display = "flex";
    dialogueBox.style.flexDirection = "column";
    dialogueBox.style.gap = "3px";
    shot.dialogue = Array.isArray(shot.dialogue) ? shot.dialogue : [];
    function renderDialogue() {
      dialogueBox.replaceChildren();
      shot.dialogue.forEach((dl, j) => {
        const row = document.createElement("div");
        row.className = "zoey-director-dlg";
        const spkSel = document.createElement("select");
        spkSel.className = "spk";
        for (const s of SPEAKER_IDS) {
          const o = document.createElement("option");
          o.value = s; o.textContent = s;
          if (dl.speaker === s) o.selected = true;
        }
        if (dl.speaker && !SPEAKER_IDS.includes(dl.speaker)) {
          const o = document.createElement("option");
          o.value = dl.speaker; o.textContent = dl.speaker; o.selected = true;
          spkSel.appendChild(o);
        }
        spkSel.addEventListener("change", () => { dl.speaker = spkSel.value; composePreview(); });
        const lang = document.createElement("input");
        lang.className = "lang";
        lang.value = dl.lang || "English";
        lang.placeholder = "语言";
        lang.addEventListener("input", () => { dl.lang = lang.value; composePreview(); });
        const txt = document.createElement("input");
        txt.className = "txt";
        txt.value = dl.text || "";
        txt.placeholder = "台词原文（<d> 内逐字保留，不翻译）";
        txt.addEventListener("input", () => { dl.text = txt.value; composePreview(); });
        const del = document.createElement("button");
        del.className = "del";
        del.textContent = "✕";
        del.title = "删除对白";
        del.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); shot.dialogue.splice(j, 1); renderDialogue(); composePreview(); });
        row.append(spkSel, lang, txt, del);
        dialogueBox.appendChild(row);
      });
      const add = document.createElement("button");
      add.className = "zoey-director-dlgadd";
      add.textContent = "＋ 对白";
      add.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); shot.dialogue.push({ speaker: "S1", lang: "English", text: "" }); renderDialogue(); composePreview(); });
      dialogueBox.appendChild(add);
    }
    renderDialogue();

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
      composePreview();
    });
    const durUnit = document.createElement("span");
    durUnit.textContent = "s";
    durRow.append(durLbl, durInput, durUnit);

    const trans = document.createElement("select");
    trans.className = "zoey-director-shot-trans";
    for (const t of TRANSITION_PRESETS) {
      const o = document.createElement("option");
      o.value = t;
      o.textContent = t || "（无转场）";
      if ((shot.transition || "") === t) o.selected = true;
    }
    if (shot.transition && !TRANSITION_PRESETS.includes(shot.transition)) {
      const o = document.createElement("option");
      o.value = shot.transition;
      o.textContent = shot.transition;
      o.selected = true;
      trans.appendChild(o);
    }
    trans.addEventListener("change", () => { shot.transition = trans.value; composePreview(); });

    card.append(head, cam, editor, dialogueBox, durRow, trans);
    return card;
  }

  function composePreview() {
    const d = { ref_decl: refDecl, shots, speakers, characters, soundscape, music, consistent };
    const sum = shots.reduce((s, x) => s + (parseFloat(x.duration) || 0), 0);
    const capped = Math.min(sum, MAX_TOTAL_SECONDS);
    const frames = frameCount(capped);
    total.textContent = `总时长 ${sum}s → ${capped}s (${frames}帧)` + (sum > MAX_TOTAL_SECONDS ? "  ⚠ 超15s将截断" : "");
    try {
      const modeW = node.widgets?.find((w) => w.name === "mode");
      const mode = modeW ? modeW.value : "参考";
      preview.value = (mode === "T2V" || mode === "I2V")
        ? composePlainDirector(node, d)
        : composeDirector(node, d);
    } catch (e) {
      console.error("[Zoey MiniMax H3] composePreview:", e);
      preview.value = String(e?.message || e);
    }
  }

  addBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault(); e.stopPropagation();
    shots.push({ prompt: "", duration: 5, transition: "", dialogue: [] });
    render();
  });

  // 参考素材连接变化时刷新角色槽缩略图/编译预览
  const prevConn = node.onConnectionsChange;
  node.onConnectionsChange = (type, slot, isConnected, ...rest) => {
    try {
      prevConn?.call(node, type, slot, isConnected, ...rest);
    } catch (e) {
      console.error("[Zoey MiniMax H3] director onConnectionsChange:", e);
    }
    render();
  };

  render();
  return widget;
}

// ---- 素材库（角色/道具/场景/音频）控件 ----
function createLibraryWidget(node, name, inputData) {
  const container = document.createElement("div");
  container.className = "zoey-library";

  const head = document.createElement("div");
  head.className = "zoey-director-header";
  const title = document.createElement("span");
  title.textContent = "🧰 素材库（全局·跨工作流）";
  const addBtn = document.createElement("button");
  addBtn.textContent = "＋ 添加";
  addBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault(); e.stopPropagation();
    globalLibrary.push({ kind: "character", file: "", name: "", appearance: "", audio_file: "", desc: "" });
    renderLibrary();
    scheduleSaveLibrary();
  });
  const refreshBtn = document.createElement("button");
  refreshBtn.textContent = "↻";
  refreshBtn.title = "从服务器刷新（别的节点可能改过）";
  refreshBtn.addEventListener("pointerdown", async (e) => {
    e.preventDefault(); e.stopPropagation();
    await loadGlobalLibrary();
    renderLibrary();
  });
  head.append(title, addBtn, refreshBtn);

  const list = document.createElement("div");
  list.className = "zoey-library-list";
  container.append(head, list);

  const widget = new DOMWidgetImpl({
    node,
    name,
    type: "customtext",
    element: container,
    options: { hideOnZoom: true, getValue: () => "", setValue: () => {} },
  });
  widget.inputEl = list;
  addWidget(node, widget);

  function renderLibrary() {
    list.replaceChildren();
    if (!globalLibraryLoaded) {
      list.appendChild($el("div", { className: "zoey-library-hint", textContent: "加载素材库中…" }));
      loadGlobalLibrary().then(renderLibrary);
      return;
    }
    if (!globalLibrary.length) {
      list.appendChild($el("div", { className: "zoey-library-hint", textContent: "永久素材库：上传角色/道具/场景/音频，提示词里 @L1 调用" }));
    }
    globalLibrary.forEach((entry, i) => list.appendChild(libRow(entry, i)));
  }

  function uploadFileInto(entry, field) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = field === "audio_file" || entry.kind === "audio" ? "audio/*,video/*" : "image/*";
    input.onchange = () => {
      const f = input.files?.[0];
      if (!f) return;
      const fd = new FormData();
      fd.append("file", f, f.name);
      fd.append("kind", entry.kind || "");
      fetch(api.apiURL("/zoey/library/upload"), { method: "POST", body: fd })
        .then((r) => r.json())
        .then((d) => {
          if (d.filename) {
            entry[field] = d.filename;
            renderLibrary();
            scheduleSaveLibrary();
          } else {
            console.error("[Zoey MiniMax H3] upload failed:", d.error);
          }
        })
        .catch((err) => console.error("[Zoey MiniMax H3] upload:", err));
    };
    input.click();
  }

  function libRow(entry, i) {
    const row = document.createElement("div");
    row.className = "zoey-library-row";
    row.style.flexWrap = "wrap";

    const isAudio = entry.kind === "audio";
    const mainBtn = document.createElement("button");
    mainBtn.className = "zoey-library-slot";
    mainBtn.title = isAudio ? "点击上传音频" : "点击上传图片";
    if (entry.file) {
      if (isAudio) {
        mainBtn.textContent = "🔊";
      } else {
        const url = libraryMediaUrl(entry.file);
        const img = document.createElement("img");
        img.onerror = () => { mainBtn.textContent = "🧩"; };
        img.src = url;
        mainBtn.appendChild(img);
      }
    } else {
      mainBtn.textContent = "＋";
    }
    mainBtn.addEventListener("pointerdown", (e) => {
      e.preventDefault(); e.stopPropagation();
      uploadFileInto(entry, "file");
    });

    const kindSel = buildPopSelect({
      title: "素材类型（角色=人物参考 / 道具 / 场景 / 音频）",
      options: [["character", "角色"], ["prop", "道具"], ["scene", "场景"], ["audio", "音频"]]
        .map(([v, l]) => ({ value: v, label: l })),
      get: () => entry.kind,
      set: (k) => {
        entry.kind = k;
        renderLibrary();
        scheduleSaveLibrary();
      },
    });

    const name = document.createElement("input");
    name.value = entry.name || "";
    name.placeholder = "名称";
    name.addEventListener("input", () => { entry.name = name.value; });

    const appearance = document.createElement("input");
    appearance.value = entry.appearance || "";
    appearance.placeholder = "外貌备注";
    appearance.style.display = entry.kind === "character" ? "" : "none";
    appearance.addEventListener("input", () => { entry.appearance = appearance.value; });

    const voiceBtn = document.createElement("button");
    voiceBtn.className = "zoey-director-mini-btn";
    voiceBtn.textContent = entry.audio_file ? "🎙✓" : "🎙";
    voiceBtn.title = "上传角色语音";
    voiceBtn.style.display = entry.kind === "character" ? "" : "none";
    voiceBtn.addEventListener("pointerdown", (e) => {
      e.preventDefault(); e.stopPropagation();
      uploadFileInto(entry, "audio_file");
    });

    const desc = document.createElement("input");
    desc.className = "lib-desc";
    desc.value = entry.desc || "";
    desc.placeholder = "备注";
    desc.addEventListener("input", () => { entry.desc = desc.value; });

    const del = document.createElement("button");
    del.className = "zoey-director-mini-btn";
    del.textContent = "✕";
    del.title = "删除";
    del.addEventListener("pointerdown", async (e) => {
      e.preventDefault(); e.stopPropagation();
      const removed = globalLibrary.splice(i, 1)[0] || {};
      renderLibrary();
      await saveLibrary();
      for (const f of [removed.file, removed.audio_file]) {
        if (!f) continue;
        try {
          await api.fetchApi("/zoey/library/delete_file", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: f }),
          });
        } catch (err) { /* 孤儿文件删除失败不影响功能 */ }
      }
    });

    row.append(mainBtn, kindSel, name, appearance, voiceBtn, desc, del);
    return row;
  }

  renderLibrary();
  return widget;
}

// ---- LoRA 列表控件（默认一行，＋ 添加，强度与 lora 同行） ----
function createLorasWidget(node, name, inputData) {
  const container = document.createElement("div");
  container.className = "zoey-loras";

  const head = document.createElement("div");
  head.className = "zoey-director-header";
  const title = document.createElement("span");
  title.textContent = "🎚 LoRA";
  const addBtn = document.createElement("button");
  addBtn.textContent = "＋ 添加";
  addBtn.title = "添加一行 LoRA（名称 + 模型强度 + CLIP 强度）";
  addBtn.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); addRow(); });
  head.append(title, addBtn);

  const list = document.createElement("div");
  list.className = "zoey-loras-list";
  container.append(head, list);

  const widget = new DOMWidgetImpl({
    node,
    name,
    type: "customtext",
    element: container,
    options: {
      hideOnZoom: true,
      getValue: () => JSON.stringify(node._zoeyLoras || []),
      setValue: (v) => {
        try { node._zoeyLoras = JSON.parse(v || "[]") || []; } catch (e) { node._zoeyLoras = []; }
        renderLoras();
      },
    },
  });
  widget.inputEl = list;
  addWidget(node, widget);

  function loraOptions() {
    const listW = node.widgets?.find((w) => w.name === "lora_list");
    const vals = listW?.options?.values;
    return (Array.isArray(vals) && vals.length) ? vals : [];
  }

  function normalizeLoras() {
    if (!Array.isArray(node._zoeyLoras)) node._zoeyLoras = [];
    if (!node._zoeyLoras.length) node._zoeyLoras = [{ lora: "", model: 1.0, clip: 1.0 }];
    node._zoeyLoras.forEach((e) => {
      if (typeof e.model !== "number" || !isFinite(e.model)) e.model = 1.0;
      if (typeof e.clip !== "number" || !isFinite(e.clip)) e.clip = 1.0;
      if (typeof e.lora !== "string") e.lora = "";
    });
  }

  function addRow() {
    normalizeLoras();
    node._zoeyLoras.push({ lora: loraOptions()[0] || "", model: 1.0, clip: 1.0 });
    renderLoras();
  }

  function renderLoras() {
    normalizeLoras();
    list.replaceChildren();
    node._zoeyLoras.forEach((entry, i) => list.appendChild(loraRow(entry, i)));
  }

  function loraRow(entry, i) {
    const row = document.createElement("div");
    row.className = "zoey-lora-row";

    const opts = loraOptions();
    const sel = document.createElement("select");
    for (const o of opts) {
      const opt = document.createElement("option");
      opt.value = o; opt.textContent = o;
      sel.appendChild(opt);
    }
    if (entry.lora && opts.includes(entry.lora)) sel.value = entry.lora;
    else if (!opts.includes(entry.lora)) { entry.lora = opts[0] || ""; sel.value = entry.lora; }
    sel.addEventListener("change", () => { entry.lora = sel.value; });

    const modelLbl = document.createElement("label");
    modelLbl.textContent = "模型";
    const modelIn = document.createElement("input");
    modelIn.type = "number"; modelIn.step = "0.01"; modelIn.value = entry.model;
    modelIn.addEventListener("input", () => { entry.model = parseFloat(modelIn.value) || 0; });

    const clipLbl = document.createElement("label");
    clipLbl.textContent = "CLIP";
    const clipIn = document.createElement("input");
    clipIn.type = "number"; clipIn.step = "0.01"; clipIn.value = entry.clip;
    clipIn.addEventListener("input", () => { entry.clip = parseFloat(clipIn.value) || 0; });

    const del = document.createElement("button");
    del.className = "zoey-lora-del";
    del.textContent = "✕";
    del.title = "删除这一行";
    del.addEventListener("pointerdown", (e) => {
      e.preventDefault(); e.stopPropagation();
      node._zoeyLoras.splice(i, 1);
      normalizeLoras();
      renderLoras();
    });

    row.append(sel, modelLbl, modelIn, clipLbl, clipIn, del);
    return row;
  }

  renderLoras();
  return widget;
}

// 自愈：加载时把不在选项里的 combo 值、非数字的 float 值重置为默认，避免 schema 变更导致校验失败
function selfHealWidgets(node) {
  for (const w of node.widgets || []) {
    if (!w) continue;
    try {
      if (w.type === "combo" && Array.isArray(w.options?.values) && !w.options.values.includes(w.value)) {
        const dflt = w.options.default ?? w.options.values[0];
        w.value = w.options.values.includes(dflt) ? dflt : w.options.values[0];
        if (w.callback) { try { w.callback(w.value); } catch (e) {} }
      } else if (w.type === "number") {
        const n = parseFloat(w.value);
        if (typeof w.value !== "number" || !isFinite(n) || String(w.value).trim() === "") {
          const d = w.options?.default;
          w.value = (typeof d === "number" && isFinite(d)) ? d : 0;
          if (w.callback) { try { w.callback(w.value); } catch (e) {} }
        }
      }
    } catch (e) { /* 单个控件自愈失败不影响整体 */ }
  }
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

    // 阻止 Ctrl/Cmd+V 冒泡到全局 keybinding：contenteditable 编辑器不被全局识别为文本输入，
    // 否则粘贴提示词时会连带把之前复制的节点粘贴上（ComfyUI 自带 tiptap 文本控件同样用 stopPropagation）
    editor.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v") e.stopPropagation();
    });

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
        textContent: hasAny ? "没有匹配的参考素材" : "还没连接素材：把图片/视频拖到节点左侧 ref_image/ref_video 端口，提示词里用 @ 引用",
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
    const re = /@[PpVvAaCcLl]\d+/g;
    let m;
    while ((m = re.exec(text))) used.add(m[0].toUpperCase());
    const entries = collectEntries(this.node);
    const byTag = new Map(entries.map((e) => [e.tag, e]));
    const items = [...used].map((tag) => byTag.get(tag)).filter(Boolean);
    if (!items.length) {
      const hasAny = (this.node?.inputs || []).some((i) => i.link != null);
      this.strip.replaceChildren($el("div", {
        className: "zoey-ref-preview-hint",
        textContent: hasAny ? "在提示词里输入 @，选择已连接的素材（@P 图 / @V 视频 / @A 音频）" : "还没连接素材，先把图片/视频拖到左侧端口",
      }));
      return;
    }
    this.strip.replaceChildren(...items.map((it) => this.#previewItem(it)));
  }

  #previewItem(entry) {
    const tag = $el("span", { className: "zoey-ref-preview-tag", textContent: entry.tag });
    let item;
    if (entry.mediaType === "audio") {
      item = $el("div", { className: "zoey-ref-preview-item", title: entry.hint || entry.tag }, [
        $el("span", { className: "zoey-ref-preview-icon", textContent: "🔊" }),
        tag,
      ]);
    } else {
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
      item = $el("div", { className: "zoey-ref-preview-item", title: entry.hint || entry.tag }, [media, tag]);
    }
    attachHover(item, entry);
    // 已连接参考图（@P）：下方加用途标注下拉（官方要求，否则保主体不保背景）
    if (entry.kind === "图") {
      const node = this.node;
      const wrap = $el("div", { className: "zoey-ref-preview-wrap" });
      const btn = buildPopSelect({
        title: `${entry.tag} 用途标注（官方要求：不标则保主体不保背景）`,
        options: REF_PURPOSE_OPTIONS.map((o) => ({ value: o.key, label: o.label || "（不标注）" })),
        get: () => (node._zoeyRefPurposes || {})[entry.slot] || "",
        set: (key) => {
          if (!node._zoeyRefPurposes) node._zoeyRefPurposes = {};
          if (key) node._zoeyRefPurposes[entry.slot] = key;
          else delete node._zoeyRefPurposes[entry.slot];
          app.graph?.setDirtyCanvas(true);
          this.refreshPreview();
        },
      });
      btn.classList.add("zoey-ref-purpose");
      wrap.append(item, btn);
      return wrap;
    }
    return item;
  }

  hide() {
    HoverPreview.hide();
    if (this.dropdown) {
      this.dropdown.remove();
      this.dropdown = null;
    }
  }
}

// ---- 包装 STRING/FLOAT：prompt 换成富文本编辑器（含预览条），duration 换成秒计滑动条 ----
// 中文化参数标签（只改显示名，参数名不变，不影响工作流序列化）
const WIDGET_LABELS = {
  "resolution": "分辨率",
  "aspect": "比例",
  "duration": "时长",
  "ref_image_size": "参考图尺寸",
  "auto_declaration": "自动声明",
  "director_mode": "导演台",
  "dialogue_convert": "引号转对白",
  "mode": "模式",
};
function relabelWidget(node, inputName, widget) {
  if (!widget || node?.comfyClass !== NODE_TYPE) return widget;
  const label = WIDGET_LABELS[inputName];
  if (label) widget.label = label;
  return widget;
}

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
    if (node?.comfyClass === NODE_TYPE && inputName === "library") {
      return { widget: createLibraryWidget(node, inputName, inputData) };
    }
    if (node?.comfyClass === NODE_TYPE && inputName === "ref_purposes") {
      return { widget: createRefPurposesWidget(node, inputName, inputData) };
    }
    if (node?.comfyClass === NODE_TYPE && inputName === "loras") {
      const loraWidget = createLorasWidget(node, inputName, inputData);
      // 链到 onConfigure：所有控件值就绪后自愈一次（schema 变更导致的错位值）
      const prevConf = node.onConfigure;
      node.onConfigure = (...args) => {
        try { prevConf?.call(node, ...args); } catch (e) { console.error("[Zoey MiniMax H3] loras onConfigure:", e); }
        try { selfHealWidgets(node); } catch (e) { console.error("[Zoey MiniMax H3] selfHeal:", e); }
      };
      return { widget: loraWidget };
    }
    const strRes = origString.apply(this, arguments);
    relabelWidget(node, inputName, strRes?.widget);
    return strRes;
  };

  const origFloat = ComfyWidgets.FLOAT;
  ComfyWidgets.FLOAT = function (node, inputName, inputData, opts) {
    if (node?.comfyClass === NODE_TYPE && inputName === "duration") {
      return { widget: createDurationWidget(node, inputName, inputData) };
    }
    const fltRes = origFloat.apply(this, arguments);
    relabelWidget(node, inputName, fltRes?.widget);
    return fltRes;
  };

  const origCombo = ComfyWidgets.COMBO;
  ComfyWidgets.COMBO = function (node, inputName, inputData, opts) {
    if (node?.comfyClass === NODE_TYPE && inputName === "mode") {
      return { widget: createModeWidget(node, inputName, inputData) };
    }
    const cmbRes = origCombo.apply(this, arguments);
    relabelWidget(node, inputName, cmbRes?.widget);
    if (node?.comfyClass === NODE_TYPE && inputName === "lora_list" && cmbRes?.widget) {
      // 仅作为 LoRA 行下拉的选项源，不显示在节点上
      cmbRes.widget.hidden = true;
      if (cmbRes.widget.element) cmbRes.widget.element.style.display = "none";
    }
    return cmbRes;
  };

  const origBoolean = ComfyWidgets.BOOLEAN;
  ComfyWidgets.BOOLEAN = function (node, inputName, inputData, opts) {
    const res = origBoolean.apply(this, arguments);
    relabelWidget(node, inputName, res?.widget);
    if (node?.comfyClass === NODE_TYPE && inputName === "dialogue_convert") {
      const w = res?.widget;
      if (w) {
        const prevCb = w.callback;
        w.callback = (v) => {
          try { prevCb?.(v); } catch (e) { console.error("[Zoey MiniMax H3] dialogue_convert callback:", e); }
          node._zoeyDirector?.composePreview?.();
        };
      }
    }
    return res;
  };
}

function attachPicker(node, widget) {
  const tryAttach = () => {
    const el = widget.inputEl || widget.element;
    if (!el) return false;
    const picker = new RefPicker(node, widget, el);
    if (widget.strip) picker.attachStrip(widget.strip);
    // 用途下拉加载后需要重画预览条（隐藏的 ref_purposes 控件 setValue 时调用）
    node._zoeyRefreshRefStrip = () => { try { picker.refreshPreview(); } catch (e) { console.error("[Zoey MiniMax H3] refreshRefStrip:", e); } };

    const refresh = () => {
      try { picker.refreshPreview(); } catch (e) { console.error("[Zoey MiniMax H3] refreshPreview:", e); }
      // 素材库 widget 晚于 prompt 创建，@L 芯片需在加载后重渲染才有缩略图
      try {
        const text = serializePrompt(el);
        renderPrompt(el, text, node);
      } catch (e) { console.error("[Zoey MiniMax H3] rerenderPrompt:", e); }
      // 提示词区高度跟随节点（加载后按当前节点尺寸同步一次）
      try { node._zoeySyncPromptHeight?.(); } catch (e) { console.error("[Zoey MiniMax H3] syncPromptHeight:", e); }
    };
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
    // 全局素材库异步加载完成后刷新 @L 芯片缩略图
    const onLibLoaded = () => refresh();
    document.addEventListener("zoey:library-loaded", onLibLoaded);
    setTimeout(refresh, 200);

    const prevOnRemoved = node.onRemoved;
    node.onRemoved = () => {
      if (node._zoeySizePoller) { clearInterval(node._zoeySizePoller); node._zoeySizePoller = null; }
      document.removeEventListener("zoey:library-loaded", onLibLoaded);
      picker.hide();
      prevOnRemoved?.call(node);
    };
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
    document.addEventListener("focusin", (e) => {
      if (e.target && e.target.isContentEditable) lastRefEditor = e.target;
    });
  },
});
