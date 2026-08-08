import { app } from "/scripts/app.js";

const NODE = "ZoeySystemMonitor";
const STATS_URL = "http://127.0.0.1:18888/stats";
const CLEANUP_URL = "http://127.0.0.1:18888/cleanup";

const pal = v => v == null ? "#555" : v < 60 ? "#22c55e" : v < 85 ? "#eab308" : "#ef4444";
const pal2 = v => v == null ? "#555" : v < 60 ? "#4ade80" : v < 85 ? "#facc15" : "#f87171";

function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function updateNodes(stats) {
  const nodes = app.graph?._nodes || [];
  for (const node of nodes) {
    if (node.type !== NODE) continue;
    node._mon = { ...stats };
    node._monDirty = true;
    node.setDirtyCanvas(true, true);
  }
  app.graph?.setDirtyCanvas?.(true);
}

function triggerCleanup() {
  fetch(CLEANUP_URL).then(r => r.json()).then(s => { if (s) updateNodes(s); }).catch(() => {});
}

app.registerExtension({
  name: "zoey.systemMonitor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const oc = nodeType.prototype.onNodeCreated;
    const oe = nodeType.prototype.onExecuted;
    const os = nodeType.prototype.computeSize;
    const od = nodeType.prototype.onDrawForeground;

    nodeType.prototype.onNodeCreated = function() {
      const r = oc?.apply(this, arguments) ?? undefined;
      const w = this.widgets?.find(w => w.name === "_cached");
      if (w?.value) { try { this._mon = JSON.parse(w.value); } catch(e) {} }
      this._monDirty = true;
      return r;
    };

    nodeType.prototype.onExecuted = function(msg) {
      try { oe?.apply(this, arguments); } catch(e) {}
      try {
        const raw = msg?.ui?.monitor?.[0] ?? msg?.monitor?.[0];
        if (raw) {
          this._mon = typeof raw === "string" ? JSON.parse(raw) : raw;
          this._monDirty = true;
          this.setDirtyCanvas(true, true);
        }
      } catch(e) {}
    };

    nodeType.prototype.computeSize = function(...args) {
      const s = os?.apply(this, args) ?? [220, 100];
      s[1] = Math.max(s[1], 230);
      if (this._mon) {
        let r = 4;
        if (this._mon.cpu != null) r++;
        if (this._mon.ram != null) r++;
        if (this._mon.vram != null) r++;
        if (this._mon.gpu_temp != null) r++;
        if (this._mon.cleaned && this._mon.freed_gb) r++;
        s[1] = Math.max(s[1], 85 + r * 23);
      }
      return s;
    };

    nodeType.prototype.onDrawForeground = function(ctx) {
      od?.apply(this, arguments);
      const d = this._mon;
      if (!d && !this._monDirty) return;

      const W = this.size[0] - 10;

      let top = 28;
      if (this.widgets) {
        for (const w of this.widgets) {
          if (w.name?.startsWith("_") || w.type === "hidden") continue;
          if (w.y == null) continue;
          let h = 22;
          try { const s = w.computeSize?.(this.size[0]); h = Array.isArray(s) ? (s[1] ?? 20) : (s ?? 20); } catch(e) {}
          top = Math.max(top, w.y + h + 4);
        }
      }
      if (top <= 28) top = 110;

      const cy = top + 4;
      const ch = this.size[1] - cy - 6;
      if (ch < 24) return;

      // ── Panel bg ──
      ctx.fillStyle = "rgba(18,22,30,0.92)";
      rrect(ctx, 5, cy - 2, W, ch, 10);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      rrect(ctx, 5, cy - 2, W, ch, 10);
      ctx.stroke();

      // ── Accent bar ──
      ctx.fillStyle = d ? (d.cleaned ? "#22c55e" : d.vram >= d.threshold ? "#ef4444" : "#6366f1") : "#222";
      rrect(ctx, 5, cy - 2, W, 3, 2);
      ctx.fill();

      let y = cy + 16;
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "600 10px system-ui,sans-serif";
      ctx.fillText("SYSTEM MONITOR", 14, y);
      ctx.textAlign = "right";
      ctx.fillStyle = "rgba(255,255,255,0.25)";
      ctx.font = "8px monospace";
      ctx.fillText(new Date().toLocaleTimeString("zh-CN"), W - 4, y);
      ctx.textAlign = "left";
      y += 12;

      ctx.fillStyle = "rgba(255,255,255,0.04)";
      ctx.fillRect(14, y, W - 26, 1);
      y += 10;

      // ── Metric rows ──
      const bX = 40, bW = W - 56, bH = 4;

      const drawRow = (label, val, detail) => {
        const v = Math.min(val ?? 0, 100);
        const c2 = pal2(val ?? 0);
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.font = "9px system-ui,sans-serif";
        ctx.fillText(label, 14, y + 8);
        ctx.fillStyle = "rgba(255,255,255,0.04)";
        rrect(ctx, bX, y + 2, bW, bH, 2);
        ctx.fill();
        if (d) {
          const fw = Math.max(2, (bW - 16) * v / 100);
          ctx.fillStyle = pal(val ?? 0);
          rrect(ctx, bX, y + 2, fw, bH, 2);
          ctx.fill();
          ctx.globalAlpha = 0.35;
          ctx.fillStyle = c2;
          ctx.fillRect(bX, y + 2, fw, bH / 2);
          ctx.globalAlpha = 1;
        }
        ctx.fillStyle = d ? c2 : "rgba(255,255,255,0.12)";
        ctx.font = "600 9px monospace";
        ctx.textAlign = "right";
        ctx.fillText(d ? (detail || val.toFixed(1) + "%") : "--", W - 4, y + 8);
        ctx.textAlign = "left";
        y += 22;
      };

      drawRow("CPU", d?.cpu ?? 0);
      drawRow("RAM", d?.ram ?? 0, d ? `${d.ram_used?.toFixed(1)}/${d.ram_total?.toFixed(0)}GB` : "");
      drawRow("VRAM", d?.vram ?? 0, d ? `${d.vram_used?.toFixed(1)}/${d.vram_total?.toFixed(0)}GB` : "");
      // GPU 行：优先显示真实利用率，温度为详情
      const gpuVal = d?.gpu_util != null
        ? Math.min(d.gpu_util, 100)
        : (d?.gpu_temp != null ? Math.min(d.gpu_temp, 100) : 0);
      const gpuDetail = d ? (d.gpu_util != null
        ? `${d.gpu_util.toFixed(0)}% · ${d.gpu_temp ?? "--"}°C`
        : (d.gpu_temp != null ? `${d.gpu_temp}°C` : "")) : "";
      drawRow("GPU", gpuVal, gpuDetail);

      if (!d) {
        y += 6;
        ctx.fillStyle = "rgba(255,255,255,0.08)";
        ctx.font = "9px system-ui,sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("等待数据...", W / 2, y + 8);
        ctx.textAlign = "left";
        this._monDirty = false;
        return;
      }

      y += 4;
      const isAlarm = d.vram >= d.threshold;
      const isClean = d.cleaned;
      const dotC = isClean ? "#22c55e" : isAlarm ? "#ef4444" : "#6366f1";
      ctx.fillStyle = dotC;
      ctx.beginPath();
      ctx.arc(20, y + 7, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = isClean ? "#4ade80" : isAlarm ? "#f87171" : "rgba(255,255,255,0.55)";
      ctx.font = "9px system-ui,sans-serif";
      ctx.fillText(isClean ? "已清理" : isAlarm ? "显存告警" : "运行正常", 28, y + 10);
      if (isClean && d.freed_gb) {
        ctx.fillStyle = "rgba(74,222,128,0.7)";
        ctx.font = "8px system-ui,sans-serif";
        ctx.textAlign = "right";
        ctx.fillText("释放 " + d.freed_gb.toFixed(2) + " GB", W - 4, y + 10);
        ctx.textAlign = "left";
      }

      y += 22;
      ctx.fillStyle = "rgba(255,255,255,0.18)";
      ctx.font = "8px system-ui,sans-serif";
      ctx.fillText(
        `GPU ${d.gpu_util != null ? d.gpu_util.toFixed(0) + "%" : "--"} · ${d.gpu_temp != null ? d.gpu_temp + "°C" : "--"} · VRAM ${d.vram != null ? d.vram.toFixed(0) + "%" : "--"} · RAM ${d.ram != null ? d.ram.toFixed(0) + "%" : "--"}`,
        14, y + 8
      );

      this._monDirty = false;
    };
  },
});

// ── Polling ──
(function() {
  function poll() {
    fetch(STATS_URL).then(r => r.json()).then(s => { if (s) updateNodes(s); }).catch(() => {});
  }
  setTimeout(poll, 500);
  setInterval(poll, 2000);
})();

// ── Cleanup on execution end ──
(function() {
  function hookWS() {
    const ws = app.api?.socket;
    if (!ws) { setTimeout(hookWS, 500); return; }
    const orig = ws.onmessage;
    ws.onmessage = function(event) {
      try { if (typeof orig?.call === 'function') orig.call(this, event); } catch(e) {}
      if (typeof event.data !== 'string') return;
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "execution_success" || msg.type === "execution_cached") {
          setTimeout(triggerCleanup, 1000);
        }
      } catch(e) {}
    };
  }
  setTimeout(hookWS, 1000);
})();
