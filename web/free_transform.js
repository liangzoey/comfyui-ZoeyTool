import { app } from "/scripts/app.js";

app.registerExtension({
    name: "zoey.freeTransform",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ZoeyFreeTransform") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const rv = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            const getW = (name) => node.widgets?.find(w => w.name === name);

            // ── Hide raw widgets ──
            ["corners", "grid_overlay"].forEach(name => {
                const w = getW(name);
                if (w) {
                    w.computeSize = () => [0, 0];
                    w.draw = () => {};
                    w.mouse = () => false;
                    if (w.element) w.element.style.display = "none";
                    else setTimeout(() => { if (w.element) w.element.style.display = "none"; }, 100);
                }
            });

            // ── State ──
            const s = {
                node, img: null, loaded: false, loadError: null, imgAspect: null,
                tl: {x:0, y:0}, tr: {x:1, y:0}, br: {x:1, y:1}, bl: {x:0, y:1},
                gridOverlay: getW("grid_overlay")?.value ?? true,
                mode: null, mx0: 0, my0: 0, corner0: null, drag0: null, lastInfo: null,
            };
            node._ftState = s;

            // ── DOM ──
            const root = document.createElement("div");
            root.style.cssText = "width:100%;display:flex;flex-direction:column;gap:4px;padding:2px 0;user-select:none;";

            const wrap = document.createElement("div");
            wrap.style.cssText = "width:100%;position:relative;background:#1a1a2e;border-radius:6px;overflow:hidden;border:1px solid #333;min-height:180px;height:220px;";

            const cv = document.createElement("canvas");
            cv.style.cssText = "width:100%;height:100%;display:block;cursor:crosshair;";
            wrap.appendChild(cv);
            root.appendChild(wrap);

            // ── Controls bar ──
            const bar = document.createElement("div");
            bar.style.cssText = "display:flex;align-items:center;gap:6px;padding:2px 0;flex-wrap:wrap;";

            const reloadBtn = document.createElement("button");
            reloadBtn.textContent = "⟳";
            reloadBtn.title = "重新加载图像";
            reloadBtn.style.cssText = "font-size:14px;padding:0 8px;border:1px solid #555;border-radius:4px;background:#2a2a3e;color:#ccc;cursor:pointer;height:26px;line-height:26px;";

            const resetBtn = document.createElement("button");
            resetBtn.textContent = "⊞";
            resetBtn.title = "重置透视变换";
            resetBtn.style.cssText = "font-size:14px;padding:0 8px;border:1px solid #4caf50;border-radius:4px;background:#1b5e20;color:#a5d6a7;cursor:pointer;height:26px;line-height:26px;";

            const gridBtn = document.createElement("button");
            gridBtn.textContent = s.gridOverlay ? "⧉" : "⊞";
            gridBtn.title = s.gridOverlay ? "隐藏网格" : "显示网格";
            gridBtn.style.cssText = `font-size:14px;padding:0 8px;border:1px solid ${s.gridOverlay ? "#ffb74d" : "#555"};border-radius:4px;background:${s.gridOverlay ? "#5c3d0e" : "#2a2a3e"};color:${s.gridOverlay ? "#ffb74d" : "#888"};cursor:pointer;height:26px;line-height:26px;`;

            const infoLbl = document.createElement("span");
            infoLbl.style.cssText = "font-size:9px;font-family:monospace;color:#666;white-space:nowrap;flex:1;text-align:right;";

            bar.appendChild(reloadBtn);
            bar.appendChild(resetBtn);
            bar.appendChild(gridBtn);
            bar.appendChild(infoLbl);
            root.appendChild(bar);

            // ── Button events ──
            resetBtn.addEventListener("click", () => {
                s.tl = {x:0, y:0}; s.tr = {x:1, y:0}; s.br = {x:1, y:1}; s.bl = {x:0, y:1};
                syncW(); draw();
            });

            gridBtn.addEventListener("click", () => {
                s.gridOverlay = !s.gridOverlay;
                const gw = getW("grid_overlay");
                if (gw) { gw.value = s.gridOverlay; if (gw.callback) gw.callback(s.gridOverlay); }
                gridBtn.textContent = s.gridOverlay ? "⧉" : "⊞";
                gridBtn.title = s.gridOverlay ? "隐藏网格" : "显示网格";
                gridBtn.style.borderColor = s.gridOverlay ? "#ffb74d" : "#555";
                gridBtn.style.color = s.gridOverlay ? "#ffb74d" : "#888";
                gridBtn.style.background = s.gridOverlay ? "#5c3d0e" : "#2a2a3e";
                draw();
            });

            reloadBtn.addEventListener("click", () => {
                retryLoad(1, 0);
            });

            // ── Coordinate helpers ──
            function vp() {
                const pad = 0.15;
                const xs = [s.tl.x, s.tr.x, s.br.x, s.bl.x, 0, 1];
                const ys = [s.tl.y, s.tr.y, s.br.y, s.bl.y, 0, 1];
                const minX = Math.min(...xs) - pad;
                const maxX = Math.max(...xs) + pad;
                const minY = Math.min(...ys) - pad;
                const maxY = Math.max(...ys) + pad;
                return { vl: minX, vt: minY, vr: maxX, vb: maxY, vw: maxX - minX, vh: maxY - minY };
            }

            function n2c(nx, ny, info) {
                return { x: info.ox + (nx - info.vl) * info.scX, y: info.oy + (ny - info.vt) * info.scY };
            }
            function c2n(cx, cy, info) {
                return { nx: (cx - info.ox) / info.scX + info.vl, ny: (cy - info.oy) / info.scY + info.vt };
            }

            // ── Drawing ──
            function draw() {
                const rect = wrap.getBoundingClientRect();
                if (rect.width < 10 || rect.height < 10) return;
                const dpr = window.devicePixelRatio || 1;
                cv.width = rect.width * dpr;
                cv.height = rect.height * dpr;
                const ctx = cv.getContext("2d");
                ctx.scale(dpr, dpr);
                const cw = rect.width, ch = rect.height;

                ctx.fillStyle = "#1a1a2e";
                ctx.fillRect(0, 0, cw, ch);

                if (!s.loaded || !s.img) {
                    ctx.fillStyle = "#555";
                    ctx.font = "13px sans-serif";
                    ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    if (s.loadError) {
                        ctx.fillStyle = "#ff6b6b";
                        ctx.fillText("加载失败: " + s.loadError, cw / 2, ch / 2 - 8);
                        ctx.fillStyle = "#888"; ctx.font = "11px sans-serif";
                        ctx.fillText("点击 ⟳ 重试，或检查控制台日志", cw / 2, ch / 2 + 14);
                    } else {
                        ctx.fillText("连接图像源后实时预览", cw / 2, ch / 2 - 8);
                        ctx.font = "11px sans-serif"; ctx.fillStyle = "#444";
                        ctx.fillText("拖拽彩色角点进行透视变换", cw / 2, ch / 2 + 14);
                    }
                    s.lastInfo = null;
                    return;
                }

                const { vl, vt, vw, vh } = vp();
                const imgAspect = s.imgAspect || 1;
                let scX, scY;
                if (imgAspect >= 1) {
                    scX = cw / vw;
                    scY = scX / imgAspect;
                    if (scY * vh > ch) { scY = ch / vh; scX = scY * imgAspect; }
                } else {
                    scY = ch / vh;
                    scX = scY * imgAspect;
                    if (scX * vw > cw) { scX = cw / vw; scY = scX / imgAspect; }
                }
                const ox = (cw - vw * scX) / 2;
                const oy = (ch - vh * scY) / 2;
                const info = { vl, vt, vw, vh, scX, scY, ox, oy };
                s.lastInfo = info;

                ctx.fillStyle = "#0d0d1a";
                ctx.fillRect(0, 0, cw, ch);

                // Draw image stretched to quad
                const pts = [
                    n2c(s.tl.x, s.tl.y, info),
                    n2c(s.tr.x, s.tr.y, info),
                    n2c(s.br.x, s.br.y, info),
                    n2c(s.bl.x, s.bl.y, info),
                ];

                ctx.save();
                ctx.beginPath();
                ctx.moveTo(pts[0].x, pts[0].y);
                ctx.lineTo(pts[1].x, pts[1].y);
                ctx.lineTo(pts[2].x, pts[2].y);
                ctx.lineTo(pts[3].x, pts[3].y);
                ctx.closePath();
                ctx.clip();

                // Draw image within quad
                ctx.save();
                ctx.transform(
                    pts[1].x - pts[0].x, pts[1].y - pts[0].y,
                    pts[3].x - pts[0].x, pts[3].y - pts[0].y,
                    pts[0].x, pts[0].y
                );
                ctx.drawImage(s.img, 0, 0, 1, 1);
                ctx.restore();

                // Fill outside quad with dark overlay
                ctx.restore();
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, 0, cw, ch);
                ctx.moveTo(pts[0].x, pts[0].y);
                ctx.lineTo(pts[1].x, pts[1].y);
                ctx.lineTo(pts[2].x, pts[2].y);
                ctx.lineTo(pts[3].x, pts[3].y);
                ctx.closePath();
                ctx.fillStyle = "rgba(0,0,0,0.55)";
                ctx.fill("evenodd");
                ctx.restore();

                // Grid overlay
                if (s.gridOverlay) {
                    ctx.save();
                    ctx.strokeStyle = "rgba(255,255,255,0.25)";
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 3]);

                    function lerp(a, b, t) {
                        return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
                    }

                    for (let i = 1; i < 4; i++) {
                        const t = i / 4;
                        const top = lerp(pts[0], pts[1], t);
                        const bot = lerp(pts[3], pts[2], t);
                        const left = lerp(pts[0], pts[3], t);
                        const right = lerp(pts[1], pts[2], t);
                        ctx.beginPath();
                        ctx.moveTo(top.x, top.y); ctx.lineTo(bot.x, bot.y);
                        ctx.moveTo(left.x, left.y); ctx.lineTo(right.x, right.y);
                        ctx.stroke();
                    }
                    ctx.setLineDash([]);
                    ctx.restore();
                }

                // Quad outline
                ctx.save();
                ctx.strokeStyle = "#4fc3f7";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(pts[0].x, pts[0].y);
                ctx.lineTo(pts[1].x, pts[1].y);
                ctx.lineTo(pts[2].x, pts[2].y);
                ctx.lineTo(pts[3].x, pts[3].y);
                ctx.closePath();
                ctx.stroke();
                ctx.restore();

                // Corner handles with colors and labels
                const CORNERS = [
                    { pt: pts[0], color: "#ff5252", label: "TL" },
                    { pt: pts[1], color: "#69f0ae", label: "TR" },
                    { pt: pts[2], color: "#448aff", label: "BR" },
                    { pt: pts[3], color: "#ffd740", label: "BL" },
                ];

                const hs = 12;
                CORNERS.forEach(c => {
                    ctx.save();
                    ctx.fillStyle = c.color;
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(c.pt.x, c.pt.y, hs / 2, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();

                    // Label
                    ctx.fillStyle = "#fff";
                    ctx.font = "bold 8px monospace";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "bottom";
                    ctx.fillText(c.label, c.pt.x, c.pt.y - hs / 2 - 3);
                    ctx.restore();
                });

                // Info
                const numLbl = `${s.img.naturalWidth}×${s.img.naturalHeight}`;
                ctx.fillStyle = "rgba(255,255,255,0.4)";
                ctx.font = "10px sans-serif";
                ctx.textAlign = "left"; ctx.textBaseline = "top";
                ctx.fillText(numLbl, 6, 4);
                infoLbl.textContent = `TL(${s.tl.x.toFixed(2)},${s.tl.y.toFixed(2)})  TR(${s.tr.x.toFixed(2)},${s.tr.y.toFixed(2)})`;
            }

            // ── Widget sync ──
            function syncW() {
                const corners = { tl: s.tl, tr: s.tr, br: s.br, bl: s.bl };
                const w = getW("corners");
                if (w) {
                    w.value = JSON.stringify(corners);
                    if (w.callback) w.callback(w.value);
                }
                app.graph.setDirtyCanvas(true, true);
            }

            // ── Hit testing ──
            function getCornerHit(mx, my) {
                if (!s.lastInfo) return -1;
                const CORNERS = [
                    n2c(s.tl.x, s.tl.y, s.lastInfo),
                    n2c(s.tr.x, s.tr.y, s.lastInfo),
                    n2c(s.br.x, s.br.y, s.lastInfo),
                    n2c(s.bl.x, s.bl.y, s.lastInfo),
                ];
                for (let i = 0; i < 4; i++) {
                    const dx = mx - CORNERS[i].x;
                    const dy = my - CORNERS[i].y;
                    if (Math.sqrt(dx * dx + dy * dy) < 14) return i;
                }
                return -1;
            }

            function inQuad(mx, my) {
                if (!s.lastInfo) return false;
                const pts = [
                    n2c(s.tl.x, s.tl.y, s.lastInfo),
                    n2c(s.tr.x, s.tr.y, s.lastInfo),
                    n2c(s.br.x, s.br.y, s.lastInfo),
                    n2c(s.bl.x, s.bl.y, s.lastInfo),
                ];
                // Point-in-convex-quad using cross products
                function cross(ax, ay, bx, by) { return ax * by - ay * bx; }
                function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
                let pos = 0, neg = 0;
                for (let i = 0; i < 4; i++) {
                    const j = (i + 1) % 4;
                    const d = sub(pts[j], pts[i]);
                    const v = sub({ x: mx, y: my }, pts[i]);
                    const cp = cross(d.x, d.y, v.x, v.y);
                    if (cp > 0.1) pos = 1;
                    else if (cp < -0.1) neg = 1;
                    if (pos && neg) return false;
                }
                return true;
            }

            // ── Interaction ──
            cv.addEventListener("mousedown", (e) => {
                const rect = cv.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;

                const hitIdx = getCornerHit(mx, my);
                if (hitIdx >= 0) {
                    s.mode = "corner";
                    s.cornerIdx = hitIdx;
                    s.mx0 = mx; s.my0 = my;
                    const names = ["tl", "tr", "br", "bl"];
                    s.corner0 = { ...s[names[hitIdx]] };
                    e.preventDefault();
                    return;
                }

                if (inQuad(mx, my)) {
                    s.mode = "move";
                    s.mx0 = mx; s.my0 = my;
                    s.drag0 = {
                        tl: { ...s.tl }, tr: { ...s.tr },
                        br: { ...s.br }, bl: { ...s.bl },
                    };
                    cv.style.cursor = "grabbing";
                    e.preventDefault();
                    return;
                }
            });

            const onMM = (e) => {
                const rect = cv.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;

                if (!s.lastInfo || !s.mode) {
                    const onHit = getCornerHit(mx, my);
                    cv.style.cursor = onHit >= 0 ? "pointer" : inQuad(mx, my) ? "grab" : "crosshair";
                    return;
                }

                const p = c2n(mx, my, s.lastInfo);

                if (s.mode === "corner") {
                    const names = ["tl", "tr", "br", "bl"];
                    const idx = s.cornerIdx;
                    s[names[idx]] = {
                        x: p.nx,
                        y: p.ny,
                    };
                    syncW(); draw(); e.preventDefault();
                    return;
                }

                if (s.mode === "move") {
                    const p0 = c2n(s.mx0, s.my0, s.lastInfo);
                    const dx = p.nx - p0.nx;
                    const dy = p.ny - p0.ny;
                    const d0 = s.drag0;
                    s.tl = { x: d0.tl.x + dx, y: d0.tl.y + dy };
                    s.tr = { x: d0.tr.x + dx, y: d0.tr.y + dy };
                    s.br = { x: d0.br.x + dx, y: d0.br.y + dy };
                    s.bl = { x: d0.bl.x + dx, y: d0.bl.y + dy };
                    syncW(); draw(); e.preventDefault();
                }
            };

            const onMU = () => {
                if (s.mode) {
                    s.mode = null;
                    cv.style.cursor = "crosshair";
                }
            };
            window.addEventListener("mousemove", onMM);
            window.addEventListener("mouseup", onMU);

            // ── Image loading ──
            function loadImage(url) {
                s.loadError = null;
                const img = new Image();
                if (url?.startsWith("blob:") || url?.startsWith("data:")) img.crossOrigin = "anonymous";
                img.onload = () => {
                    s.img = img; s.loaded = true; s.loadError = null;
                    s.imgAspect = img.naturalWidth / img.naturalHeight;
                    draw();
                };
                img.onerror = () => { s.loadError = "HTTP error"; draw(); };
                img.src = url;
            }

            function findSourceNode() {
                const inp = s.node.inputs?.find(i => i.name === "image");
                if (!inp || inp.link == null) return null;
                const tid = inp.link;
                for (const n of (app.graph._nodes || [])) {
                    if (n === s.node || !n.outputs) continue;
                    for (const o of n.outputs) {
                        if (!o.links) continue;
                        for (const lid of o.links) if (lid != null && lid === tid) return n;
                    }
                }
                try {
                    const ld = app.graph.links?.[tid];
                    if (ld) {
                        const id = Array.isArray(ld) ? ld[0] : ld?.origin_id ?? ld?.[0];
                        return app.graph.getNodeById(id);
                    }
                } catch (e) {}
                try {
                    const links = app.graph.links;
                    if (links) for (const [lid, ld] of Object.entries(links)) {
                        if (Number(lid) === tid) {
                            const id = Array.isArray(ld) ? ld[0] : ld?.origin_id ?? ld?.[0];
                            return app.graph.getNodeById(id);
                        }
                    }
                } catch (e) {}
                return null;
            }

            function loadImageViaFetch(url) {
                return fetch(url).then(r => r.ok ? r.blob() : null).then(b => b ? URL.createObjectURL(b) : null).catch(() => null);
            }

            function tryLoadFromSource() {
                const src = findSourceNode();
                if (!src) return false;
                let url = null;
                if (src.imgs?.length > 0) {
                    const el = src.imgs[0];
                    if (typeof el === "string") url = el;
                    else if (el?.src) url = el.src;
                    else if (el?._src) url = el._src;
                    else if (el?.tagName === "CANVAS") url = el.toDataURL?.("image/png");
                    if (url) { loadImage(url); return true; }
                }
                if (src.image) {
                    url = typeof src.image === "string" ? src.image : src.image?.src || src.image?._src;
                    if (url) { loadImage(url); return true; }
                }
                const imgW = src.widgets?.find(w => w.name === "image");
                if (imgW) {
                    let fn = "", sub = "", tp = "input";
                    if (typeof imgW.value === "string") {
                        const parts = imgW.value.split("/"); fn = parts.pop() || ""; sub = parts.join("/");
                    } else if (imgW.value && typeof imgW.value === "object") {
                        fn = imgW.value.filename || ""; sub = imgW.value.subfolder || ""; tp = imgW.value.type || "input";
                    }
                    if (fn) {
                        url = `${location.origin}/view?filename=${encodeURIComponent(fn)}&type=${tp}${sub ? "&subfolder="+encodeURIComponent(sub) : ""}&rand=${Date.now()}`;
                        loadImageViaFetch(url).then(u => { if (u) loadImage(u); });
                        return true;
                    }
                }
                for (const w of (src.widgets || [])) {
                    let v = w.value;
                    if (typeof v === "object" && v) v = v.filename || v.name || "";
                    if (typeof v === "string" && /\.(png|jpg|jpeg|webp|bmp)$/i.test(v)) {
                        const parts = v.split("/"); const fn = parts.pop();
                        const sub = parts.join("/");
                        url = `${location.origin}/view?filename=${encodeURIComponent(fn)}&type=input${sub ? "&subfolder="+encodeURIComponent(sub) : ""}&rand=${Date.now()}`;
                        loadImageViaFetch(url).then(u => { if (u) loadImage(u); });
                        return true;
                    }
                }
                return false;
            }

            function retryLoad(maxTries, delay) {
                let tries = 0;
                const at = () => {
                    tries++;
                    if (tryLoadFromSource()) return;
                    if (tries >= maxTries) { s.loadError = "重试耗尽"; draw(); return; }
                    setTimeout(at, delay);
                };
                at();
            }

            // ── Configure (workflow restore) ──
            const origCfg = this.configure;
            this.configure = function (data) {
                if (origCfg) origCfg.apply(this, arguments);
                const st = this._ftState;
                if (!st) return;
                try {
                    const raw = getW("corners")?.value || "{}";
                    const cfg = JSON.parse(raw);
                    if (cfg.tl) st.tl = cfg.tl;
                    if (cfg.tr) st.tr = cfg.tr;
                    if (cfg.br) st.br = cfg.br;
                    if (cfg.bl) st.bl = cfg.bl;
                } catch (e) {}
                st.gridOverlay = getW("grid_overlay")?.value ?? true;
                gridBtn.textContent = st.gridOverlay ? "⧉" : "⊞";
                gridBtn.title = st.gridOverlay ? "隐藏网格" : "显示网格";
                gridBtn.style.borderColor = st.gridOverlay ? "#ffb74d" : "#555";
                gridBtn.style.color = st.gridOverlay ? "#ffb74d" : "#888";
                gridBtn.style.background = st.gridOverlay ? "#5c3d0e" : "#2a2a3e";
                setTimeout(() => { retryLoad(20, 300); }, 500);
            };

            // ── On executed ──
            const origExec = this.onExecuted;
            this.onExecuted = function (msg) {
                if (origExec) origExec.apply(this, arguments);
                const st = this._ftState;
                if (!st) return;
                setTimeout(() => retryLoad(10, 300), 200);
            };

            // ── DOM widget ──
            const widget = this.addDOMWidget("ft_preview", "FT_PREVIEW", root, {
                getValue() { return ""; }, setValue() {},
            });
            widget.computeSize = (width) => {
                const w = Math.max(width || 350, 280);
                const h = Math.max(180, Math.floor(w * 0.72));
                wrap.style.height = h + "px";
                return [w, h + 38];
            };
            this.setSize([350, 260]);

            // ── ResizeObserver ──
            let rt;
            const ro = new ResizeObserver(() => {
                if (rt) cancelAnimationFrame(rt);
                rt = requestAnimationFrame(() => { draw(); rt = null; });
            });
            ro.observe(wrap);

            // ── Cleanup ──
            const origRM = this.onRemoved;
            this.onRemoved = function () {
                ro.disconnect();
                window.removeEventListener("mousemove", onMM);
                window.removeEventListener("mouseup", onMU);
                if (rt) cancelAnimationFrame(rt);
                this._ftState = null;
                if (origRM) origRM.apply(this, arguments);
            };

            this.setSize([350, 260]);
            setTimeout(() => retryLoad(12, 400), 300);
            return rv;
        };
    },
});
