import { app } from "/scripts/app.js";

app.registerExtension({
    name: "zoey.outpaintFrame",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ZoeyOutpaintFrame") return;

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot, isConnected, link, outputSlot) {
            if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
            const st = this._opState;
            if (!st) return;
            if (slotType === 1 && this.inputs[slot]?.name === "image") {
                if (isConnected) st._loadRetry(20, 300);
                else { st.loaded = false; st.img = null; if (st._opDraw) st._opDraw(); }
            }
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const rv = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            const getW = (name) => node.widgets?.find(w => w.name === name);

            // ── Hide raw widgets ──
            ["frame_left", "frame_top", "frame_right", "frame_bottom", "填充颜色", "fill_mode", "feather"].forEach(name => {
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
                fl: getW("frame_left")?.value ?? -0.1,
                ft: getW("frame_top")?.value ?? -0.1,
                fr: getW("frame_right")?.value ?? 1.1,
                fb: getW("frame_bottom")?.value ?? 1.1,
                fill: getW("填充颜色")?.value ?? "#808080",
                fillMode: getW("fill_mode")?.value ?? true,
                feather: getW("feather")?.value ?? 0,
                mode: null, mx0: 0, my0: 0, frame0: null, lastInfo: null, _hits: null,
            };
            node._opState = s;

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

            // Reload button (first)
            const reloadBtn = document.createElement("button");
            reloadBtn.textContent = "⟳";
            reloadBtn.title = "重新加载图像";
            reloadBtn.style.cssText = "font-size:14px;padding:0 8px;border:1px solid #555;border-radius:4px;background:#2a2a3e;color:#ccc;cursor:pointer;height:26px;line-height:26px;";

            // Fit-to-image button
            const fitBtn = document.createElement("button");
            fitBtn.textContent = "⊞";
            fitBtn.title = "框架贴合图像边缘";
            fitBtn.style.cssText = "font-size:14px;padding:0 8px;border:1px solid #4caf50;border-radius:4px;background:#1b5e20;color:#a5d6a7;cursor:pointer;height:26px;line-height:26px;";

            // Fill mode toggle
            const fillBtn = document.createElement("button");
            fillBtn.textContent = s.fillMode ? "◧" : "○";
            fillBtn.title = s.fillMode ? "裁剪模式：开启后图像尺寸不变，裁剪区域自动填充颜色" : "裁剪模式：关闭后只裁剪不外扩不填充";
            fillBtn.style.cssText = `font-size:14px;padding:0 8px;border:1px solid ${s.fillMode ? "#4fc3f7" : "#555"};border-radius:4px;background:${s.fillMode ? "#0d3b5e" : "#2a2a3e"};color:${s.fillMode ? "#4fc3f7" : "#888"};cursor:pointer;height:26px;line-height:26px;`;

            // Swatch + color picker
            const swatch = document.createElement("div");
            swatch.style.cssText = "width:26px;height:26px;border-radius:4px;border:2px solid #555;cursor:pointer;flex-shrink:0;";
            swatch.style.backgroundColor = s.fill;
            swatch.title = "点击选择填充色";

            const hexLbl = document.createElement("span");
            hexLbl.style.cssText = "font-size:10px;font-family:monospace;color:#888;min-width:42px;";
            hexLbl.textContent = s.fill.toUpperCase();

            const ci = document.createElement("input");
            ci.type = "color";
            ci.value = s.fill;
            ci.style.cssText = "width:0;height:0;padding:0;border:none;position:absolute;opacity:0;pointer-events:none;";

            swatch.addEventListener("click", () => ci.click());
            ci.addEventListener("input", () => {
                s.fill = ci.value;
                swatch.style.backgroundColor = s.fill;
                hexLbl.textContent = s.fill.toUpperCase();
                const w = getW("填充颜色");
                if (w) { w.value = s.fill; if (w.callback) w.callback(s.fill); }
                draw();
            });

            // Dimension info
            const dimLabel = document.createElement("span");
            dimLabel.style.cssText = "font-size:9px;font-family:monospace;color:#666;white-space:nowrap;";
            function updateDimLabel() {
                const w = s.fr - s.fl;
                const h = s.fb - s.ft;
                dimLabel.textContent = `${w.toFixed(2)}×${h.toFixed(2)} (${(w/h).toFixed(2)}:1)`;
            }

            // ── Feather control ──
            const featherLbl = document.createElement("span");
            featherLbl.textContent = "羽化";
            featherLbl.style.cssText = "font-size:9px;color:#888;margin-left:4px;flex:none;";

            const featherVal = document.createElement("span");
            featherVal.textContent = `${s.feather}px`;
            featherVal.style.cssText = "font-size:10px;font-family:monospace;color:#aaa;width:30px;text-align:right;flex:none;";

            const featherSlider = document.createElement("input");
            featherSlider.type = "range";
            featherSlider.min = 0; featherSlider.max = 200; featherSlider.value = s.feather;
            featherSlider.style.cssText = "width:52px;height:14px;cursor:pointer;flex:none;";
            featherSlider.title = "边缘羽化像素值";

            featherSlider.addEventListener("input", () => {
                s.feather = parseInt(featherSlider.value);
                featherVal.textContent = `${s.feather}px`;
                const fw = getW("feather");
                if (fw) { fw.value = s.feather; if (fw.callback) fw.callback(s.feather); }
                draw();
            });

            bar.appendChild(reloadBtn);
            bar.appendChild(fitBtn);
            bar.appendChild(fillBtn);
            bar.appendChild(swatch);
            bar.appendChild(hexLbl);
            bar.appendChild(ci);
            bar.appendChild(dimLabel);
            bar.appendChild(featherLbl);
            bar.appendChild(featherSlider);
            bar.appendChild(featherVal);
            root.appendChild(bar);

            // ── Button events ──
            fitBtn.addEventListener("click", () => {
                s.fl = 0; s.ft = 0; s.fr = 1; s.fb = 1;
                syncW(); draw();
            });

            fillBtn.addEventListener("click", () => {
                const wasCrop = !s.fillMode;
                s.fillMode = !s.fillMode;
                fillBtn.textContent = s.fillMode ? "◧" : "○";
                fillBtn.style.borderColor = s.fillMode ? "#4fc3f7" : "#555";
                fillBtn.style.color = s.fillMode ? "#4fc3f7" : "#888";
                fillBtn.style.background = s.fillMode ? "#0d3b5e" : "#2a2a3e";
                fillBtn.title = s.fillMode ? "裁剪模式：开启后图像尺寸不变，裁剪区域自动填充颜色" : "裁剪模式：关闭后只裁剪不外扩不填充";
                const fmw = getW("fill_mode");
                if (fmw) { fmw.value = s.fillMode; if (fmw.callback) fmw.callback(s.fillMode); }
                if (s.fillMode) {
                    if (wasCrop) {
                        const pad = 0.05;
                        if (s.fl === 0) s.fl = -pad;
                        if (s.ft === 0) s.ft = -pad;
                        if (s.fr === 1) s.fr = 1 + pad;
                        if (s.fb === 1) s.fb = 1 + pad;
                    }
                } else {
                    s.fl = Math.max(0, s.fl); s.ft = Math.max(0, s.ft);
                    s.fr = Math.min(1, s.fr); s.fb = Math.min(1, s.fb);
                }
                syncW(); draw();
            });

            reloadBtn.addEventListener("click", () => {
                retryLoad(1, 0);
            });

            // ── Coordinate helpers ──
            function vp() {
                const vl = Math.min(0, s.fl) - 0.08;
                const vt = Math.min(0, s.ft) - 0.08;
                const vr = Math.max(1, s.fr) + 0.08;
                const vb = Math.max(1, s.fb) + 0.08;
                return { vl, vt, vr, vb, vw: vr - vl, vh: vb - vt };
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
                        ctx.fillText("拖拽蓝色边框定义外扩/裁剪区域", cw / 2, ch / 2 + 14);
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
                updateDimLabel();

                ctx.fillStyle = "#0d0d1a";
                ctx.fillRect(0, 0, cw, ch);

                const ft = n2c(s.fl, s.ft, info);
                const fb = n2c(s.fr, s.fb, info);
                const fw = fb.x - ft.x, fh = fb.y - ft.y;

                // Frame area: fill color + image
                ctx.save();
                ctx.beginPath();
                ctx.rect(ft.x, ft.y, fw, fh);
                ctx.clip();
                ctx.fillStyle = s.fill;
                ctx.fillRect(0, 0, cw, ch);
                const it = n2c(0, 0, info);
                const ib = n2c(1, 1, info);
                ctx.drawImage(s.img, it.x, it.y, ib.x - it.x, ib.y - it.y);
                ctx.restore();

                // Outside frame: fill color (when fillMode + inside image) or dim overlay
                const isFillExterior = s.fillMode && s.fl >= 0 && s.ft >= 0 && s.fr <= 1 && s.fb <= 1;
                ctx.fillStyle = isFillExterior ? s.fill : "rgba(0,0,0,0.6)";
                ctx.fillRect(0, 0, cw, ft.y);
                ctx.fillRect(0, fb.y, cw, ch - fb.y);
                ctx.fillRect(0, ft.y, ft.x, fh);
                ctx.fillRect(fb.x, ft.y, cw - fb.x, fh);

                // Image boundary (yellow dashed)
                const i0 = n2c(0, 0, info);
                const i1 = n2c(1, 1, info);
                ctx.save();
                ctx.strokeStyle = "rgba(255, 200, 50, 0.5)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(i0.x, i0.y, i1.x - i0.x, i1.y - i0.y);
                ctx.setLineDash([]);
                const cm = 6;
                ctx.strokeStyle = "rgba(255, 200, 50, 0.7)";
                ctx.lineWidth = 2;
                [[i0.x,i0.y,1,1],[i1.x,i0.y,-1,1],[i0.x,i1.y,1,-1],[i1.x,i1.y,-1,-1]].forEach(([cx,cy,dx,dy]) => {
                    ctx.beginPath();
                    ctx.moveTo(cx,cy); ctx.lineTo(cx+dx*cm,cy);
                    ctx.moveTo(cx,cy); ctx.lineTo(cx,cy+dy*cm);
                    ctx.stroke();
                });
                ctx.restore();

                // Frame outline
                ctx.strokeStyle = "#4fc3f7";
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.strokeRect(ft.x, ft.y, fw, fh);
                ctx.setLineDash([]);

                // Feather indicator: gradient ring at frame boundary
                if (s.feather > 0 && s.lastInfo) {
                    const featherNorm = s.feather / (s.img?.naturalWidth || 1);
                    const featherPixels = Math.abs(n2c(s.fl + featherNorm, 0, s.lastInfo).x - ft.x);
                    if (featherPixels > 1) {
                        const grad = ctx.createRadialGradient(
                            ft.x + fw / 2, ft.y + fh / 2, Math.max(fw, fh) / 2 - featherPixels,
                            ft.x + fw / 2, ft.y + fh / 2, Math.max(fw, fh) / 2
                        );
                        grad.addColorStop(0, "rgba(79,195,247,0)");
                        grad.addColorStop(1, "rgba(79,195,247,0.15)");
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(ft.x, ft.y, fw, fh);
                        ctx.rect(ft.x + featherPixels, ft.y + featherPixels, fw - featherPixels * 2, fh - featherPixels * 2);
                        ctx.closePath();
                        ctx.fillStyle = grad;
                        ctx.fill("evenodd");
                        ctx.restore();
                    }
                }

                // Handles
                const hs = 11;
                const hDefs = [
                    {id:"nw",x:ft.x,y:ft.y},{id:"n",x:ft.x+fw/2,y:ft.y},{id:"ne",x:fb.x,y:ft.y},
                    {id:"e",x:fb.x,y:ft.y+fh/2},{id:"se",x:fb.x,y:fb.y},{id:"s",x:ft.x+fw/2,y:fb.y},
                    {id:"sw",x:ft.x,y:fb.y},{id:"w",x:ft.x,y:ft.y+fh/2},
                ];
                s._hits = hDefs.map(h => {
                    ctx.fillStyle = "#fff"; ctx.strokeStyle = "#222"; ctx.lineWidth = 1.5;
                    ctx.beginPath(); ctx.rect(h.x-hs/2, h.y-hs/2, hs, hs); ctx.fill(); ctx.stroke();
                    return {...h, cx:h.x, cy:h.y};
                });
            }
            updateDimLabel();
            s._opDraw = draw;

            // ── Widget sync ──
            function syncW() {
                const set = (name, val) => {
                    const w = getW(name);
                    if (w) { w.value = val; if (w.callback) w.callback(val); }
                };
                set("frame_left", Math.round(s.fl * 1000) / 1000);
                set("frame_top", Math.round(s.ft * 1000) / 1000);
                set("frame_right", Math.round(s.fr * 1000) / 1000);
                set("frame_bottom", Math.round(s.fb * 1000) / 1000);
                updateDimLabel();
                app.graph.setDirtyCanvas(true, true);
            }

            // ── Interaction ──
            function getHit(mx, my) {
                if (!s._hits) return null;
                for (const h of s._hits) {
                    if (Math.abs(mx - h.cx) < 8 && Math.abs(my - h.cy) < 8) return h;
                }
                return null;
            }
            function inFrame(mx, my) {
                if (!s.lastInfo) return false;
                const p = c2n(mx, my, s.lastInfo);
                return p.nx >= s.fl && p.nx <= s.fr && p.ny >= s.ft && p.ny <= s.fb;
            }
            const CURS = { nw:"nwse-resize", n:"ns-resize", ne:"nesw-resize", e:"ew-resize", se:"nwse-resize", s:"ns-resize", sw:"nesw-resize", w:"ew-resize" };

            cv.addEventListener("mousedown", (e) => {
                const rect = cv.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;
                const hit = getHit(mx, my);
                if (hit) s.mode = hit.id;
                else if (inFrame(mx, my)) s.mode = "move";
                else return;
                s.mx0 = mx; s.my0 = my;
                s.frame0 = { fl: s.fl, ft: s.ft, fr: s.fr, fb: s.fb };
                e.preventDefault();
            });

            const onMM = (e) => {
                const rect = cv.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;
                if (!s.mode) {
                    const hit = getHit(mx, my);
                    cv.style.cursor = hit ? (CURS[hit.id]||"crosshair") : inFrame(mx,my) ? "move" : "crosshair";
                    return;
                }
                if (!s.lastInfo) return;
                const p = c2n(mx, my, s.lastInfo);
                const f0 = s.frame0, MIN = 0.005;
                switch (s.mode) {
                    case "move": {
                        const dn = c2n(s.mx0, s.my0, s.lastInfo);
                        s.fl = f0.fl + p.nx - dn.nx; s.ft = f0.ft + p.ny - dn.ny;
                        s.fr = f0.fr + p.nx - dn.nx; s.fb = f0.fb + p.ny - dn.ny;
                        break;
                    }
                    case "n":  s.ft = Math.min(p.ny, f0.fb - MIN); break;
                    case "s":  s.fb = Math.max(p.ny, f0.ft + MIN); break;
                    case "w":  s.fl = Math.min(p.nx, f0.fr - MIN); break;
                    case "e":  s.fr = Math.max(p.nx, f0.fl + MIN); break;
                    case "nw": s.fl = Math.min(p.nx, f0.fr - MIN); s.ft = Math.min(p.ny, f0.fb - MIN); break;
                    case "ne": s.fr = Math.max(p.nx, f0.fl + MIN); s.ft = Math.min(p.ny, f0.fb - MIN); break;
                    case "sw": s.fl = Math.min(p.nx, f0.fr - MIN); s.fb = Math.max(p.ny, f0.ft + MIN); break;
                    case "se": s.fr = Math.max(p.nx, f0.fl + MIN); s.fb = Math.max(p.ny, f0.ft + MIN); break;
                }
                // Crop mode: clamp to image
                if (!s.fillMode) {
                    s.fl = Math.max(0, Math.min(1, s.fl));
                    s.ft = Math.max(0, Math.min(1, s.ft));
                    s.fr = Math.max(0, Math.min(1, s.fr));
                    s.fb = Math.max(0, Math.min(1, s.fb));
                }
                syncW(); draw(); e.preventDefault();
            };
            const onMU = () => { if (s.mode) { s.mode = null; syncW(); } };
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
                let url = null, viaFetch = false;
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

            function retryLoad(maxTries = 15, delay = 300) {
                let tries = 0;
                const at = () => {
                    tries++;
                    if (tryLoadFromSource()) return;
                    if (tries >= maxTries) { s.loadError = "重试耗尽"; draw(); return; }
                    setTimeout(at, delay);
                };
                at();
            }
            s._loadRetry = retryLoad;

            // ── Configure (workflow restore) ──
            const origCfg = this.configure;
            this.configure = function (data) {
                if (origCfg) origCfg.apply(this, arguments);
                const st = this._opState;
                if (!st) return;
                st.fl = getW("frame_left")?.value ?? st.fl;
                st.ft = getW("frame_top")?.value ?? st.ft;
                st.fr = getW("frame_right")?.value ?? st.fr;
                st.fb = getW("frame_bottom")?.value ?? st.fb;
                st.fill = getW("填充颜色")?.value ?? st.fill;
                st.fillMode = getW("fill_mode")?.value ?? st.fillMode;
                st.feather = getW("feather")?.value ?? st.feather;
                if (!st.fillMode) {
                    st.fl = Math.max(0, Math.min(1, st.fl));
                    st.ft = Math.max(0, Math.min(1, st.ft));
                    st.fr = Math.max(0, Math.min(1, st.fr));
                    st.fb = Math.max(0, Math.min(1, st.fb));
                }
                fillBtn.textContent = st.fillMode ? "◧" : "○";
                fillBtn.style.borderColor = st.fillMode ? "#4fc3f7" : "#555";
                fillBtn.style.color = st.fillMode ? "#4fc3f7" : "#888";
                fillBtn.style.background = st.fillMode ? "#0d3b5e" : "#2a2a3e";
                fillBtn.title = st.fillMode ? "裁剪模式：开启后图像尺寸不变，裁剪区域自动填充颜色" : "裁剪模式：关闭后只裁剪不外扩不填充";
                swatch.style.backgroundColor = st.fill;
                hexLbl.textContent = st.fill.toUpperCase();
                // Restore feather
                if (getW("feather")) {
                    st.feather = getW("feather").value;
                    featherSlider.value = st.feather;
                    featherVal.textContent = `${st.feather}px`;
                }
                setTimeout(() => { setWrapHeight(); retryLoad(20, 300); }, 500);
            };

            // ── On executed ──
            const origExec = this.onExecuted;
            this.onExecuted = function (msg) {
                if (origExec) origExec.apply(this, arguments);
                const st = this._opState;
                if (!st) return;
                setTimeout(() => st._loadRetry(10, 300), 200);
            };

            // ── Canvas sizing ──
            function setWrapHeight() {
                const rect = wrap.getBoundingClientRect();
                if (rect.width > 20) wrap.style.height = Math.max(180, Math.floor(rect.width * 0.72)) + "px";
                draw();
            }

            // ── DOM widget ──
            const widget = this.addDOMWidget("outpaint_preview", "OUTPAINT_PREVIEW", root, {
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
                this._opState = null;
                if (origRM) origRM.apply(this, arguments);
            };

            this.setSize([350, 260]);
            setTimeout(() => retryLoad(12, 400), 300);
            return rv;
        };
    },
});
