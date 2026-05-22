import { app } from "/scripts/app.js";

app.registerExtension({
    name: "zoey.textOverlay",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ZoeyTextOverlay") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const rv = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            const getW = (name) => node.widgets?.find(w => w.name === name);

            // ── Hide raw widgets ──
            ["text_config", "font_path"].forEach(name => {
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
                text: getW("text")?.value ?? "Hello",
                x: 0.5, y: 0.5, size: 48, rotation: 0, opacity: 1,
                align: "center", color: "#ffffff", fontPath: "",
                mode: null, mx0: 0, my0: 0, pos0: null, lastInfo: null,
            };
            node._toState = s;

            // ── DOM ──
            const root = document.createElement("div");
            root.style.cssText = "width:100%;display:flex;flex-direction:column;gap:4px;padding:2px 0;user-select:none;";

            const wrap = document.createElement("div");
            wrap.style.cssText = "width:100%;position:relative;background:#1a1a2e;border-radius:6px;overflow:hidden;border:1px solid #333;min-height:180px;height:220px;";

            const cv = document.createElement("canvas");
            cv.style.cssText = "width:100%;height:100%;display:block;cursor:default;";
            wrap.appendChild(cv);
            root.appendChild(wrap);

            // ── Controls ──
            const bar = document.createElement("div");
            bar.style.cssText = "display:flex;align-items:center;gap:6px;padding:2px 0;flex-wrap:wrap;";

            // Text input
            const textInput = document.createElement("input");
            textInput.type = "text";
            textInput.value = s.text;
            textInput.style.cssText = "flex:1;min-width:60px;padding:2px 4px;font-size:11px;background:#1a1a2e;border:1px solid #444;border-radius:4px;color:#ddd;height:22px;";
            textInput.placeholder = "输入文字...";

            // Size
            const sizeLbl = document.createElement("span");
            sizeLbl.textContent = "字号";
            sizeLbl.style.cssText = "font-size:9px;color:#888;flex:none;";

            const sizeSlider = document.createElement("input");
            sizeSlider.type = "range";
            sizeSlider.min = 8; sizeSlider.max = 200; sizeSlider.value = s.size;
            sizeSlider.style.cssText = "width:56px;height:14px;cursor:pointer;flex:none;";

            const sizeVal = document.createElement("span");
            sizeVal.textContent = `${s.size}`;
            sizeVal.style.cssText = "font-size:10px;font-family:monospace;color:#aaa;width:28px;text-align:right;flex:none;";

            // Color
            const swatch = document.createElement("div");
            swatch.style.cssText = "width:22px;height:22px;border-radius:4px;border:2px solid #555;cursor:pointer;flex-shrink:0;";
            swatch.style.backgroundColor = s.color;

            const ci = document.createElement("input");
            ci.type = "color";
            ci.value = s.color;
            ci.style.cssText = "width:0;height:0;padding:0;border:none;position:absolute;opacity:0;pointer-events:none;";

            swatch.addEventListener("click", () => ci.click());
            ci.addEventListener("input", () => {
                s.color = ci.value;
                swatch.style.backgroundColor = s.color;
                syncW(); draw();
            });

            // Opacity
            const opLbl = document.createElement("span");
            opLbl.textContent = "不透明";
            opLbl.style.cssText = "font-size:9px;color:#888;flex:none;";

            const opSlider = document.createElement("input");
            opSlider.type = "range";
            opSlider.min = 0; opSlider.max = 100; opSlider.value = Math.round(s.opacity * 100);
            opSlider.style.cssText = "width:40px;height:14px;cursor:pointer;flex:none;";

            const opVal = document.createElement("span");
            opVal.textContent = `${Math.round(s.opacity * 100)}%`;
            opVal.style.cssText = "font-size:9px;font-family:monospace;color:#888;width:26px;text-align:right;flex:none;";

            // Rotation label
            const rotLbl = document.createElement("span");
            rotLbl.textContent = "旋转";
            rotLbl.style.cssText = "font-size:9px;color:#888;flex:none;";

            const rotVal = document.createElement("span");
            rotVal.textContent = `${Math.round(s.rotation)}°`;
            rotVal.style.cssText = "font-size:9px;font-family:monospace;color:#888;width:24px;text-align:right;flex:none;";

            bar.appendChild(textInput);
            bar.appendChild(sizeLbl);
            bar.appendChild(sizeSlider);
            bar.appendChild(sizeVal);
            bar.appendChild(swatch);
            bar.appendChild(ci);
            bar.appendChild(opLbl);
            bar.appendChild(opSlider);
            bar.appendChild(opVal);
            bar.appendChild(rotLbl);
            bar.appendChild(rotVal);
            root.appendChild(bar);

            // ── Events ──
            textInput.addEventListener("input", () => {
                s.text = textInput.value || " ";
                const tw = getW("text");
                if (tw) { tw.value = s.text; if (tw.callback) tw.callback(s.text); }
                draw();
            });

            sizeSlider.addEventListener("input", () => {
                s.size = parseInt(sizeSlider.value);
                sizeVal.textContent = `${s.size}`;
                syncW(); draw();
            });

            opSlider.addEventListener("input", () => {
                s.opacity = opSlider.value / 100;
                opVal.textContent = `${opSlider.value}%`;
                syncW(); draw();
            });

            // ── Coordinate helpers ──
            function vp() {
                return { vl: 0, vt: 0, vr: 1, vb: 1, vw: 1, vh: 1 };
            }

            function n2c(nx, ny, info) {
                return { x: info.ox + (nx - info.vl) * info.scX, y: info.oy + (ny - info.vt) * info.scY };
            }
            function c2n(cx, cy, info) {
                return { nx: (cx - info.ox) / info.scX + info.vl, ny: (cy - info.oy) / info.scY + info.vt };
            }

            // ── Drawing ──
            let _textCache = { text: "", size: 0, width: 0, height: 0 };
            function measureText(ctx, text, size) {
                if (_textCache.text === text && _textCache.size === size) return _textCache;
                const font = `${size}px sans-serif`;
                ctx.font = font;
                const m = ctx.measureText(text);
                const h = size * 1.2;
                _textCache = { text, size, width: m.width, height: h };
                return _textCache;
            }

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
                        ctx.fillText("点击 ⟳ 重试", cw / 2, ch / 2 + 14);
                    } else {
                        ctx.fillText("连接图像源后实时预览", cw / 2, ch / 2 - 8);
                        ctx.font = "11px sans-serif"; ctx.fillStyle = "#444";
                        ctx.fillText("拖拽文字调整位置", cw / 2, ch / 2 + 14);
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

                // Image
                const it = n2c(0, 0, info);
                const ib = n2c(1, 1, info);
                ctx.drawImage(s.img, it.x, it.y, ib.x - it.x, ib.y - it.y);

                // Text overlay
                const ts = Math.max(8, s.size * (ib.x - it.x) / s.img.naturalWidth);
                const font = `${ts}px sans-serif`;
                ctx.font = font;
                const tm = ctx.measureText(s.text);
                const tw = tm.width;
                const th = ts * 1.2;

                const tc = n2c(s.x, s.y, info);
                const tx = tc.x - tw / 2;
                const ty = tc.y - th / 2;

                ctx.save();
                ctx.translate(tc.x, tc.y);
                ctx.rotate(s.rotation * Math.PI / 180);

                // Text shadow
                ctx.font = font;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.shadowColor = "rgba(0,0,0,0.7)";
                ctx.shadowBlur = 4;
                ctx.fillStyle = s.color;
                ctx.globalAlpha = s.opacity;
                ctx.fillText(s.text, 0, 0);
                ctx.shadowBlur = 0;
                ctx.globalAlpha = 1;

                // Bounding box (selected state indicator)
                ctx.strokeStyle = "rgba(79,195,247,0.5)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.strokeRect(-tw / 2 - 4, -th / 2 - 2, tw + 8, th + 4);
                ctx.setLineDash([]);

                // Rotation handle
                const handleY = -th / 2 - 22;
                ctx.beginPath();
                ctx.moveTo(0, -th / 2 - 4);
                ctx.lineTo(0, handleY);
                ctx.strokeStyle = "rgba(255,255,255,0.6)";
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.fillStyle = "#4fc3f7";
                ctx.beginPath();
                ctx.arc(0, handleY, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1;
                ctx.stroke();

                s._rotHandle = {
                    x: tc.x, y: tc.y - (th / 2 + 22) * (s.rotation ? 1 : 1),
                    angle: s.rotation,
                };
                // Better: compute rotated handle world position
                const rad = s.rotation * Math.PI / 180;
                const cosR = Math.cos(rad), sinR = Math.sin(rad);
                const handleLocalX = 0;
                const handleLocalY = -th / 2 - 22;
                const handleWX = tc.x + handleLocalX * cosR - handleLocalY * sinR;
                const handleWY = tc.y + handleLocalX * sinR + handleLocalY * cosR;
                s._rotHandle = { x: handleWX, y: handleWY };

                ctx.restore();

                // Corner resize handles (bottom-right corner)
                const hs = 8;
                const corners = [
                    { dx: -tw/2 - 4, dy: -th/2 - 2, id: "tl" },
                    { dx: tw/2 + 4, dy: -th/2 - 2, id: "tr" },
                    { dx: tw/2 + 4, dy: th/2 + 2, id: "br" },
                    { dx: -tw/2 - 4, dy: th/2 + 2, id: "bl" },
                ];
                s._resizeHandles = [];
                corners.forEach(c => {
                    const wx = tc.x + c.dx * cosR - c.dy * sinR;
                    const wy = tc.y + c.dx * sinR + c.dy * cosR;
                    s._resizeHandles.push({ x: wx, y: wy, id: c.id });
                    ctx.save();
                    ctx.translate(wx, wy);
                    ctx.rotate(s.rotation * Math.PI / 180); // keep handles axis-aligned in text space
                    ctx.fillStyle = c.id === "br" ? "#4fc3f7" : "rgba(255,255,255,0.6)";
                    ctx.strokeStyle = "#222";
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.rect(-hs/2, -hs/2, hs, hs);
                    ctx.fill();
                    ctx.stroke();
                    ctx.restore();
                });

                // Info
                ctx.fillStyle = "rgba(255,255,255,0.4)";
                ctx.font = "10px sans-serif";
                ctx.textAlign = "left"; ctx.textBaseline = "top";
                ctx.fillText(`${s.img.naturalWidth}×${s.img.naturalHeight}`, 6, 4);

                // Rotation angle badge
                if (s.rotation !== 0) {
                    ctx.fillStyle = "rgba(255,255,255,0.5)";
                    ctx.font = "9px monospace";
                    ctx.textAlign = "right"; ctx.textBaseline = "bottom";
                    ctx.fillText(`${Math.round(s.rotation)}°`, cw - 4, ch - 4);
                }
            }

            // ── Widget sync ──
            function syncW() {
                const cfg = {
                    x: s.x, y: s.y, size: s.size,
                    r: s.rotation, o: s.opacity,
                    color: s.color, align: s.align,
                };
                const w = getW("text_config");
                if (w) {
                    w.value = JSON.stringify(cfg);
                    if (w.callback) w.callback(w.value);
                }
                app.graph.setDirtyCanvas(true, true);
            }

            // ── Hit testing ──
            function hitText(mx, my) {
                if (!s.lastInfo) return false;
                const ts = Math.max(8, s.size * (n2c(1, 1, s.lastInfo).x - n2c(0, 0, s.lastInfo).x) / s.img.naturalWidth);
                const ctx = cv.getContext("2d");
                ctx.font = `${ts}px sans-serif`;
                const tm = ctx.measureText(s.text);
                const tw = tm.width, th = ts * 1.2;
                const tc = n2c(s.x, s.y, s.lastInfo);
                const rad = s.rotation * Math.PI / 180;
                const dx = mx - tc.x;
                const dy = my - tc.y;
                const lx = dx * Math.cos(-rad) - dy * Math.sin(-rad);
                const ly = dx * Math.sin(-rad) + dy * Math.cos(-rad);
                return Math.abs(lx) <= tw / 2 + 6 && Math.abs(ly) <= th / 2 + 4;
            }

            function hitRotHandle(mx, my) {
                if (!s._rotHandle) return false;
                const dx = mx - s._rotHandle.x;
                const dy = my - s._rotHandle.y;
                return Math.sqrt(dx * dx + dy * dy) < 12;
            }

            function hitResizeHandle(mx, my) {
                if (!s._resizeHandles) return -1;
                for (let i = 0; i < s._resizeHandles.length; i++) {
                    const dx = mx - s._resizeHandles[i].x;
                    const dy = my - s._resizeHandles[i].y;
                    if (Math.sqrt(dx * dx + dy * dy) < 10) return i;
                }
                return -1;
            }

            // Get text bounding box half-dimensions in canvas pixels (at current state)
            function getTextHalfDims() {
                if (!s.lastInfo || !s.img) return { hw: 1, hh: 1 };
                const it = n2c(0, 0, s.lastInfo);
                const ib = n2c(1, 1, s.lastInfo);
                const ts = Math.max(8, s.size * (ib.x - it.x) / s.img.naturalWidth);
                const ctx = cv.getContext("2d");
                ctx.font = `${ts}px sans-serif`;
                const tm = ctx.measureText(s.text);
                return { hw: tm.width / 2 + 4, hh: ts * 1.2 / 2 + 2 };
            }

            // ── Interaction ──
            cv.addEventListener("mousedown", (e) => {
                const rect = cv.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;

                const rIdx = hitResizeHandle(mx, my);
                if (rIdx >= 0) {
                    s.mode = "resize";
                    s.resizeCorner = s._resizeHandles[rIdx].id;
                    s.mx0 = mx; s.my0 = my;
                    s.pos0 = { size: s.size, x: s.x, y: s.y, rotation: s.rotation };
                    const hd = getTextHalfDims();
                    s._resizeHalfDims = hd;
                    e.preventDefault();
                    return;
                }

                if (hitRotHandle(mx, my)) {
                    s.mode = "rotate";
                    s.mx0 = mx; s.my0 = my;
                    s.pos0 = { rotation: s.rotation, x: s.x, y: s.y };
                    e.preventDefault();
                    return;
                }

                if (hitText(mx, my)) {
                    s.mode = "move";
                    s.mx0 = mx; s.my0 = my;
                    s.pos0 = { x: s.x, y: s.y };
                    cv.style.cursor = "grabbing";
                    e.preventDefault();
                }
            });

            const onMM = (e) => {
                const rect = cv.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;

                if (!s.mode) {
                    const onRS = hitResizeHandle(mx, my) >= 0;
                    cv.style.cursor = onRS ? "nwse-resize" : hitText(mx, my) ? "grab" : hitRotHandle(mx, my) ? "crosshair" : "default";
                    return;
                }
                if (!s.lastInfo) return;

                if (s.mode === "rotate") {
                    const tc = n2c(s.pos0.x, s.pos0.y, s.lastInfo);
                    const curAngle = Math.atan2(my - tc.y, mx - tc.x);
                    const startAngle = Math.atan2(s.my0 - tc.y, s.mx0 - tc.x);
                    let deltaDeg = (curAngle - startAngle) * 180 / Math.PI;
                    if (e.shiftKey) {
                        const raw = (s.pos0?.rotation ?? 0) + deltaDeg;
                        s.rotation = ((Math.round(raw / 15) * 15) % 360 + 360) % 360;
                    } else {
                        s.rotation = (((s.pos0?.rotation ?? 0) + deltaDeg) % 360 + 360) % 360;
                    }
                    rotVal.textContent = `${Math.round(s.rotation)}°`;
                    syncW(); draw(); e.preventDefault();
                    return;
                }

                if (s.mode === "move") {
                    const p = c2n(mx, my, s.lastInfo);
                    const p0 = c2n(s.mx0, s.my0, s.lastInfo);
                    s.x = (s.pos0?.x ?? 0.5) + p.nx - p0.nx;
                    s.y = (s.pos0?.y ?? 0.5) + p.ny - p0.ny;
                    s.x = Math.max(0.05, Math.min(0.95, s.x));
                    s.y = Math.max(0.05, Math.min(0.95, s.y));
                    syncW(); draw(); e.preventDefault();
                }

                if (s.mode === "resize") {
                    if (!s.lastInfo || !s.img) return;
                    const tc = n2c(s.pos0.x, s.pos0.y, s.lastInfo);
                    const rad = (s.pos0.rotation ?? s.rotation) * Math.PI / 180;
                    const cosR = Math.cos(rad), sinR = Math.sin(rad);
                    // Mouse position in text-local coords
                    const dx = mx - tc.x;
                    const dy = my - tc.y;
                    const lx = dx * cosR + dy * sinR;
                    const ly = -dx * sinR + dy * cosR;
                    // Use the larger axis ratio
                    const hw = s._resizeHalfDims?.hw ?? 1;
                    const hh = s._resizeHalfDims?.hh ?? 1;
                    const scaleX = Math.abs(lx) / hw;
                    const scaleY = Math.abs(ly) / hh;
                    const scale = Math.max(0.3, Math.min(10, (scaleX + scaleY) / 2));
                    const newSize = Math.round(Math.max(8, s.pos0.size * scale));
                    if (newSize !== s.size) {
                        s.size = newSize;
                        sizeSlider.value = s.size;
                        sizeVal.textContent = `${s.size}`;
                        syncW(); draw();
                    }
                    e.preventDefault();
                }
            };

            const onMU = () => {
                if (s.mode) {
                    s.mode = null;
                    cv.style.cursor = "default";
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
                const st = this._toState;
                if (!st) return;
                st.text = getW("text")?.value ?? "Hello";
                textInput.value = st.text;
                try {
                    const raw = getW("text_config")?.value || "{}";
                    const cfg = JSON.parse(raw);
                    st.x = cfg.x ?? 0.5;
                    st.y = cfg.y ?? 0.5;
                    st.size = cfg.size ?? 48;
                    st.rotation = (cfg.r ?? 0) % 360;
                    st.opacity = cfg.o ?? 1;
                    st.color = cfg.color ?? "#ffffff";
                    st.align = cfg.align ?? "center";
                } catch (e) {}
                sizeSlider.value = st.size;
                sizeVal.textContent = `${st.size}`;
                opSlider.value = Math.round(st.opacity * 100);
                opVal.textContent = `${Math.round(st.opacity * 100)}%`;
                rotVal.textContent = `${Math.round(st.rotation)}°`;
                swatch.style.backgroundColor = st.color;
                ci.value = st.color;
                setTimeout(() => { retryLoad(20, 300); }, 500);
            };

            // ── On executed ──
            const origExec = this.onExecuted;
            this.onExecuted = function (msg) {
                if (origExec) origExec.apply(this, arguments);
                const st = this._toState;
                if (!st) return;
                setTimeout(() => retryLoad(10, 300), 200);
            };

            // ── DOM widget ──
            const widget = this.addDOMWidget("to_preview", "TO_PREVIEW", root, {
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
                this._toState = null;
                if (origRM) origRM.apply(this, arguments);
            };

            this.setSize([350, 260]);
            setTimeout(() => retryLoad(12, 400), 300);
            return rv;
        };
    },
});
