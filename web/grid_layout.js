import { app } from "/scripts/app.js";

app.registerExtension({
    name: "zoey.gridLayout",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ZoeyGridLayout") return;

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot, isConnected, link, outputSlot) {
            if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
            const st = this._glState;
            if (!st) return;
            if (slotType === 1) {
                const inp = this.inputs[slot];
                if (!inp || !inp.name?.startsWith("image")) return;
                if (isConnected) {
                    setTimeout(() => loadOneImage(st, inp.name), 100);
                } else {
                    const idx = parseInt(inp.name.replace("image", "")) - 1;
                    if (st.images[idx]) { st.images[idx].img = null; st.images[idx].loaded = false; }
                    draw(st);
                }
            }
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const rv = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            const getW = (name) => node.widgets?.find(w => w.name === name);

            // ── Hide raw widgets ──
            const hideW = (name) => {
                const w = getW(name);
                if (w) {
                    w.computeSize = () => [0, 0];
                    w.draw = () => {};
                    w.mouse = () => false;
                    setTimeout(() => { if (w.element) w.element.style.display = "none"; }, 100);
                }
            };
            hideW("grid_config");
            ["image1","image2","image3","image4","image5","image6","image7","image8","image9"].forEach(hideW);

            // ── State ──
            const s = {
                node,
                images: Array.from({length: 9}, () => ({ img: null, loaded: false, loadError: null })),
                cols: 3, gap: 4, border: 0, radius: 0,
                bg: "#222222", borderColor: "#444444",
                lastInfo: null,
            };
            node._glState = s;

            // ── DOM ──
            const root = document.createElement("div");
            root.style.cssText = "width:100%;display:flex;flex-direction:column;gap:4px;padding:2px 0;user-select:none;";

            const wrap = document.createElement("div");
            wrap.style.cssText = "width:100%;position:relative;background:#1a1a2e;border-radius:6px;overflow:hidden;border:1px solid #333;min-height:180px;height:220px;";

            const cv = document.createElement("canvas");
            cv.style.cssText = "width:100%;height:100%;display:block;cursor:default;";
            wrap.appendChild(cv);
            root.appendChild(wrap);

            // ── Controls bar ──
            const bar = document.createElement("div");
            bar.style.cssText = "display:flex;align-items:center;gap:6px;padding:2px 0;flex-wrap:wrap;";

            // Reload
            const reloadBtn = document.createElement("button");
            reloadBtn.textContent = "⟳";
            reloadBtn.title = "重新加载所有图像";
            reloadBtn.style.cssText = "font-size:14px;padding:0 8px;border:1px solid #555;border-radius:4px;background:#2a2a3e;color:#ccc;cursor:pointer;height:26px;line-height:26px;";

            // Columns
            const colsLbl = document.createElement("span");
            colsLbl.textContent = "列数";
            colsLbl.style.cssText = "font-size:9px;color:#888;flex:none;";

            const colsSlider = document.createElement("input");
            colsSlider.type = "range";
            colsSlider.min = 1; colsSlider.max = 9; colsSlider.value = s.cols;
            colsSlider.style.cssText = "width:40px;height:14px;cursor:pointer;flex:none;";

            const colsVal = document.createElement("span");
            colsVal.textContent = `${s.cols}`;
            colsVal.style.cssText = "font-size:10px;font-family:monospace;color:#aaa;width:14px;text-align:right;flex:none;";

            // Gap
            const gapLbl = document.createElement("span");
            gapLbl.textContent = "间距";
            gapLbl.style.cssText = "font-size:9px;color:#888;flex:none;";

            const gapSlider = document.createElement("input");
            gapSlider.type = "range";
            gapSlider.min = 0; gapSlider.max = 40; gapSlider.value = s.gap;
            gapSlider.style.cssText = "width:40px;height:14px;cursor:pointer;flex:none;";

            const gapVal = document.createElement("span");
            gapVal.textContent = `${s.gap}px`;
            gapVal.style.cssText = "font-size:10px;font-family:monospace;color:#aaa;width:28px;text-align:right;flex:none;";

            // Border
            const bdrLbl = document.createElement("span");
            bdrLbl.textContent = "边框";
            bdrLbl.style.cssText = "font-size:9px;color:#888;flex:none;";

            const bdrSlider = document.createElement("input");
            bdrSlider.type = "range";
            bdrSlider.min = 0; bdrSlider.max = 20; bdrSlider.value = s.border;
            bdrSlider.style.cssText = "width:40px;height:14px;cursor:pointer;flex:none;";

            const bdrVal = document.createElement("span");
            bdrVal.textContent = `${s.border}px`;
            bdrVal.style.cssText = "font-size:10px;font-family:monospace;color:#aaa;width:24px;text-align:right;flex:none;";

            // Radius
            const radLbl = document.createElement("span");
            radLbl.textContent = "圆角";
            radLbl.style.cssText = "font-size:9px;color:#888;flex:none;";

            const radSlider = document.createElement("input");
            radSlider.type = "range";
            radSlider.min = 0; radSlider.max = 60; radSlider.value = s.radius;
            radSlider.style.cssText = "width:40px;height:14px;cursor:pointer;flex:none;";

            const radVal = document.createElement("span");
            radVal.textContent = `${s.radius}px`;
            radVal.style.cssText = "font-size:10px;font-family:monospace;color:#aaa;width:24px;text-align:right;flex:none;";

            // BG color
            const bgSwatch = document.createElement("div");
            bgSwatch.style.cssText = "width:18px;height:18px;border-radius:3px;border:2px solid #555;cursor:pointer;flex-shrink:0;";
            bgSwatch.style.backgroundColor = s.bg;
            bgSwatch.title = "背景色";

            const bgCi = document.createElement("input");
            bgCi.type = "color";
            bgCi.value = s.bg;
            bgCi.style.cssText = "width:0;height:0;padding:0;border:none;position:absolute;opacity:0;pointer-events:none;";

            bgSwatch.addEventListener("click", () => bgCi.click());
            bgCi.addEventListener("input", () => {
                s.bg = bgCi.value;
                bgSwatch.style.backgroundColor = s.bg;
                syncW(); draw();
            });

            // Border color
            const bdrSwatch = document.createElement("div");
            bdrSwatch.style.cssText = "width:18px;height:18px;border-radius:3px;border:2px solid #555;cursor:pointer;flex-shrink:0;margin-left:4px;";
            bdrSwatch.style.backgroundColor = s.borderColor;
            bdrSwatch.title = "边框色";

            const bdrCi = document.createElement("input");
            bdrCi.type = "color";
            bdrCi.value = s.borderColor;
            bdrCi.style.cssText = "width:0;height:0;padding:0;border:none;position:absolute;opacity:0;pointer-events:none;";

            bdrSwatch.addEventListener("click", () => bdrCi.click());
            bdrCi.addEventListener("input", () => {
                s.borderColor = bdrCi.value;
                bdrSwatch.style.backgroundColor = s.borderColor;
                syncW(); draw();
            });

            bar.appendChild(reloadBtn);
            bar.appendChild(colsLbl);
            bar.appendChild(colsSlider);
            bar.appendChild(colsVal);
            bar.appendChild(gapLbl);
            bar.appendChild(gapSlider);
            bar.appendChild(gapVal);
            bar.appendChild(bdrLbl);
            bar.appendChild(bdrSlider);
            bar.appendChild(bdrVal);
            bar.appendChild(radLbl);
            bar.appendChild(radSlider);
            bar.appendChild(radVal);
            bar.appendChild(bgSwatch);
            bar.appendChild(bgCi);
            bar.appendChild(bdrSwatch);
            bar.appendChild(bdrCi);
            root.appendChild(bar);

            // ── Events ──
            reloadBtn.addEventListener("click", () => {
                for (let i = 0; i < 9; i++) loadOneImage(s, `image${i+1}`);
            });

            colsSlider.addEventListener("input", () => {
                s.cols = parseInt(colsSlider.value);
                colsVal.textContent = `${s.cols}`;
                syncW(); draw();
            });

            gapSlider.addEventListener("input", () => {
                s.gap = parseInt(gapSlider.value);
                gapVal.textContent = `${s.gap}px`;
                syncW(); draw();
            });

            bdrSlider.addEventListener("input", () => {
                s.border = parseInt(bdrSlider.value);
                bdrVal.textContent = `${s.border}px`;
                syncW(); draw();
            });

            radSlider.addEventListener("input", () => {
                s.radius = parseInt(radSlider.value);
                radVal.textContent = `${s.radius}px`;
                syncW(); draw();
            });

            // ── Coordinate helpers ──
            function vp() {
                return { vl: 0, vt: 0, vr: 1, vb: 1, vw: 1, vh: 1 };
            }
            function n2c(nx, ny, info) {
                return { x: info.ox + (nx - info.vl) * info.scX, y: info.oy + (ny - info.vt) * info.scY };
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

                const loadedImgs = s.images.filter(l => l.loaded && l.img);
                if (loadedImgs.length === 0) {
                    ctx.fillStyle = "#555";
                    ctx.font = "13px sans-serif";
                    ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    ctx.fillText("连接图像源后实时预览", cw / 2, ch / 2 - 8);
                    ctx.font = "11px sans-serif"; ctx.fillStyle = "#444";
                    ctx.fillText("支持 1~9 张图像网格排列", cw / 2, ch / 2 + 14);
                    s.lastInfo = null;
                    return;
                }

                // Use the first loaded image's aspect to determine canvas layout
                const firstLoaded = loadedImgs[0];
                const imgAspect = (firstLoaded.img.naturalWidth / firstLoaded.img.naturalHeight) || 1;
                let scX, scY;
                const vw = 1, vh = 1;
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
                const info = { scX, scY, ox, oy };
                s.lastInfo = info;

                // Background
                ctx.fillStyle = s.bg;
                ctx.fillRect(0, 0, cw, ch);

                const cols = Math.min(s.cols, loadedImgs.length);
                const rows = Math.ceil(loadedImgs.length / cols);
                const actualCols = cols;

                // Compute cell size in normalized coords, then pixel
                const gapNorm = s.gap / Math.max(cw, ch);
                const borderNorm = s.border / Math.max(cw, ch);

                // We'll work in pixel space on the canvas
                const padPx = 10;
                const usableW = cw - padPx * 2;
                const usableH = ch - padPx * 2;
                const totalGapX = (actualCols - 1) * s.gap;
                const totalGapY = (rows - 1) * s.gap;
                const totalBorderX = borderNorm * 2 * actualCols * scX;
                const totalBorderY = borderNorm * 2 * rows * scY;
                // const totalBorderX = 0; // border rendered differently

                const cellWPx = (usableW - totalGapX) / actualCols;
                const cellHPx = (usableH - totalGapY) / rows;
                const cellSize = Math.min(cellWPx, cellHPx);

                const gridW = actualCols * cellSize + totalGapX;
                const gridH = rows * cellSize + totalGapY;
                const gridOX = (cw - gridW) / 2;
                const gridOY = (ch - gridH) / 2;

                // Draw cells
                for (let i = 0; i < loadedImgs.length; i++) {
                    const col = i % actualCols;
                    const row = Math.floor(i / actualCols);

                    const cx = gridOX + col * (cellSize + s.gap);
                    const cy = gridOY + row * (cellSize + s.gap);

                    const img = loadedImgs[i].img;
                    const iw = img.naturalWidth;
                    const ih = img.naturalHeight;

                    // Scale image to fit cell preserving aspect
                    const imgScale = Math.min(cellSize / iw, cellSize / ih);
                    const drawW = iw * imgScale;
                    const drawH = ih * imgScale;
                    const dx = cx + (cellSize - drawW) / 2;
                    const dy = cy + (cellSize - drawH) / 2;

                    // Border background
                    if (s.border > 0) {
                        ctx.fillStyle = s.borderColor;
                        ctx.fillRect(cx - s.border, cy - s.border, cellSize + s.border * 2, cellSize + s.border * 2);
                    }

                    // Rounded rect clip
                    if (s.radius > 0) {
                        ctx.save();
                        const r = Math.min(s.radius, cellSize / 2);
                        ctx.beginPath();
                        ctx.moveTo(dx + r, dy);
                        ctx.lineTo(dx + drawW - r, dy);
                        ctx.quadraticCurveTo(dx + drawW, dy, dx + drawW, dy + r);
                        ctx.lineTo(dx + drawW, dy + drawH - r);
                        ctx.quadraticCurveTo(dx + drawW, dy + drawH, dx + drawW - r, dy + drawH);
                        ctx.lineTo(dx + r, dy + drawH);
                        ctx.quadraticCurveTo(dx, dy + drawH, dx, dy + drawH - r);
                        ctx.lineTo(dx, dy + r);
                        ctx.quadraticCurveTo(dx, dy, dx + r, dy);
                        ctx.closePath();
                        ctx.clip();
                        ctx.drawImage(img, dx, dy, drawW, drawH);
                        ctx.restore();
                    } else {
                        ctx.drawImage(img, dx, dy, drawW, drawH);
                    }
                }

                // Info
                ctx.fillStyle = "rgba(255,255,255,0.4)";
                ctx.font = "10px sans-serif";
                ctx.textAlign = "left"; ctx.textBaseline = "top";
                ctx.fillText(`${loadedImgs.length} 张 · ${cols}×${rows}`, 6, 4);
            }

            // ── Widget sync ──
            function syncW() {
                const cfg = {
                    cols: s.cols, gap: s.gap, border: s.border,
                    radius: s.radius, bg: s.bg, border_color: s.borderColor,
                };
                const w = getW("grid_config");
                if (w) {
                    w.value = JSON.stringify(cfg);
                    if (w.callback) w.callback(w.value);
                }
                app.graph.setDirtyCanvas(true, true);
            }

            // ── Image loading ──
            function loadImageForSlot(st, inputName, imgObj) {
                const src = findSourceNode(st, inputName);
                if (!src) { imgObj.loaded = false; imgObj.img = null; draw(st); return; }

                let url = null;
                if (src.imgs?.length > 0) {
                    const el = src.imgs[0];
                    if (typeof el === "string") url = el;
                    else if (el?.src) url = el.src;
                    else if (el?._src) url = el._src;
                    else if (el?.tagName === "CANVAS") url = el.toDataURL?.("image/png");
                }
                if (!url && src.image) {
                    url = typeof src.image === "string" ? src.image : src.image?.src || src.image?._src;
                }
                if (!url) {
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
                        }
                    }
                }
                if (!url) {
                    for (const w of (src.widgets || [])) {
                        let v = w.value;
                        if (typeof v === "object" && v) v = v.filename || v.name || "";
                        if (typeof v === "string" && /\.(png|jpg|jpeg|webp|bmp)$/i.test(v)) {
                            const parts = v.split("/"); const fn = parts.pop();
                            const sub = parts.join("/");
                            url = `${location.origin}/view?filename=${encodeURIComponent(fn)}&type=input${sub ? "&subfolder="+encodeURIComponent(sub) : ""}&rand=${Date.now()}`;
                            break;
                        }
                    }
                }
                if (!url) return;

                imgObj.loadError = null;
                const img = new Image();
                if (url.startsWith("blob:") || url.startsWith("data:")) img.crossOrigin = "anonymous";
                img.onload = () => {
                    imgObj.img = img; imgObj.loaded = true; imgObj.loadError = null;
                    draw(st);
                };
                img.onerror = () => { imgObj.loadError = true; imgObj.loaded = false; draw(st); };
                img.src = url;
            }

            function loadOneImage(st, inputName) {
                const idx = parseInt(inputName.replace("image", "")) - 1;
                if (idx < 0 || idx >= 9) return;
                loadImageForSlot(st, inputName, st.images[idx]);
            }

            function findSourceNode(st, inputName) {
                const inp = st.node.inputs?.find(i => i.name === inputName);
                if (!inp || inp.link == null) return null;
                const tid = inp.link;
                for (const n of (app.graph._nodes || [])) {
                    if (n === st.node || !n.outputs) continue;
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

            function loadAllImages(st) {
                for (let i = 0; i < 9; i++) loadOneImage(st, `image${i+1}`);
            }

            // ── Configure (workflow restore) ──
            const origCfg = this.configure;
            this.configure = function (data) {
                if (origCfg) origCfg.apply(this, arguments);
                const st = this._glState;
                if (!st) return;
                try {
                    const raw = getW("grid_config")?.value || "{}";
                    const cfg = JSON.parse(raw);
                    st.cols = cfg.cols ?? 3;
                    st.gap = cfg.gap ?? 4;
                    st.border = cfg.border ?? 0;
                    st.radius = cfg.radius ?? 0;
                    st.bg = cfg.bg ?? "#222222";
                    st.borderColor = cfg.border_color ?? "#444444";
                } catch (e) {}
                colsSlider.value = st.cols;
                colsVal.textContent = `${st.cols}`;
                gapSlider.value = st.gap;
                gapVal.textContent = `${st.gap}px`;
                bdrSlider.value = st.border;
                bdrVal.textContent = `${st.border}px`;
                radSlider.value = st.radius;
                radVal.textContent = `${st.radius}px`;
                bgSwatch.style.backgroundColor = st.bg;
                bgCi.value = st.bg;
                bdrSwatch.style.backgroundColor = st.borderColor;
                bdrCi.value = st.borderColor;
                setTimeout(() => { loadAllImages(st); }, 500);
            };

            // ── On executed ──
            const origExec = this.onExecuted;
            this.onExecuted = function (msg) {
                if (origExec) origExec.apply(this, arguments);
                const st = this._glState;
                if (!st) return;
                setTimeout(() => loadAllImages(st), 200);
            };

            // ── DOM widget ──
            const widget = this.addDOMWidget("gl_preview", "GL_PREVIEW", root, {
                getValue() { return ""; }, setValue() {},
            });
            widget.computeSize = (width) => {
                const w = Math.max(width || 350, 280);
                wrap.style.height = Math.max(180, Math.floor(w * 0.72)) + "px";
                return [w, Math.floor(w * 0.72) + 38];
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
                if (rt) cancelAnimationFrame(rt);
                this._glState = null;
                if (origRM) origRM.apply(this, arguments);
            };

            this.setSize([350, 260]);
            setTimeout(() => loadAllImages(s), 400);
            return rv;
        };
    },
});
