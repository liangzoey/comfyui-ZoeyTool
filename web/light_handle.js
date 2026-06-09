import { app } from "../../../../scripts/app.js";
import { VIEWER_HTML } from "./light_handle_3d_viewer.js";

app.registerExtension({
    name: "zoey.lightHandle",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ZoeyLightHandle") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                const getW = (name) => node.widgets?.find((w) => w.name === name);

                // ── Debounce ──
                const debounce = (fn, ms) => {
                    let timer;
                    return (...args) => {
                        clearTimeout(timer);
                        timer = setTimeout(() => fn(...args), ms);
                    };
                };

                // ── Flag to prevent widget-change double sync ──
                let _inColorPick = false;

                // ── Handle state ──
                let handles = [];
                let activeHandleIndex = 0;

                // ── Hide text widgets ──
                const hideTextWidget = (name) => {
                    const w = node.widgets?.find(w => w.name === name);
                    if (!w) return;
                    w.computeSize = () => [0, -4];
                    setTimeout(() => { if (w.element) w.element.style.display = "none"; }, 50);
                };
                hideTextWidget("light_color");
                hideTextWidget("handles_json");

                // ── Color bar ──
                const colorBar = document.createElement("div");
                colorBar.style.display = "flex";
                colorBar.style.alignItems = "center";
                colorBar.style.gap = "6px";
                colorBar.style.padding = "2px 4px";
                colorBar.style.cursor = "pointer";
                colorBar.style.borderRadius = "4px";
                colorBar.style.border = "1px solid rgba(255,255,255,0.1)";
                colorBar.style.backgroundColor = "rgba(10,10,15,0.6)";

                const swatch = document.createElement("div");
                swatch.style.width = "20px";
                swatch.style.height = "20px";
                swatch.style.borderRadius = "3px";
                swatch.style.border = "1px solid rgba(255,255,255,0.2)";
                swatch.style.flexShrink = "0";

                const hexLabel = document.createElement("span");
                hexLabel.style.fontSize = "11px";
                hexLabel.style.color = "rgba(255,255,255,0.6)";
                hexLabel.style.fontFamily = "monospace";

                const colorInput = document.createElement("input");
                colorInput.type = "color";
                colorInput.style.display = "none";

                const updateSwatch = () => {
                    const cw = getW("light_color");
                    const val = cw?.value || "#FFFFFF";
                    swatch.style.backgroundColor = val;
                    hexLabel.textContent = val.toUpperCase();
                };

                // ── Debounced operations (avoid jank during rapid input) ──
                const syncViewer = debounce(() => {
                    if (!node._viewerReady || !iframe.contentWindow) return;
                    iframe.contentWindow.postMessage({
                        type: "SYNC",
                        azimuth: getW("azimuth")?.value ?? 0,
                        elevation: getW("elevation")?.value ?? 30,
                        lightColor: getW("light_color")?.value || "#FFFFFF",
                        ballSize: getW("ball_size")?.value ?? 0.3,
                        handleShape: getW("handle_shape")?.value || "圆形",
                        handles: handles,
                        activeIndex: activeHandleIndex,
                    }, "*");
                }, 30);
                const saveHandlesToJson = debounce(() => {
                    const w = getW("handles_json");
                    if (w) w.value = JSON.stringify(handles);
                }, 30);
                const markDirty = debounce(() => app.graph.setDirtyCanvas(true, true), 50);

                colorInput.addEventListener("input", () => {
                    const val = colorInput.value.toUpperCase();
                    swatch.style.backgroundColor = val;
                    hexLabel.textContent = val;
                    const cw = getW("light_color");
                    if (cw) {
                        _inColorPick = true;
                        cw.value = val;
                        _inColorPick = false;
                    }
                    // Update current handle color
                    if (handles.length > 0 && activeHandleIndex < handles.length) {
                        handles[activeHandleIndex].light_color = val;
                        saveHandlesToJson();
                        renderHandleBar();
                    }
                    syncViewer();
                    markDirty();
                });

                colorBar.addEventListener("click", () => colorInput.click());
                colorBar.appendChild(swatch);
                colorBar.appendChild(hexLabel);
                colorBar.appendChild(colorInput);

                // ── Handle management functions ──
                const loadHandlesFromJson = () => {
                    const w = getW("handles_json");
                    if (w && w.value && w.value !== "[]") {
                        try {
                            const parsed = JSON.parse(w.value);
                            handles = parsed.map(h => ({
                                x: h.x ?? 0.5,
                                y: h.y ?? 0.5,
                                azimuth: h.azimuth ?? 0,
                                elevation: h.elevation ?? 30,
                                ball_size: h.ball_size ?? 0.3,
                                handle_shape: h.handle_shape ?? "圆形",
                                light_color: h.light_color ?? "#FFFFFF",
                                intensity: h.intensity ?? 5.0,
                                behind_subject: h.behind_subject ?? false,
                                light_type: h.light_type ?? "摄影棚灯光",
                            }));
                            activeHandleIndex = 0;
                        } catch (e) {
                            handles = [];
                        }
                    } else {
                        handles = [];
                        activeHandleIndex = -1;
                    }
                };

                const addHandle = () => {
                    const az = getW("azimuth")?.value ?? 0;
                    const el = getW("elevation")?.value ?? 30;
                    const azRad = az * Math.PI / 180;
                    const elRad = el * Math.PI / 180;
                    const h = {
                        x: Math.max(0, Math.min(1, 0.5 + 0.5 * Math.cos(elRad) * Math.sin(azRad))),
                        y: Math.max(0, Math.min(1, 0.5 - 0.5 * Math.sin(elRad))),
                        azimuth: az,
                        elevation: el,
                        ball_size: getW("ball_size")?.value ?? 0.3,
                        handle_shape: getW("handle_shape")?.value ?? "圆形",
                        light_color: getW("light_color")?.value ?? "#FFFFFF",
                        intensity: getW("intensity")?.value ?? 5.0,
                        behind_subject: getW("behind_subject")?.value ?? false,
                        light_type: getW("light_type")?.value ?? "摄影棚灯光",
                    };
                    handles.push(h);
                    activeHandleIndex = handles.length - 1;
                    saveHandlesToJson();
                    renderHandleBar();
                    syncViewer();
                    markDirty();
                };

                const selectHandle = (idx) => {
                    if (idx < 0 || idx >= handles.length) return;
                    activeHandleIndex = idx;
                    const h = handles[idx];
                    const azW = getW("azimuth"); if (azW) azW.value = h.azimuth;
                    const elW = getW("elevation"); if (elW) elW.value = h.elevation;
                    const bsW = getW("ball_size"); if (bsW) bsW.value = h.ball_size;
                    const hsW = getW("handle_shape"); if (hsW) hsW.value = h.handle_shape;
                    const lcW = getW("light_color"); if (lcW) lcW.value = h.light_color;
                    const inW = getW("intensity"); if (inW) inW.value = h.intensity;
                    const bhW = getW("behind_subject"); if (bhW) bhW.value = h.behind_subject ?? false;
                    const ltW = getW("light_type"); if (ltW) ltW.value = h.light_type ?? "摄影棚灯光";
                    updateSwatch();
                    renderHandleBar();
                    syncViewer();
                    markDirty();
                };

                const deleteHandle = (idx) => {
                    if (handles.length === 0) return;
                    handles.splice(idx, 1);
                    if (activeHandleIndex >= handles.length) activeHandleIndex = Math.max(0, handles.length - 1);
                    saveHandlesToJson();
                    renderHandleBar();
                    syncViewer();
                    if (handles.length > 0) {
                        selectHandle(activeHandleIndex);
                    } else {
                        // Reset to defaults
                        activeHandleIndex = -1;
                    }
                    markDirty();
                };

                // ── Handle bar ──
                const handleBarContainer = document.createElement("div");
                handleBarContainer.style.display = "flex";
                handleBarContainer.style.alignItems = "center";
                handleBarContainer.style.gap = "4px";
                handleBarContainer.style.padding = "2px 4px";
                handleBarContainer.style.flexWrap = "wrap";
                handleBarContainer.style.borderRadius = "4px";
                handleBarContainer.style.backgroundColor = "rgba(10,10,15,0.4)";
                handleBarContainer.style.minHeight = "24px";

                const handleBarLabel = document.createElement("span");
                handleBarLabel.textContent = "手柄:";
                handleBarLabel.style.fontSize = "10px";
                handleBarLabel.style.color = "rgba(255,255,255,0.4)";
                handleBarLabel.style.marginRight = "2px";
                handleBarContainer.appendChild(handleBarLabel);

                const handleDots = document.createElement("div");
                handleDots.style.display = "flex";
                handleDots.style.alignItems = "center";
                handleDots.style.gap = "4px";
                handleDots.style.flex = "1";
                handleDots.style.flexWrap = "wrap";
                handleBarContainer.appendChild(handleDots);

                const handleActions = document.createElement("div");
                handleActions.style.display = "flex";
                handleActions.style.alignItems = "center";
                handleActions.style.gap = "2px";
                handleBarContainer.appendChild(handleActions);

                const renderHandleBar = debounce(() => {
                    handleDots.innerHTML = "";
                    handleActions.innerHTML = "";

                    if (handles.length === 0) {
                        const emptyLabel = document.createElement("span");
                        emptyLabel.textContent = "无";
                        emptyLabel.style.fontSize = "10px";
                        emptyLabel.style.color = "rgba(255,255,255,0.25)";
                        handleDots.appendChild(emptyLabel);
                    } else {
                        handles.forEach((h, i) => {
                            const dot = document.createElement("div");
                            dot.style.width = "14px";
                            dot.style.height = "14px";
                            dot.style.borderRadius = "50%";
                            dot.style.backgroundColor = h.light_color || "#FFFFFF";
                            dot.style.border = i === activeHandleIndex
                                ? "2px solid #FFD700"
                                : "1px solid rgba(255,255,255,0.25)";
                            dot.style.cursor = "pointer";
                            dot.style.flexShrink = "0";
                            dot.style.boxSizing = "border-box";
                            dot.title = `手柄 ${i + 1}`;
                            dot.addEventListener("click", () => selectHandle(i));
                            handleDots.appendChild(dot);
                        });
                    }

                    // Delete button
                    if (handles.length > 0) {
                        const delBtn = document.createElement("button");
                        delBtn.textContent = "✕";
                        delBtn.title = "删除当前手柄";
                        delBtn.style.background = "rgba(200,50,50,0.4)";
                        delBtn.style.border = "1px solid rgba(200,50,50,0.6)";
                        delBtn.style.borderRadius = "3px";
                        delBtn.style.color = "#fff";
                        delBtn.style.fontSize = "10px";
                        delBtn.style.cursor = "pointer";
                        delBtn.style.padding = "1px 5px";
                        delBtn.style.lineHeight = "1.2";
                        delBtn.addEventListener("click", (e) => {
                            e.stopPropagation();
                            if (activeHandleIndex >= 0 && activeHandleIndex < handles.length) {
                                deleteHandle(activeHandleIndex);
                            }
                        });
                        handleActions.appendChild(delBtn);
                    }
                }, 30);

                // ── Add Handle button ──
                const addBtn = document.createElement("button");
                addBtn.textContent = "＋ 添加手柄";
                addBtn.title = "将当前灯光参数保存为一个新手柄";
                addBtn.style.background = "rgba(70,130,200,0.35)";
                addBtn.style.border = "1px solid rgba(70,130,200,0.5)";
                addBtn.style.borderRadius = "3px";
                addBtn.style.color = "#fff";
                addBtn.style.fontSize = "10px";
                addBtn.style.cursor = "pointer";
                addBtn.style.padding = "1px 6px";
                addBtn.style.lineHeight = "1.4";
                addBtn.style.whiteSpace = "nowrap";
                addBtn.addEventListener("click", addHandle);

                // ── Refresh button ──
                const refreshBtn = document.createElement("button");
                refreshBtn.textContent = "⟳";
                refreshBtn.title = "刷新图像预览（手动重载上游图像）";
                refreshBtn.style.cssText = "font-size:15px;background:rgba(10,10,15,0.6);border:1px solid rgba(255,255,255,0.15);border-radius:4px;color:#4fc3f7;cursor:pointer;height:22px;width:26px;padding:0 0 2px;text-align:center;line-height:1;flex:none;";
                refreshBtn.addEventListener("click", () => retryLoad(30, 200));

                // ── Container / iframe 3D viewer ──
                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.width = "100%";
                container.style.gap = "4px";

                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.height = "280px";
                iframe.style.border = "1px solid rgba(255,255,255,0.1)";
                iframe.style.borderRadius = "8px";
                iframe.style.backgroundColor = "#0a0a0f";
                iframe.style.display = "block";
                iframe.allow = "autoplay";

                const blob = new Blob([VIEWER_HTML], { type: "text/html" });
                const blobUrl = URL.createObjectURL(blob);
                iframe.src = blobUrl;
                iframe.addEventListener("load", () => { iframe._blobUrl = blobUrl; });

                container.appendChild(iframe);

                // ── Controls row: color bar + add button ──
                const controlsRow = document.createElement("div");
                controlsRow.style.display = "flex";
                controlsRow.style.alignItems = "center";
                controlsRow.style.gap = "4px";
                controlsRow.style.width = "100%";
                colorBar.style.flex = "1";
                controlsRow.appendChild(colorBar);
                controlsRow.appendChild(refreshBtn);
                controlsRow.appendChild(addBtn);
                container.appendChild(controlsRow);

                // ── Handle bar ──
                container.appendChild(handleBarContainer);

                // ── DOM widget ──
                const widget = this.addDOMWidget("light_handle_viewer", "LIGHT_HANDLE", container, {
                    getValue() { return ""; },
                    setValue() {},
                });
                widget.computeSize = (width) => [Math.max(width || 350, 350), 360];

                // ── Viewer state ──
                node._viewerIframe = iframe;
                node._viewerReady = false;
                node._pendingImageSend = null;

                // ── postMessage handler ──
                const onMessage = (event) => {
                    if (event.source !== iframe.contentWindow) return;
                    const data = event.data;
                    if (!data || !data.type) return;

                    if (data.type === "VIEWER_READY") {
                        node._viewerReady = true;
                        loadHandlesFromJson();
                        if (handles.length > 0) selectHandle(activeHandleIndex);
                        const azW = getW("azimuth");
                        const elW = getW("elevation");
                        const cW = getW("light_color");
                        const bsW = getW("ball_size");
                        const hsW = getW("handle_shape");
                        iframe.contentWindow.postMessage({
                            type: "INIT",
                            azimuth: azW?.value ?? 0,
                            elevation: elW?.value ?? 30,
                            lightColor: cW?.value || "#FFFFFF",
                            ballSize: bsW?.value ?? 0.3,
                            handleShape: hsW?.value || "圆形",
                            handles: handles,
                            activeIndex: activeHandleIndex,
                        }, "*");
                        renderHandleBar();
                        if (node._pendingImageSend) {
                            node._pendingImageSend();
                            node._pendingImageSend = null;
                        }
                    } else if (data.type === "ANGLE_UPDATE") {
                        const azW = getW("azimuth");
                        const elW = getW("elevation");
                        const bsW = getW("behind_subject");
                        if (azW) azW.value = data.azimuth;
                        if (elW) elW.value = data.elevation;
                        // Auto-toggle behind_subject: behind when |azimuth| > 90
                        if (bsW) {
                            const behind = Math.abs(data.azimuth) > 90;
                            if (bsW.value !== behind) {
                                bsW.value = behind;
                            }
                        }
                        // Update current handle in list
                        if (activeHandleIndex >= 0 && activeHandleIndex < handles.length) {
                            const a = data.azimuth * Math.PI / 180;
                            const e = data.elevation * Math.PI / 180;
                            handles[activeHandleIndex].x = Math.max(0, Math.min(1, 0.5 + 0.5 * Math.cos(e) * Math.sin(a)));
                            handles[activeHandleIndex].y = Math.max(0, Math.min(1, 0.5 - 0.5 * Math.sin(e)));
                            handles[activeHandleIndex].azimuth = data.azimuth;
                            handles[activeHandleIndex].elevation = data.elevation;
                            saveHandlesToJson();
                        }
                        updateSwatch();
                        markDirty();
                    } else if (data.type === "BALL_SIZE_UPDATE") {
                        const bsW = getW("ball_size");
                        if (bsW) bsW.value = data.ballSize;
                        if (activeHandleIndex >= 0 && activeHandleIndex < handles.length) {
                            handles[activeHandleIndex].ball_size = data.ballSize;
                            saveHandlesToJson();
                        }
                        markDirty();
                    }
                };
                window.addEventListener("message", onMessage);

                // ── Image: convert to data URL, send to viewer ──
                const sendImageToViewer = (url) => {
                    const send = () => {
                        if (iframe.contentWindow) {
                            iframe.contentWindow.postMessage({ type: "UPDATE_IMAGE", imageUrl: url }, "*");
                        }
                    };
                    if (node._viewerReady) send();
                    else node._pendingImageSend = send;
                };

                const loadImage = (url) => {
                    const img = new Image();
                    img.crossOrigin = "anonymous";
                    img.onload = () => {
                        const c = document.createElement("canvas");
                        c.width = img.naturalWidth || img.width;
                        c.height = img.naturalHeight || img.height;
                        const cx = c.getContext("2d");
                        cx.drawImage(img, 0, 0);
                        sendImageToViewer(c.toDataURL("image/png"));
                    };
                    img.onerror = () => {};
                    img.src = url;
                };

                // ── Try to load image from connected input node ──
                const tryLoadFromInput = () => {
                    const imgInput = node.inputs?.find(inp => inp.name === "image");
                    if (!imgInput || imgInput.link == null) return false;

                    const targetLinkId = imgInput.link;
                    let srcNode = null;
                    for (const n of (app.graph._nodes || [])) {
                        if (n === node || !n.outputs) continue;
                        for (const output of n.outputs) {
                            if (!output.links) continue;
                            for (const lid of output.links) {
                                if (lid != null && lid === targetLinkId) { srcNode = n; break; }
                            }
                            if (srcNode) break;
                        }
                        if (srcNode) break;
                    }
                    if (!srcNode) {
                        const linkData = app.graph.links?.[targetLinkId];
                        if (linkData) srcNode = app.graph.getNodeById(linkData[0]);
                        if (!srcNode) return false;
                    }

                    if (srcNode.imgs && srcNode.imgs.length > 0) {
                        const first = srcNode.imgs[0];
                        const src = typeof first === "string" ? first : (first?.src || first?._src || null);
                        if (src) { loadImage(src); return true; }
                    }
                    if (srcNode.image) {
                        const src = typeof srcNode.image === "string" ? srcNode.image : srcNode.image?.src;
                        if (src) { loadImage(src); return true; }
                    }
                    const imgWidget = srcNode.widgets?.find(w =>
                        w.name === "image" && typeof w.value === "string" && /\.\w+$/.test(w.value)
                    );
                    if (imgWidget) {
                        const parts = imgWidget.value.split("/");
                        const fn = parts.pop();
                        const sub = parts.join("/");
                        loadImage(
                          window.location.origin +
                          `/view?filename=${encodeURIComponent(fn)}&type=input&subfolder=${encodeURIComponent(sub)}&rand=${Date.now()}`
                        );
                        return true;
                    }
                    for (const w of (srcNode.widgets || [])) {
                        if (typeof w.value === "string" && /\.(png|jpg|jpeg|webp|bmp)$/i.test(w.value)) {
                            const parts = w.value.split("/");
                            const fn = parts.pop();
                            const sub = parts.join("/");
                            loadImage(
                              window.location.origin +
                              `/view?filename=${encodeURIComponent(fn)}&type=input&subfolder=${encodeURIComponent(sub)}&rand=${Date.now()}`
                            );
                            return true;
                        }
                    }
                    return false;
                };

                const retryLoad = (maxRetries = 10, delay = 500) => {
                    let tries = 0;
                    const attempt = () => {
                        tries++;
                        if (tryLoadFromInput()) return;
                        if (tries < maxRetries) setTimeout(attempt, delay);
                    };
                    attempt();
                };

                // ── Connection change ──
                const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
                nodeType.prototype.onConnectionsChange = function (slotType, slot, isConnected, link, outputSlot) {
                    if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
                    if (slotType === 1 && node.inputs[slot]?.name === "image") {
                        if (isConnected) {
                            if (!tryLoadFromInput()) {
                                setTimeout(() => retryLoad(8, 400), 100);
                            }
                        }
                    }
                };

                // ── Configure (workflow restore) ──
                const origConfigure = this.configure;
                this.configure = function (data) {
                    if (origConfigure) origConfigure.apply(this, arguments);
                    setTimeout(() => {
                        loadHandlesFromJson();
                        renderHandleBar();
                        if (handles.length > 0) selectHandle(activeHandleIndex);
                        retryLoad(12, 400);
                    }, 200);
                };

                // ── Receive executed image ──
                const onExecuted = this.onExecuted;
                this.onExecuted = function (message) {
                    if (onExecuted) onExecuted.apply(this, arguments);
                    if (message?.image_base64?.[0]) {
                        sendImageToViewer(message.image_base64[0]);
                    }
                };

                // ── Widget change → sync viewer ──
                const origOnWidgetChanged = this.onWidgetChanged;
                this.onWidgetChanged = function (name, value, old_val, w) {
                    if (origOnWidgetChanged) origOnWidgetChanged.apply(this, arguments);
                    // Skip light_color changes triggered by the color picker (already handled)
                    if (name === "light_color" && _inColorPick) return;
                    if (["azimuth", "elevation", "light_color", "ball_size", "intensity", "behind_subject", "light_type"].includes(name)) {
                        if (name === "light_color") updateSwatch();
                        // Update current handle in list on value change
                        if (activeHandleIndex >= 0 && activeHandleIndex < handles.length) {
                            if (name === "azimuth" || name === "elevation") {
                                const az = getW("azimuth")?.value ?? 0;
                                const el = getW("elevation")?.value ?? 30;
                                const a = az * Math.PI / 180;
                                const e = el * Math.PI / 180;
                                handles[activeHandleIndex].x = Math.max(0, Math.min(1, 0.5 + 0.5 * Math.cos(e) * Math.sin(a)));
                                handles[activeHandleIndex].y = Math.max(0, Math.min(1, 0.5 - 0.5 * Math.sin(e)));
                                handles[activeHandleIndex].azimuth = az;
                                handles[activeHandleIndex].elevation = el;
                            } else if (name === "ball_size") handles[activeHandleIndex].ball_size = value;
                            else if (name === "light_color") handles[activeHandleIndex].light_color = value;
                            else if (name === "intensity") handles[activeHandleIndex].intensity = value;
                            else if (name === "behind_subject") handles[activeHandleIndex].behind_subject = value;
                            else if (name === "light_type") handles[activeHandleIndex].light_type = value;
                            saveHandlesToJson();
                            renderHandleBar();
                        }
                        syncViewer();
                    }
                    if (name === "handle_shape") {
                        if (activeHandleIndex >= 0 && activeHandleIndex < handles.length) {
                            handles[activeHandleIndex].handle_shape = value;
                            saveHandlesToJson();
                        }
                        if (node._viewerReady && iframe.contentWindow) {
                            iframe.contentWindow.postMessage({
                                type: "SHAPE_UPDATE",
                                handleShape: value,
                                handles: handles,
                                activeIndex: activeHandleIndex,
                            }, "*");
                        }
                    }
                };

                // ── Resize → notify viewer ──
                let resizeTimeout = null;
                const resizeObserver = new ResizeObserver(() => {
                    if (resizeTimeout) clearTimeout(resizeTimeout);
                    resizeTimeout = setTimeout(() => {
                        if (node._viewerReady && iframe.contentWindow) {
                            iframe.contentWindow.postMessage({ type: "RESIZE" }, "*");
                        }
                    }, 100);
                });
                resizeObserver.observe(iframe);

                // ── Global graph change listener ──
                app.graph.onAfterChange = (() => {
                    const orig = app.graph.onAfterChange;
                    return function (event) {
                        if (orig) orig.apply(this, arguments);
                        setTimeout(() => retryLoad(5, 500), 100);
                    };
                })();

                // ── Cleanup ──
                const origOnRemoved = this.onRemoved;
                this.onRemoved = function () {
                    resizeObserver.disconnect();
                    window.removeEventListener("message", onMessage);
                    if (resizeTimeout) clearTimeout(resizeTimeout);
                    node._pendingImageSend = null;
                    node._viewerReady = false;
                    node._viewerIframe = null;
                    if (iframe._blobUrl) URL.revokeObjectURL(iframe._blobUrl);
                    if (origOnRemoved) origOnRemoved.apply(this, arguments);
                };

                // ── Init ──
                loadHandlesFromJson();
                renderHandleBar();

                this.setSize([350, 430]);
                setTimeout(() => retryLoad(12, 400), 300);

                return r;
            };
        }
    },
});
