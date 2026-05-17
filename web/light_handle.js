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

                // ── Hide light_color text widget ──
                const hideTextWidget = (name) => {
                    const w = node.widgets?.find(w => w.name === name);
                    if (!w) return;
                    w.computeSize = () => [0, -4];
                    setTimeout(() => { if (w.element) w.element.style.display = "none"; }, 50);
                };
                hideTextWidget("light_color");

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

                const syncViewer = () => {
                    if (!node._viewerReady || !iframe.contentWindow) return;
                    const azW = getW("azimuth");
                    const elW = getW("elevation");
                    const cW = getW("light_color");
                    const bsW = getW("ball_size");
                    const hsW = getW("handle_shape");
                    iframe.contentWindow.postMessage({
                        type: "SYNC",
                        azimuth: azW?.value ?? 0,
                        elevation: elW?.value ?? 30,
                        lightColor: cW?.value || "#FFFFFF",
                        ballSize: bsW?.value ?? 0.3,
                        handleShape: hsW?.value || "圆形",
                    }, "*");
                };

                colorInput.addEventListener("input", () => {
                    const val = colorInput.value.toUpperCase();
                    swatch.style.backgroundColor = val;
                    hexLabel.textContent = val;
                    const cw = getW("light_color");
                    if (cw) cw.value = val;
                    syncViewer();
                    app.graph.setDirtyCanvas(true, true);
                });

                colorBar.addEventListener("click", () => colorInput.click());
                colorBar.appendChild(swatch);
                colorBar.appendChild(hexLabel);
                colorBar.appendChild(colorInput);

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
                container.appendChild(colorBar);

                // ── DOM widget ──
                const widget = this.addDOMWidget("light_handle_viewer", "LIGHT_HANDLE", container, {
                    getValue() { return ""; },
                    setValue() {},
                });
                widget.computeSize = (width) => [Math.max(width || 350, 350), 320];

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
                        }, "*");
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
                        updateSwatch();
                        app.graph.setDirtyCanvas(true, true);
                    } else if (data.type === "BALL_SIZE_UPDATE") {
                        const bsW = getW("ball_size");
                        if (bsW) bsW.value = data.ballSize;
                        app.graph.setDirtyCanvas(true, true);
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
                    setTimeout(() => retryLoad(12, 400), 200);
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
                    if (["azimuth", "elevation", "light_color", "ball_size"].includes(name)) {
                        if (name === "light_color") updateSwatch();
                        syncViewer();
                    }
                    if (name === "handle_shape" && node._viewerReady && iframe.contentWindow) {
                        iframe.contentWindow.postMessage({ type: "SHAPE_UPDATE", handleShape: value }, "*");
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

                this.setSize([350, 400]);
                setTimeout(() => retryLoad(12, 400), 300);

                return r;
            };
        }
    },
});
