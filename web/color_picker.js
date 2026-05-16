import { app } from "/scripts/app.js";

// 给所有 ZoeyMaskDrawBox 节点的 "框颜色" 字段替换为弹出调色板
app.registerExtension({
    name: "zoey.colorPicker",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ZoeyMaskDrawBox") return;

        // 保存原始的 onNodeCreated
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // 找到 "框颜色" widget
            const colorWidget = this.widgets?.find(w => w.name === "框颜色");
            if (!colorWidget) return r;

            // 保存原始输入元素
            const origInput = colorWidget.inputEl || colorWidget.element?.querySelector("input");

            // 创建颜色选择器容器
            const container = document.createElement("div");
            container.style.display = "flex";
            container.style.alignItems = "center";
            container.style.gap = "6px";
            container.style.padding = "2px 0";

            // 色块预览
            const swatch = document.createElement("div");
            swatch.style.width = "28px";
            swatch.style.height = "28px";
            swatch.style.borderRadius = "4px";
            swatch.style.border = "2px solid #555";
            swatch.style.cursor = "pointer";
            swatch.style.flexShrink = "0";
            swatch.style.backgroundColor = colorWidget.value || "#FF0000";

            // 隐藏的原生 color input
            const colorInput = document.createElement("input");
            colorInput.type = "color";
            colorInput.value = colorWidget.value || "#FF0000";
            colorInput.style.width = "0";
            colorInput.style.height = "0";
            colorInput.style.padding = "0";
            colorInput.style.border = "none";
            colorInput.style.position = "absolute";
            colorInput.style.opacity = "0";
            colorInput.style.pointerEvents = "none";

            // HEX 文本显示
            const hexLabel = document.createElement("span");
            hexLabel.style.fontSize = "12px";
            hexLabel.style.fontFamily = "monospace";
            hexLabel.style.color = "#ccc";
            hexLabel.textContent = colorWidget.value || "#FF0000";

            // 点击色块弹出调色板
            swatch.addEventListener("click", () => {
                colorInput.click();
            });

            // 选色后更新
            colorInput.addEventListener("input", () => {
                const val = colorInput.value;
                swatch.style.backgroundColor = val;
                hexLabel.textContent = val;
                colorWidget.value = val;
                // 触发 ComfyUI 的 widget 变化事件
                if (colorWidget.callback) {
                    colorWidget.callback(val);
                }
                // 标记节点已改变
                if (this.graph) {
                    this.setDirtyCanvas(true, true);
                }
            });

            container.appendChild(swatch);
            container.appendChild(hexLabel);
            container.appendChild(colorInput);

            // 替换 widget 显示元素
            if (colorWidget.element) {
                colorWidget.element.innerHTML = "";
                colorWidget.element.style.overflow = "visible";
                colorWidget.element.appendChild(container);
            } else if (colorWidget.parentEl) {
                colorWidget.parentEl.innerHTML = "";
                colorWidget.parentEl.appendChild(container);
            }

            // 支持通过 widget 值更新色块
            const origSetValue = colorWidget.setValue;
            if (origSetValue) {
                colorWidget.setValue = function (v) {
                    origSetValue.call(this, v);
                    swatch.style.backgroundColor = v;
                    hexLabel.textContent = v;
                    colorInput.value = v;
                };
            }

            return r;
        };
    },
});
