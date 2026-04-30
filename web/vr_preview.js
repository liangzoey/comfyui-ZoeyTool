import { app } from "/scripts/app.js";

const VR_STATE = {
    pannellumLoaded: false
};

function injectStyles() {
    if (document.getElementById('zoey-vr-styles')) return;
    const style = document.createElement('style');
    style.id = 'zoey-vr-styles';
    style.textContent = `
        /* 全屏遮罩层 */
        .zoey-vr-overlay {
            position: fixed !important;
            top: 0 !important; left: 0 !important;
            width: 100vw !important; height: 100vh !important;
            z-index: 999999 !important;
            background: rgba(0, 0, 0, 0.85) !important;
            display: flex;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(5px);
        }
        /* VR 容器 */
        .zoey-vr-container {
            width: 80vw !important;
            height: 80vh !important;
            background: #000;
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 0 50px rgba(0,0,0,0.9);
            border: 2px solid #444;
        }
        .zoey-vr-container canvas {
            width: 100% !important;
            height: 100% !important;
            display: block !important;
        }
        /* 关闭按钮 */
        .zoey-vr-close-btn {
            position: absolute;
            top: -40px;
            right: 0;
            padding: 8px 16px;
            background: #ff4444;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
        }
        .zoey-vr-close-btn:hover {
            background: #ff0000;
        }
        /* 标题 */
        .zoey-vr-title {
            position: absolute;
            top: -40px;
            left: 0;
            color: white;
            font-size: 18px;
            font-weight: bold;
        }
    `;
    document.head.appendChild(style);
}

async function loadPannellum() {
    if (window.pannellum) return true;
    console.log("[VR] 加载 Pannellum...");
    if (!document.getElementById('pannellum-css')) {
        const css = document.createElement('link');
        css.id = 'pannellum-css';
        css.rel = 'stylesheet';
        css.href = 'https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css';
        document.head.appendChild(css);
    }
    return new Promise((resolve, reject) => {
        if (!document.getElementById('pannellum-js')) {
            const js = document.createElement('script');
            js.id = 'pannellum-js';
            js.src = 'https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js';
            js.onload = () => resolve(true);
            js.onerror = () => reject(new Error('CDN 失败'));
            document.head.appendChild(js);
        } else {
            const check = setInterval(() => {
                if (window.pannellum) { clearInterval(check); resolve(true); }
            }, 100);
            setTimeout(() => { clearInterval(check); reject(new Error('超时')); }, 5000);
        }
    });
}

app.registerExtension({
    name: "zoey.vr.overlay_final",
    async nodeCreated(node) {
        if (node.comfyClass !== "VR360PreviewNode") return;
        console.log("[VR] 节点已注册 (全屏弹窗版)");
        injectStyles();

        node.addWidget("button", " 启动 VR (全屏弹窗)", "activate", async function (_, __, targetNode) {
            console.log("[VR] 按钮点击");

            let imgSrc = null;
            if (targetNode.imgs && targetNode.imgs.length > 0) {
                imgSrc = targetNode.imgs[0].src;
            } else if (targetNode.images && targetNode.images.length > 0) {
                const meta = targetNode.images[0];
                imgSrc = `/view?filename=${meta.filename}&subfolder=${meta.subfolder || ''}&type=${meta.type || 'output'}&t=${Date.now()}`;
            }

            if (!imgSrc) {
                alert("未找到图片！请先运行工作流。");
                return;
            }

            try {
                await loadPannellum();
                openVRPopup(imgSrc);
            } catch (err) {
                alert("VR 失败: " + err.message);
            }
        });
    }
});

function openVRPopup(src) {
    console.log("[VR] 打开全屏 VR 弹窗...");

    // 清理旧弹窗
    const oldOverlay = document.querySelector('.zoey-vr-overlay');
    if (oldOverlay) oldOverlay.remove();

    // 创建遮罩层
    const overlay = document.createElement('div');
    overlay.className = 'zoey-vr-overlay';

    // 创建 VR 容器
    const container = document.createElement('div');
    container.className = 'zoey-vr-container';

    // 标题
    const title = document.createElement('div');
    title.className = 'zoey-vr-title';
    title.textContent = '360° VR 预览';
    container.appendChild(title);

    // 关闭按钮
    const closeBtn = document.createElement('button');
    closeBtn.className = 'zoey-vr-close-btn';
    closeBtn.textContent = '❌ 关闭';
    container.appendChild(closeBtn);

    overlay.appendChild(container);
    document.body.appendChild(overlay);

    // 点击遮罩关闭
    overlay.onclick = (e) => {
        if (e.target === overlay) closeVR();
    };
    closeBtn.onclick = closeVR;

    function closeVR() {
        if (viewer) viewer.destroy();
        overlay.remove();
        console.log("[VR] 已关闭");
    }

    let viewer = null;
    // 初始化 Pannellum
    try {
        console.log("[VR] 初始化 Pannellum...");
        viewer = pannellum.viewer(container, {
            type: "equirectangular",
            panorama: src,
            autoLoad: true,
            showControls: true,
            showZoomCtrl: true,
            mouseZoom: true,
            draggable: true,
            hfov: 90
        });
        console.log("[VR] ✅ 成功");
    } catch (e) {
        console.error("[VR] 失败:", e);
        container.innerHTML += `<div style="color:red; padding:20px;">${e.message}</div>`;
    }
}
