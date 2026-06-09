export const VIEWER_HTML = `
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;overflow:hidden;font-family:monospace;touch-action:none}
canvas{display:block;width:100%;height:100%;cursor:grab}
canvas:active{cursor:grabbing}
.info{
  position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
  color:rgba(255,255,255,0.25);font-size:11px;pointer-events:none;
  text-align:center;white-space:nowrap
}
</style>
</head>
<body>
<div class="info" id="info">Az: <span id="azVal">0</span>&deg; El: <span id="elVal">30</span>&deg; &bull; 拖拽旋转 | 双击定位 | 滚轮调大小 | <span id="handleCount"></span></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\/script>
<script>
// --- Scene ---
var scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

var aspect = window.innerWidth / window.innerHeight;
var camera = new THREE.PerspectiveCamera(40, aspect, 0.1, 20);
camera.position.set(0, 0.3, 4.8);
camera.lookAt(0, 0, 0);

var renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0a0a0f, 1);
document.body.appendChild(renderer.domElement);

// --- Lights ---
var ambient = new THREE.AmbientLight(0x555577, 0.5);
scene.add(ambient);
var dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(1, 2, 4);
scene.add(dirLight);

// --- Orbit sphere (wireframe) ---
var SPHERE_R = 1.8;
var sphereGeo = new THREE.SphereGeometry(SPHERE_R, 24, 16);
var edges = new THREE.EdgesGeometry(sphereGeo);
var sphereWire = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
  color: 0x555588, transparent: true, opacity: 0.15
}));
scene.add(sphereWire);

// --- Image card ---
var cardMat = new THREE.MeshBasicMaterial({ color: 0x222233, side: THREE.DoubleSide });
var cardGeo = new THREE.PlaneGeometry(2, 2);
var card = new THREE.Mesh(cardGeo, cardMat);
scene.add(card);

var borderMat = new THREE.LineBasicMaterial({ color: 0x555588, transparent: true, opacity: 0.5 });
var border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(2, 2)), borderMat);
scene.add(border);

function setCardAspect(w, h) {
  var a = w / h;
  var cw = a >= 1 ? 2 : 2 * a;
  var ch = a >= 1 ? 2 / a : 2;
  card.scale.set(cw / 2, ch / 2, 1);
  scene.remove(border);
  border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(cw, ch)), borderMat);
  scene.add(border);
}

// --- Projected light spot on card (active handle) ---
var spotGroup = new THREE.Group();
spotGroup.position.z = 0.01;
card.add(spotGroup);

var glowMat = new THREE.MeshBasicMaterial({
  color: 0xffffff, transparent: true, opacity: 0.15,
  depthTest: false, depthWrite: false, side: THREE.DoubleSide
});
var ringMat = new THREE.MeshBasicMaterial({
  color: 0xffffff, transparent: true, opacity: 0.5,
  depthTest: false, depthWrite: false, side: THREE.DoubleSide
});
var dotMat = new THREE.MeshBasicMaterial({
  color: 0xffffff, transparent: true, opacity: 0.7,
  depthTest: false, depthWrite: false, side: THREE.DoubleSide
});

// Center dot (always visible regardless of shape)
var centerDot = new THREE.Mesh(new THREE.CircleGeometry(0.04, 16), dotMat);
spotGroup.add(centerDot);

// Circle shape
var circleGlow = new THREE.Mesh(new THREE.CircleGeometry(1, 32), glowMat);
spotGroup.add(circleGlow);
var circleRing = new THREE.Mesh(new THREE.RingGeometry(0.88, 1.0, 32), ringMat);
spotGroup.add(circleRing);

// Square shape
var sqGlow = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), glowMat);
sqGlow.visible = false;
spotGroup.add(sqGlow);
var sqRing = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.PlaneGeometry(2, 2)),
  new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, depthWrite: false })
);
sqRing.visible = false;
spotGroup.add(sqRing);

// Diamond shape
var diGlow = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), glowMat);
diGlow.rotation.z = Math.PI / 4;
diGlow.visible = false;
spotGroup.add(diGlow);
var diRing = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.PlaneGeometry(2, 2)),
  new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, depthWrite: false })
);
diRing.rotation.z = Math.PI / 4;
diRing.visible = false;
spotGroup.add(diRing);

var currentSpotShape = '圆形';

function setSpotShape(shape) {
  currentSpotShape = shape || '圆形';
  circleGlow.visible = currentSpotShape === '圆形';
  circleRing.visible = currentSpotShape === '圆形';
  sqGlow.visible = currentSpotShape === '方形';
  sqRing.visible = currentSpotShape === '方形';
  diGlow.visible = currentSpotShape === '菱形';
  diRing.visible = currentSpotShape === '菱形';
}

function updateProjectedSpot() {
  var azRad = azimuth * Math.PI / 180;
  var elRad = elevation * Math.PI / 180;
  var hx = 0.5 + 0.5 * Math.cos(elRad) * Math.sin(azRad);
  var hy = 0.5 - 0.5 * Math.sin(elRad);
  spotGroup.position.set((hx - 0.5) * 2, (0.5 - hy) * 2, 0.01);
  var s = Math.max(0.02, ballSize * 2);

  circleGlow.scale.set(s * 2.5, s * 2.5, 1);
  circleRing.scale.set(s, s, 1);
  sqGlow.scale.set(s, s, 1);
  sqRing.scale.set(s, s, 1);
  diGlow.scale.set(s, s, 1);
  diRing.scale.set(s, s, 1);

  // Dot scales with ball_size
  var dotR = Math.max(0.04, 0.04 * (ballSize / 0.3));
  centerDot.scale.set(dotR / 0.04, dotR / 0.04, 1);
}

// --- Light orb (active handle) ---
var orbGeo = new THREE.SphereGeometry(0.2, 24, 24);
var orbMat = new THREE.MeshStandardMaterial({
  color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.6,
  roughness: 0.1, metalness: 0.05
});
var orb = new THREE.Mesh(orbGeo, orbMat);
orb.renderOrder = 2;
scene.add(orb);

var orbOutline = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.SphereGeometry(1, 20, 16)),
  new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 })
);
orb.add(orbOutline);

// --- Ghost orb (visible when behind the card) ---
var ghostMat = new THREE.MeshBasicMaterial({
  color: 0xffffff, transparent: true, opacity: 0.0,
  depthWrite: false, side: THREE.DoubleSide
});
var ghostOrb = new THREE.Mesh(new THREE.SphereGeometry(0.2, 16, 16), ghostMat);
ghostOrb.renderOrder = 1;
scene.add(ghostOrb);

var ghostOutline = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.SphereGeometry(1, 16, 12)),
  new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.0 })
);
ghostOrb.add(ghostOutline);

// --- Direction ray (from orb through center) ---
var rayMat = new THREE.LineBasicMaterial({
  color: 0xffffff, transparent: true, opacity: 0.08, depthWrite: false
});
var rayPositions = new Float32Array(6);
var rayGeo = new THREE.BufferGeometry();
rayGeo.setAttribute('position', new THREE.BufferAttribute(rayPositions, 3));
var rayLine = new THREE.Line(rayGeo, rayMat);
rayLine.renderOrder = 0;
scene.add(rayLine);

// ===== EXTRA HANDLES (non-active) =====
var extraOrbs = [];        // { orb, outline, ghost, spotGroup }
var extraGhosts = [];
var extraSpotGroups = [];

function clearExtraHandles() {
  extraOrbs.forEach(function(o) {
    scene.remove(o);
  });
  extraGhosts.forEach(function(g) {
    scene.remove(g);
  });
  extraSpotGroups.forEach(function(sg) {
    card.remove(sg);
  });
  extraOrbs = [];
  extraGhosts = [];
  extraSpotGroups = [];
}

function rebuildExtraHandles(handlesArray, activeIdx) {
  clearExtraHandles();
  if (!handlesArray || handlesArray.length < 2) return;

  handlesArray.forEach(function(h, i) {
    if (i === activeIdx) return; // skip active handle

    var a = (h.azimuth || 0) * Math.PI / 180;
    var e = (h.elevation || 30) * Math.PI / 180;
    var col = new THREE.Color(h.light_color || '#FFFFFF');

    // --- Small orb on sphere ---
    var smallRad = 0.1;
    var sog = new THREE.SphereGeometry(smallRad, 16, 16);
    var som = new THREE.MeshStandardMaterial({
      color: col, emissive: col, emissiveIntensity: 0.25,
      roughness: 0.3, metalness: 0.05, transparent: true, opacity: 0.5
    });
    var smallOrb = new THREE.Mesh(sog, som);
    smallOrb.position.set(
      SPHERE_R * Math.cos(e) * Math.sin(a),
      SPHERE_R * Math.sin(e),
      SPHERE_R * Math.cos(e) * Math.cos(a)
    );
    smallOrb.renderOrder = 1;
    scene.add(smallOrb);
    extraOrbs.push(smallOrb);

    // --- Small outline ---
    var sol = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.SphereGeometry(smallRad, 12, 10)),
      new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.2 })
    );
    smallOrb.add(sol);

    // --- Ghost for behind-card ---
    var sGhostMat = new THREE.MeshBasicMaterial({
      color: col, transparent: true, opacity: 0.0,
      depthWrite: false, side: THREE.DoubleSide
    });
    var sGhost = new THREE.Mesh(new THREE.SphereGeometry(smallRad, 12, 12), sGhostMat);
    sGhost.renderOrder = 1;
    scene.add(sGhost);
    extraGhosts.push(sGhost);

    // --- Card spot (small, colored) ---
    var hx = 0.5 + 0.5 * Math.cos(e) * Math.sin(a);
    var hy = 0.5 - 0.5 * Math.sin(e);
    var s = Math.max(0.02, (h.ball_size || 0.3) * 2);

    var spotMat = new THREE.MeshBasicMaterial({
      color: col, transparent: true, opacity: 0.12,
      depthTest: false, depthWrite: false, side: THREE.DoubleSide
    });
    var spotMesh = new THREE.Mesh(new THREE.CircleGeometry(s, 24), spotMat);
    spotMesh.position.set((hx - 0.5) * 2, (0.5 - hy) * 2, 0.005);
    card.add(spotMesh);
    extraSpotGroups.push(spotMesh);

    // Small dot for this handle
    var tinyDotMat = new THREE.MeshBasicMaterial({
      color: col, transparent: true, opacity: 0.4,
      depthTest: false, depthWrite: false, side: THREE.DoubleSide
    });
    var tinyDot = new THREE.Mesh(new THREE.CircleGeometry(0.02, 8), tinyDotMat);
    tinyDot.position.set((hx - 0.5) * 2, (0.5 - hy) * 2, 0.006);
    card.add(tinyDot);
    extraSpotGroups.push(tinyDot);

    // Store ghost reference for update
    sGhost.userData = {
      pos: new THREE.Vector3(
        SPHERE_R * Math.cos(e) * Math.sin(a),
        SPHERE_R * Math.sin(e),
        SPHERE_R * Math.cos(e) * Math.cos(a)
      ),
      opacity: 0
    };
  });

  updateExtraGhosts();
}

function updateExtraGhosts() {
  extraGhosts.forEach(function(g) {
    var p = g.userData.pos;
    if (!p) return;
    var isBehind = p.z < 0;
    g.visible = isBehind;
    if (isBehind) {
      g.position.set(-p.x * 0.85, -p.y * 0.85, -p.z * 0.85);
      var ghostOpacity = Math.min(0.2, 0.08 + 0.12 * (Math.abs(p.z) / SPHERE_R));
      g.material.opacity = ghostOpacity;
    }
    g.material.needsUpdate = true;
  });
}

// --- State ---
var azimuth = 0, elevation = 30, ballSize = 0.3, lightColor = '#FFFFFF';
var isDragging = false;

function updateScene() {
  var a = azimuth * Math.PI / 180;
  var e = elevation * Math.PI / 180;

  // Orb position on sphere surface
  var ox = SPHERE_R * Math.cos(e) * Math.sin(a);
  var oy = SPHERE_R * Math.sin(e);
  var oz = SPHERE_R * Math.cos(e) * Math.cos(a);
  orb.position.set(ox, oy, oz);

  // Direction ray: from orb through center, extending past
  var dirLen = SPHERE_R + 1.0;
  rayPositions[0] = ox; rayPositions[1] = oy; rayPositions[2] = oz;
  rayPositions[3] = -ox * dirLen / SPHERE_R;
  rayPositions[4] = -oy * dirLen / SPHERE_R;
  rayPositions[5] = -oz * dirLen / SPHERE_R;
  rayGeo.attributes.position.needsUpdate = true;

  // Ghost orb: show when behind the card (z < 0)
  var isBehind = oz < 0;
  ghostOrb.visible = isBehind;
  if (isBehind) {
    var gx = -ox * 0.85;
    var gy = -oy * 0.85;
    var gz = -oz * 0.85;
    ghostOrb.position.set(gx, gy, gz);
    var ghostOpacity = Math.min(0.35, 0.15 + 0.2 * (Math.abs(oz) / SPHERE_R));
    ghostMat.opacity = ghostOpacity;
    ghostOutline.material.opacity = ghostOpacity * 0.6;
    ghostOrb.scale.copy(orb.scale);
  }
  ghostMat.needsUpdate = true;
  ghostOutline.material.needsUpdate = true;

  // Info display
  document.getElementById('azVal').textContent = Math.round(azimuth);
  document.getElementById('elVal').textContent = Math.round(elevation);
}

function setProjectedColor(hex) {
  var c = new THREE.Color(hex || '#FFFFFF');
  glowMat.color.copy(c);
  ringMat.color.copy(c);
  dotMat.color.copy(c);
  sqRing.material.color.copy(c);
  diRing.material.color.copy(c);
}

function setOrbColor(hex) {
  lightColor = hex || '#FFFFFF';
  var c = new THREE.Color(lightColor);
  orbMat.color.copy(c);
  orbMat.emissive.copy(c);
  setProjectedColor(lightColor);
}

function updateOrbSize(val) {
  ballSize = val;
  var s = 1 + 1.5 * ((val - 0.15) / 0.85);
  orb.scale.set(s, s, s);
}

function updateAll() {
  updateScene();
  updateProjectedSpot();
}

// --- Load image onto card ---
function loadImage(url) {
  if (!url) return;
  var img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = function() {
    setCardAspect(img.naturalWidth || img.width, img.naturalHeight || img.height);
    var tex = new THREE.Texture(img);
    tex.needsUpdate = true;
    cardMat.map = tex;
    cardMat.color.set(0xffffff);
    cardMat.needsUpdate = true;
  };
  img.src = url;
}

// --- Resize ---
function onResize() {
  var w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', onResize);

// --- Mouse: drag orb (mousedown anywhere) OR double-click card to position ---
var raycaster = new THREE.Raycaster();
var mouseVec = new THREE.Vector2();

function positionFromCardClick(e) {
  mouseVec.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouseVec.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouseVec, camera);
  var hits = raycaster.intersectObject(card);
  if (hits.length > 0) {
    var uv = hits[0].uv;
    if (!uv) return;
    var hx = uv.x;           // 0 left → 1 right
    var hy = 1 - uv.y;       // 0 top → 1 bottom
    // Inverse of: handle_x = 0.5 + 0.5*cos(el)*sin(az), handle_y = 0.5 - 0.5*sin(el)
    var elRad = Math.asin(Math.max(-1, Math.min(1, (0.5 - hy) / 0.5)));
    var cosEl = Math.cos(elRad);
    if (Math.abs(cosEl) > 0.001) {
      var azRad = Math.asin(Math.max(-1, Math.min(1, (hx - 0.5) / (0.5 * cosEl))));
      azimuth = Math.round(azRad * 180 / Math.PI);
    } else {
      azimuth = 0;  // gimbal lock at poles
    }
    elevation = Math.round(elRad * 180 / Math.PI);
    updateAll();
    window.parent.postMessage({ type: 'ANGLE_UPDATE', azimuth: azimuth, elevation: elevation }, '*');
  }
}

renderer.domElement.addEventListener('dblclick', function(e) {
  positionFromCardClick(e);
});

renderer.domElement.addEventListener('mousedown', function() { isDragging = true; });
window.addEventListener('mousemove', function(e) {
  if (!isDragging) return;
  azimuth = Math.max(-180, Math.min(180, azimuth + e.movementX * 0.7));
  elevation = Math.max(-90, Math.min(90, elevation - e.movementY * 0.7));
  updateAll();
  window.parent.postMessage({ type: 'ANGLE_UPDATE', azimuth: Math.round(azimuth), elevation: Math.round(elevation) }, '*');
});
window.addEventListener('mouseup', function() { isDragging = false; });

// --- Scroll → orb/spot size ---
renderer.domElement.addEventListener('wheel', function(e) {
  e.preventDefault();
  ballSize = Math.max(0.02, Math.min(1.0, ballSize + (e.deltaY > 0 ? -0.03 : 0.03)));
  ballSize = Math.round(ballSize * 100) / 100;
  updateOrbSize(ballSize);
  updateProjectedSpot();
  window.parent.postMessage({ type: 'BALL_SIZE_UPDATE', ballSize: ballSize }, '*');
}, { passive: false });

// --- Touch ---
var touchId = null, lastTX = 0, lastTY = 0;
renderer.domElement.addEventListener('touchstart', function(e) {
  if (touchId !== null) return;
  var t = e.changedTouches[0];
  touchId = t.identifier; lastTX = t.clientX; lastTY = t.clientY;
}, { passive: true });
renderer.domElement.addEventListener('touchmove', function(e) {
  var t = Array.from(e.changedTouches).find(function(x) { return x.identifier === touchId; });
  if (!t) return;
  azimuth = Math.max(-180, Math.min(180, azimuth + (t.clientX - lastTX) * 0.7));
  elevation = Math.max(-90, Math.min(90, elevation - (t.clientY - lastTY) * 0.7));
  lastTX = t.clientX; lastTY = t.clientY;
  updateAll();
  window.parent.postMessage({ type: 'ANGLE_UPDATE', azimuth: Math.round(azimuth), elevation: Math.round(elevation) }, '*');
}, { passive: true });
renderer.domElement.addEventListener('touchend', function(e) {
  var t = Array.from(e.changedTouches).find(function(x) { return x.identifier === touchId; });
  if (t) touchId = null;
}, { passive: true });

// --- Two-finger pinch → orb size (touch) ---
var lastPinchDist = 0;
renderer.domElement.addEventListener('touchstart', function(e) {
  if (e.touches.length === 2) {
    var dx = e.touches[0].clientX - e.touches[1].clientX;
    var dy = e.touches[0].clientY - e.touches[1].clientY;
    lastPinchDist = Math.sqrt(dx*dx + dy*dy);
  }
}, { passive: true });
renderer.domElement.addEventListener('touchmove', function(e) {
  if (e.touches.length === 2) {
    var dx = e.touches[0].clientX - e.touches[1].clientX;
    var dy = e.touches[0].clientY - e.touches[1].clientY;
    var dist = Math.sqrt(dx*dx + dy*dy);
    var delta = (dist - lastPinchDist) * 0.008;
    ballSize = Math.max(0.02, Math.min(1.0, ballSize + delta));
    ballSize = Math.round(ballSize * 100) / 100;
    lastPinchDist = dist;
    updateOrbSize(ballSize);
    updateProjectedSpot();
    window.parent.postMessage({ type: 'BALL_SIZE_UPDATE', ballSize: ballSize }, '*');
  }
}, { passive: true });

// --- postMessage from parent ---
window.addEventListener('message', function(e) {
  var d = e.data;
  if (!d || !d.type) return;

  if (d.type === 'INIT' || d.type === 'SYNC') {
    if (d.azimuth !== undefined) azimuth = d.azimuth;
    if (d.elevation !== undefined) elevation = d.elevation;
    if (d.lightColor) setOrbColor(d.lightColor);
    if (d.ballSize !== undefined) updateOrbSize(d.ballSize);
    if (d.handleShape) setSpotShape(d.handleShape);
    updateAll();
    setProjectedColor(lightColor);

    // Rebuild extra handles for multi-handle support
    if (d.handles) {
      rebuildExtraHandles(d.handles, d.activeIndex !== undefined ? d.activeIndex : -1);
    }
    // Update handle count
    var countEl = document.getElementById('handleCount');
    if (countEl && d.handles) {
      countEl.textContent = d.handles.length > 0 ? d.handles.length + ' 个手柄' : '';
    }
  }
  if (d.type === 'SHAPE_UPDATE') {
    if (d.handleShape) setSpotShape(d.handleShape);
    if (d.handles) {
      rebuildExtraHandles(d.handles, d.activeIndex !== undefined ? d.activeIndex : -1);
    }
  }
  if (d.type === 'UPDATE_IMAGE') loadImage(d.imageUrl);
  if (d.type === 'RESIZE') onResize();
});

// --- Signal ready ---
window.parent.postMessage({ type: 'VIEWER_READY' }, '*');

setOrbColor(lightColor);
updateAll();
updateOrbSize(ballSize);

// --- Animate ---
(function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
})();
<\/script>
</body>
</html>`;
