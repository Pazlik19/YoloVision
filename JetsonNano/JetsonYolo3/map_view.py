"""
Статическая HTML-страница карты поля.

Рендерит на HTML5 <canvas> три слоя в единой метрической системе (мм):
  1. Поле  — сетка с шагом GRID_STEP, центр (0,0) = стартовая точка камеры.
  2. Камера — прямоугольник зоны сканирования (FOV) + значок в центре.
  3. Объекты — буквы из буфера сцены: яркие в FOV, тусклые вне кадра.

Данные тянутся опросом эндпоинта /data (см. JetsonCameraWEB.py) раз в POLL_MS.
Вынесено в отдельный модуль, чтобы не смешивать разметку с логикой сервера.
"""

MAP_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Карта поля</title>
<style>
  body { background:#1e1e1e; color:#ddd; font-family:sans-serif; margin:0;
         display:flex; flex-wrap:wrap; gap:16px; padding:16px; }
  canvas { background:#252525; display:block; border-radius:6px; }
  #panel { min-width:220px; }
  h3 { margin:6px 0; }
  .obj { font-size:13px; margin:2px 0; padding:2px 4px; border-radius:3px;
         background:#2c2c2c; }
  .muted { color:#888; }
  .legend span { display:inline-block; width:12px; height:12px; border-radius:50%;
                 margin-right:6px; vertical-align:middle; }
</style>
</head>
<body>
<canvas id="map" width="700" height="700"></canvas>
<div id="panel">
  <h3>Слово: <span id="word" style="color:#ffca28">—</span></h3>
  <div>Объектов на карте: <b id="cnt">0</b></div>
  <div>Камера: <span id="cam" class="muted">—</span></div>
  <div class="legend" style="margin:8px 0;">
    <div><span style="background:#ffca28"></span>в зоне сканирования</div>
    <div><span style="background:#777"></span>сохранён в памяти</div>
    <div><span style="background:#4caf50"></span>камера / FOV</div>
  </div>
  <hr style="border-color:#444">
  <div id="list"></div>
</div>

<script>
const cv  = document.getElementById('map');
const ctx = cv.getContext('2d');

// --- Параметры отображения ---
const POLL_MS   = 300;     // период опроса /data
const GRID_STEP = 250;     // шаг сетки, мм
const MARGIN    = 60;      // поля canvas вокруг содержимого, px
const MIN_HALF  = 1000;    // минимальный полупролёт поля, мм (чтобы не зумило в упор)

// Перевод координат поля (мм) -> пиксели canvas.
// Масштаб и центр пересчитываются каждый кадр под текущий охват сцены,
// поэтому карта автоматически вмещает и камеру, и все объекты.
let view = { scale: 0.3, cx: 0, cy: 0 };

function worldToScreen(x, y) {
  return [
    cv.width  / 2 + (x - view.cx) * view.scale,
    cv.height / 2 - (y - view.cy) * view.scale   // ось Y вверх (как в get_real_coords)
  ];
}

function recomputeView(cam, objects) {
  // Собираем габариты сцены: FOV камеры + все объекты.
  let xs = [cam.x - cam.fov_w/2, cam.x + cam.fov_w/2];
  let ys = [cam.y - cam.fov_h/2, cam.y + cam.fov_h/2];
  for (const o of objects) { xs.push(o.x); ys.push(o.y); }

  let minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);

  view.cx = (minX + maxX) / 2;
  view.cy = (minY + maxY) / 2;

  let halfX = Math.max((maxX - minX) / 2, MIN_HALF);
  let halfY = Math.max((maxY - minY) / 2, MIN_HALF);

  let sx = (cv.width  / 2 - MARGIN) / halfX;
  let sy = (cv.height / 2 - MARGIN) / halfY;
  view.scale = Math.min(sx, sy);
}

function drawGrid() {
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  ctx.fillStyle = '#555'; ctx.font = '10px sans-serif';

  // Диапазон сетки выводим от текущего центра вида в обе стороны.
  const spanX = (cv.width  / 2) / view.scale;
  const spanY = (cv.height / 2) / view.scale;
  const startX = Math.floor((view.cx - spanX) / GRID_STEP) * GRID_STEP;
  const endX   = Math.ceil ((view.cx + spanX) / GRID_STEP) * GRID_STEP;
  const startY = Math.floor((view.cy - spanY) / GRID_STEP) * GRID_STEP;
  const endY   = Math.ceil ((view.cy + spanY) / GRID_STEP) * GRID_STEP;

  for (let mm = startX; mm <= endX; mm += GRID_STEP) {
    const [sx] = worldToScreen(mm, 0);
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, cv.height); ctx.stroke();
  }
  for (let mm = startY; mm <= endY; mm += GRID_STEP) {
    const [, sy] = worldToScreen(0, mm);
    ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(cv.width, sy); ctx.stroke();
  }

  // Оси через мировой (0,0)
  const [ox, oy] = worldToScreen(0, 0);
  ctx.strokeStyle = '#4a4a4a'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, cv.height); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(cv.width, oy); ctx.stroke();
}

function drawCamera(cam) {
  const [cx, cy] = worldToScreen(cam.x, cam.y);
  const w = cam.fov_w * view.scale, h = cam.fov_h * view.scale;

  ctx.fillStyle = 'rgba(76,175,80,0.08)';
  ctx.fillRect(cx - w/2, cy - h/2, w, h);
  ctx.strokeStyle = '#4caf50'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
  ctx.strokeRect(cx - w/2, cy - h/2, w, h);
  ctx.setLineDash([]);

  ctx.fillStyle = '#4caf50';
  ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
}

function inFov(o, cam) {
  return Math.abs(o.x - cam.x) <= cam.fov_w / 2 &&
         Math.abs(o.y - cam.y) <= cam.fov_h / 2;
}

function drawObject(o, cam) {
  const [sx, sy] = worldToScreen(o.x, o.y);
  const active = inFov(o, cam);
  const color = active ? '#ffca28' : '#777';

  // Стрелка угла поворота буквы (angle_mid из буфера, [0..360)).
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(-o.angle * Math.PI / 180);   // Y вверх -> поворот против часовой
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, -16); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-4, -11); ctx.lineTo(0, -16); ctx.lineTo(4, -11); ctx.stroke();
  ctx.restore();

  // Точка + буква
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(sx, sy, 8, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#000'; ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(o.label, sx, sy);
}

async function tick() {
  let d;
  try {
    d = await (await fetch('/data')).json();
  } catch (e) { return; }

  const cam = d.camera || { x: 0, y: 0, fov_w: 300, fov_h: 200 };
  const objects = d.objects || [];

  recomputeView(cam, objects);

  ctx.clearRect(0, 0, cv.width, cv.height);
  drawGrid();
  drawCamera(cam);
  objects.forEach(o => drawObject(o, cam));

  document.getElementById('word').textContent = d.current_word || '—';
  document.getElementById('cnt').textContent  = objects.length;
  document.getElementById('cam').textContent  =
    `X=${cam.x} Y=${cam.y} мм · FOV ${cam.fov_w}×${cam.fov_h}`;
  document.getElementById('list').innerHTML = objects.map(o => {
    const cls = inFov(o, cam) ? 'obj' : 'obj muted';
    return `<div class="${cls}">#${o.id} <b>${o.label}</b> ` +
           `(${o.x|0}, ${o.y|0}) мм · ${o.angle|0}°</div>`;
  }).join('') || '<div class="muted">пока пусто</div>';
}

setInterval(tick, POLL_MS);
tick();
</script>
</body>
</html>"""
