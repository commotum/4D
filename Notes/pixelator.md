// Paste into p5.js Web Editor as sketch.js
// Side-by-side: LEFT = original (128×128), RIGHT = downsampled.
// You only need to edit userDraw(g, ctx) below. No other params required.

// ————————————————————————————————————————————————
// 1) setup(128x128 canvas)
// ————————————————————————————————————————————————
const NATIVE = 512;           // native resolution for the source sketch (leave as-is)
let src, smallBuf, out;       // working buffers
let blockSize = 16;            // downsample block size (use [ / ] to change)

function setup() {
  // Main canvas is 256×128 so we can show both panels
  createCanvas(NATIVE * 2, NATIVE);
  pixelDensity(1);
  noStroke();
  // Offscreen "original" buffer your draw function renders into
  src = createGraphics(NATIVE, NATIVE); src.pixelDensity(1);
  // Lazily created/resized in downsampler()
  smallBuf = null;
  out = null;
}

// ————————————————————————————————————————————————
// 2) draw(all the drawing functions)
//    Call your code via userDraw(g, ctx). Replace ONLY that function.
// ————————————————————————————————————————————————
function draw() {
  // 2a) Let the user draw into `src` (do not draw on the main canvas here)
  src.clear();
  userDraw(src, { t: millis() * 0.001, fc: frameCount, w: src.width, h: src.height });

  // 2b) Downsample the latest `src`
  const downsampled = downsampler(src, blockSize);

  // 2c) Paint both panels to the main canvas
  background(240);

  // Left: original (smooth)
  push();
  smooth();
  image(src, 0, 0, NATIVE, NATIVE);
  pop();

  // Right: downsampled (crisp blocks)
  push();
  noSmooth();
  image(downsampled, NATIVE, 0, NATIVE, NATIVE);
  pop();

  // Optional labels/divider
  drawUI();
}

// —— Replace ONLY the body of this function with your own sketch ——
function userDraw(g, ctx) {
  // Example scene; delete/replace everything below.
  const { t, w, h } = ctx;

  // Background gradient
  for (let y = 0; y < h; y++) {
    const c = lerpColor(color(30, 40, 60), color(10, 120, 200), y / h);
    g.stroke(c);
    g.line(0, y, w, y);
  }

  // Moving shapes
  g.noStroke();
  g.fill(255, 240, 180);
  g.ellipse(w * (0.5 + 0.25 * sin(t)), h * 0.45, 360, 360);

  g.fill(245, 120, 90);
  g.ellipse(w * (0.5 + 0.30 * cos(t * 0.7)), h * 0.65, 64, 64);
}
// —— End user area ——

// Little helper UI (optional)
function drawUI() {
  stroke(0, 0, 0, 60);
  line(NATIVE, 8, NATIVE, height - 8);

  noStroke();
  fill(0, 0, 0, 120);
  textFont('monospace'); textSize(12);
  textAlign(LEFT, TOP);  text('Original (128×128)', 8, 8);
  textAlign(RIGHT, TOP); text(`Downsampled (block=${blockSize})`, width - 8, 8);
}

// Quick keyboard tweak for block size
function keyPressed() {
  if (key === '[') blockSize = max(1, blockSize - 1);
  if (key === ']') blockSize = min(64, blockSize + 1);
}

// ————————————————————————————————————————————————
// 3) downsampler()
//    shrink to tiny → scale back with no smoothing (nearest-neighbor)
// ————————————————————————————————————————————————
function downsampler(sourceGfx, block) {
  const b = max(1, floor(block));
  const dw = max(1, floor(sourceGfx.width  / b));
  const dh = max(1, floor(sourceGfx.height / b));

  // Create/resize tiny buffer
  if (!smallBuf || smallBuf.width !== dw || smallBuf.height !== dh) {
    smallBuf = createGraphics(dw, dh);
    smallBuf.pixelDensity(1);
    smallBuf.noSmooth();
  }

  // Create/resize output buffer
  if (!out || out.width !== sourceGfx.width || out.height !== sourceGfx.height) {
    out = createGraphics(sourceGfx.width, sourceGfx.height);
    out.pixelDensity(1);
    out.noSmooth();
  }

  // 1) shrink with averaging
  smallBuf.clear();
  smallBuf.image(sourceGfx, 0, 0, dw, dh);

  // 2) blow up with nearest-neighbor
  out.clear();
  out.image(smallBuf, 0, 0, out.width, out.height);

  return out;
}






More Functions:


// —— Replace ONLY the body of this function with your own sketch ——
function userDraw(g, ctx) {
  const { w, h, t } = ctx;

  // White background
  g.background(255);

  // Black outline, no fill
  g.noFill();
  g.stroke(0);
  g.strokeWeight(16);
  g.strokeJoin(ROUND);
  g.strokeCap(ROUND);

  // Center and rotate over time
  g.push();
  g.translate(w / 2, h / 2);
  g.rotate(t * 0.6); // radians per second

  // Equilateral triangle sized to canvas
  const R = Math.min(w, h) * 0.35;
  const a0 = -Math.PI / 2; // point "up" before rotation
  const x1 = R * Math.cos(a0),                 y1 = R * Math.sin(a0);
  const x2 = R * Math.cos(a0 + (2 * Math.PI/3)), y2 = R * Math.sin(a0 + (2 * Math.PI/3));
  const x3 = R * Math.cos(a0 + (4 * Math.PI/3)), y3 = R * Math.sin(a0 + (4 * Math.PI/3));

  g.triangle(x1, y1, x2, y2, x3, y3);
  g.pop();
}
// —— End user area ——




function setup() {
  createCanvas(512,512, WEBGL);
}

function draw() {
  let c1 = color(0, 0, 0)
  background(c1);
  let options = {freeRotation: true};
  orbitControl(1, 1, 1, options);
  normalMaterial();
  box(120, 200);
}

// is there a way to set "units" to make things easy? as constants?

https://natureofcode.com/

