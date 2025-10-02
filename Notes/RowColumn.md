// p5.js — Input (left) and Output (right)
// Unit size: 24 px; black bg; 2-unit border around each grid
// Labels are inside the top border, aligned with the *second* unit row.
// Gap between panels: -2 units (overlap by 2 units).

// ---- Hyperparameters ----
const UNIT = 24;
const HEIGHT_CELLS = 4;     // <-- height in cells
const WIDTH_CELLS  = 4;     // <-- width in cells
const NOISE_CELLS  = 0;     // <-- number of noise cells (light-blue), e.g., 0..(H*W-2)

// Spacing / layout
const BORDER_UNITS = 2;     // 2-unit black border
const GAP_UNITS = -2;       // negative = overlap

// Colors
const BLACK = "#000000";
const BLUE = "#2d6cdf";        // seeds (ends)
const LIGHT_BLUE = "#a5c9ff";  // connectors AND noise

let h = HEIGHT_CELLS, w = WIDTH_CELLS;
let seeds = [];         // [{r,c}]
let noiseCells = [];    // [{r,c}]
let giGrid = [];        // input grid
let goGrid = [];        // output grid

function setup() {
  textFont('sans-serif');
  textSize(12);
  noStroke();

  const panelWUnits = w + BORDER_UNITS * 2;
  const panelHUnits = h + BORDER_UNITS * 2;
  const totalWUnits = panelWUnits * 2 + GAP_UNITS;  // negative gap overlaps panels
  const totalHUnits = panelHUnits;

  createCanvas(totalWUnits * UNIT, totalHUnits * UNIT);
  regenerate();
}

function mousePressed() {
  regenerate();
}

function regenerate() {
  // Start with black background
  giGrid = Array.from({ length: h }, () => Array(w).fill(BLACK));

  // --- EXACTLY ONE connecting orientation with extreme-end seeds ---
  const rowCase = random() < 0.5;

  if (rowCase) {
    // Single row, seeds at far left and far right
    const r = floor(random(h));
    seeds = [{ r, c: 0 }, { r, c: w - 1 }];
  } else {
    // Single column, seeds at top and bottom
    const c = floor(random(w));
    seeds = [{ r: 0, c }, { r: h - 1, c }];
  }

  // Place seeds
  for (const { r, c } of seeds) giGrid[r][c] = BLUE;

  // --- Add light-blue noise (exact count), avoiding seeds ---
  const maxNoise = Math.max(0, h * w - seeds.length);
  const noiseCount = Math.min(NOISE_CELLS, maxNoise);
  const exclude = new Set(seeds.map(p => cellKey(p.r, p.c)));
  noiseCells = sampleDistinctCellsExcluding(noiseCount, h, w, exclude);
  for (const { r, c } of noiseCells) giGrid[r][c] = LIGHT_BLUE;

  // --- Build output: connectors from BLUE seeds; noise stays LIGHT_BLUE ---
  goGrid = deepCopy(giGrid);

  // Horizontal connectors (trigger only if a row has ≥2 BLUE seeds)
  for (let rr = 0; rr < h; rr++) {
    const cols = [];
    for (let cc = 0; cc < w; cc++) if (giGrid[rr][cc] === BLUE) cols.push(cc);
    if (cols.length > 1) {
      const L = Math.min(...cols), R = Math.max(...cols);
      for (let cc = L; cc <= R; cc++) goGrid[rr][cc] = LIGHT_BLUE;
    }
  }

  // Vertical connectors (trigger only if a column has ≥2 BLUE seeds)
  for (let cc = 0; cc < w; cc++) {
    const rows = [];
    for (let rr = 0; rr < h; rr++) if (giGrid[rr][cc] === BLUE) rows.push(rr);
    if (rows.length > 1) {
      const T = Math.min(...rows), B = Math.max(...rows);
      for (let rr = T; rr <= B; rr++) goGrid[rr][cc] = LIGHT_BLUE;
    }
  }

  // Restore seeds to BLUE (so ends remain blue on top of connectors)
  for (const { r, c } of seeds) goGrid[r][c] = BLUE;

  // ---- Draw ----
  background(BLACK);

  const panelW = (w + 2 * BORDER_UNITS) * UNIT;
  const leftPanelX = 0;
  const topY = 0;

  // Left panel (input)
  drawPanel(leftPanelX, topY, giGrid);
  drawLabel("Input (seeds + light-blue noise)", leftPanelX + BORDER_UNITS * UNIT, topY + 1.5 * UNIT);

  // Right panel (output) — negative gap = overlap
  const rightPanelX = panelW + GAP_UNITS * UNIT;
  drawPanel(rightPanelX, topY, goGrid);
  drawLabel("Output (connect)", rightPanelX + BORDER_UNITS * UNIT, topY + 1.5 * UNIT);
}

// ---- Helpers ----
function drawPanel(offsetX, offsetY, grid) {
  // Full panel (grid + border)
  fill(BLACK);
  rect(offsetX, offsetY, (w + BORDER_UNITS * 2) * UNIT, (h + BORDER_UNITS * 2) * UNIT);

  // Inner grid inset by the border thickness
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      fill(grid[r][c]);
      rect(
        offsetX + (c + BORDER_UNITS) * UNIT,
        offsetY + (r + BORDER_UNITS) * UNIT,
        UNIT, UNIT
      );
    }
  }

  // Overlay gridlines (subtle)
  stroke(64);
  strokeWeight(1);
  for (let x = 0; x <= w; x++) {
    line(
      offsetX + (BORDER_UNITS + x) * UNIT, offsetY + BORDER_UNITS * UNIT,
      offsetX + (BORDER_UNITS + x) * UNIT, offsetY + (BORDER_UNITS + h) * UNIT
    );
  }
  for (let y = 0; y <= h; y++) {
    line(
      offsetX + BORDER_UNITS * UNIT, offsetY + (BORDER_UNITS + y) * UNIT,
      offsetX + (BORDER_UNITS + w) * UNIT, offsetY + (BORDER_UNITS + y) * UNIT
    );
  }
  noStroke();
}

function drawLabel(textStr, x, midY) {
  fill(255);
  textAlign(LEFT, CENTER);
  text(textStr, x + 2, midY); // small inset
}

function cellKey(r, c) { return `${r},${c}`; }

function sampleDistinctCellsExcluding(n, rows, cols, excludeSet) {
  const all = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!excludeSet.has(cellKey(r, c))) all.push({ r, c });
    }
  }
  shuffleArray(all);
  return all.slice(0, Math.min(n, all.length));
}

function shuffleArray(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = floor(random(i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
}

function deepCopy(arr2d) {
  return arr2d.map(row => row.slice());
}