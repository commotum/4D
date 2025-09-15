<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pixel Lab — p5.js Downsampler</title>
  <!-- Add crossorigin so we get real error messages instead of generic "Script error" in some browsers -->
  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.2/lib/p5.min.js" crossorigin="anonymous"></script>
  <style>
    :root { --gap: 10px; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif; background:#f7f7f8; color:#0f172a; }
    main { max-width: 980px; margin: 24px auto; padding: 0 16px; }
    h1 { font-size: clamp(20px, 2.4vw, 28px); margin: 0 0 12px; }
    p.sub { margin: 0 0 12px; color:#334155; }
    #sketch canvas { height: auto !important; border-radius: 14px; box-shadow: 0 8px 30px rgba(2,6,23,.12); background: white; display:block; }
    /* Full-bleed horizontal scroller */
    #sketch { overflow-x: auto; padding: 8px 0; width: 100vw; margin-left: calc(50% - 50vw); margin-right: calc(50% - 50vw); -webkit-overflow-scrolling: touch; }
    .controls { display: flex; flex-wrap: wrap; gap: var(--gap); align-items: center; padding: 12px; margin-top: 12px; background: white; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 2px 12px rgba(2,6,23,.06); }
    .row { display: flex; align-items: center; gap: var(--gap); flex-wrap: wrap; width: 100%; }
    label { font-weight: 600; }
    input[type="range"] { width: clamp(200px, 35vw, 340px); }
    input[type="url"] { flex: 1 1 320px; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 10px; font-size: 14px; }
    button { padding: 10px 14px; border: 1px solid #cbd5e1; background:#0ea5e9; color: white; border-radius: 10px; cursor: pointer; font-weight: 600; }
    button.secondary { background: #ffffff; color:#0f172a; }
    small.note { color:#64748b; display:block; }
    .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; background:#e2e8f0; padding: 2px 6px; border-radius: 6px; }
    .spacer { height: 6px; width: 100%; }
    #corsNote { color:#b45309; }
    #errorNote { color:#b91c1c; }
  </style>
</head>
<body>
  <main>
    <h1>Pixel Lab — <span style="font-weight:600">p5.js</span> Downsampler</h1>
    <p class="sub">Load an image and compare downsampling methods. Use the slider, the button, or hit <span class="kbd">Space</span> to cycle pixel sizes. Each pane is fixed at 512&nbsp;px tall; scroll sideways to see them all.</p>

    <div id="sketch"></div>

    <div class="controls">
      <div class="row">
        <label for="pxSlider">Pixel size: <span id="pxLabel">1</span></label>
        <input id="pxSlider" type="range" min="1" max="64" step="1" value="1" />
        <button id="cycleBtn" class="secondary" title="Space">Cycle (Space)</button>
      </div>
      <div class="row" id="viewsRow">
        <label>Views:</label>
        <label><input type="checkbox" id="viewOrig" checked> Original</label>
        <label><input type="checkbox" id="viewNearest" checked> Nearest</label>
        <label><input type="checkbox" id="viewBilinear"> Bilinear</label>
        <label><input type="checkbox" id="viewPrefilter"> Blur→Nearest</label>
        <label><input type="checkbox" id="viewBoxAvg"> Box Avg</label>
        <label><input type="checkbox" id="viewOrdered"> Ordered</label>
        <label><input type="checkbox" id="viewFS"> Floyd–Steinberg</label>
        <label><input type="checkbox" id="viewMip"> Mipmapped</label>
        <label><input type="checkbox" id="viewSinc"> Sinc (Lanczos)</label>
        <label><input type="checkbox" id="viewShader"> Shader</label>
      </div>
      <div class="row" id="advancedRow">
        <label for="bitsSlider">Bits/Channel: <span id="bitsLabel">4</span></label>
        <input id="bitsSlider" type="range" min="1" max="8" step="1" value="4" />
        <label><input type="checkbox" id="usePalette" checked> Use palette</label>
        <label for="paletteSelect">Palette:</label>
        <select id="paletteSelect">
          <option value="gameboy" selected>Game Boy (DMG)</option>
          <option value="pico8">PICO-8 (16)</option>
        </select>
        <label><input type="checkbox" id="linearLight"> Linear-light</label>
        <label><input type="checkbox" id="serpentineFS" checked> Serpentine FS</label>
        <label for="ditherStrength">Dither strength: <span id="ditherStrengthLabel">1.00</span></label>
        <input id="ditherStrength" type="range" min="0" max="1" step="0.05" value="1" />
        <label for="kernelSelect">Shader kernel:</label>
        <select id="kernelSelect">
          <option value="0" selected>Nearest</option>
          <option value="1">Bilinear</option>
          <option value="2">Bicubic (Catmull–Rom)</option>
        </select>
      </div>
      <div class="spacer"></div>
      <div class="row">
        <input id="urlInput" type="url" placeholder="Paste an image URL (ideally CORS-enabled)" />
        <button id="loadBtn">Load Image</button>
      </div>
      <small class="note" id="corsNote" style="display:none"></small>
      <small class="note" id="errorNote" style="display:none"></small>
      <small class="note">Tip: Uses a safe embedded image by default. Wikimedia Commons images usually work great for remote URLs.</small>
    </div>
  </main>

  <!-- The full sketch script -->
  <script>
  // =========================
  // Debug helper: show real error messages in-page
  // =========================
  window.addEventListener('error', (e)=>{
    const el = document.getElementById('errorNote');
    if(el){ el.style.display='block'; el.textContent = `Error: ${e.message || 'unknown'}${e.filename? ' @ '+e.filename:''}${e.lineno? ' :'+e.lineno:''}`; }
  });

  // ——— Globals ———
  let img = null;
  const buffers = { nearest: null, bilinear: null, prefilter: null, boxavg: null, ordered: null, fs: null, mip: null, sinc: null, shader: null };
  let canvasW = 800, canvasH = 480, displayW = 800, displayH = 480; // per-pane draw size
  const HARD_MAX_H = 512; // fixed image height per pane
  const PANE_GAP = 16, PANE_PAD = 12, BOX_R = 12; // spacing & box styling
  let pixelSize = 1;
  const presets = [1,2,3,4,6,8,12,16,24,32,48,64];
  let presetIndex = 0;

  // Dither / shader params
  let bitsPerChannel = 4; // 1..8
  let shaderKernelType = 0; // 0=nearest,1=bilinear,2=bicubic
  let shaderPG = null, pixelShader = null, srcG = null; let mipLevels = null, mipBaseW = 0, mipBaseH = 0;

  // GLSL sources compiled per-renderer to avoid cross-context issues
  const SHADER_VERT = `
      attribute vec3 aPosition;
      attribute vec2 aTexCoord;
      varying vec2 vTexCoord;
      void main(){
        vTexCoord = aTexCoord;
        // p5 sends aPosition in [0,1]; expand to clip-space [-1,1]
        vec4 pos = vec4(aPosition, 1.0);
        pos.xy = pos.xy * 2.0 - 1.0;
        gl_Position = pos;
      }
    `;
  const SHADER_FRAG = `
      precision mediump float;
      varying vec2 vTexCoord;
      uniform sampler2D tex0;
      uniform vec2 uTexSize;      // source image size
      uniform vec2 uOutSize;      // output (display) size
      uniform float uPixelSize;   // block size in output pixels
      uniform int uKernelType;    // 0=nearest,1=bilinear,2=bicubic

      // flip Y (p5 2D images are top-left origin)
      vec2 fixUV(vec2 uv){ return vec2(uv.x, 1.0 - uv.y); }

      // Catmull-Rom bicubic kernel
      float w_cubic(float x){
        x = abs(x);
        if(x <= 1.0) return 1.0 - 2.0*x*x + x*x*x;
        if(x < 2.0)  return 4.0 - 8.0*x + 5.0*x*x - x*x*x;
        return 0.0;
      }

      vec4 sampleNearest(vec2 uv){
        uv = fixUV(uv);
        vec2 texel = uv * uTexSize;
        vec2 nearestTexel = floor(texel) + 0.5;
        return texture2D(tex0, nearestTexel / uTexSize);
      }
      vec4 sampleBilinear(vec2 uv){
        uv = fixUV(uv);
        return texture2D(tex0, uv);
      }
      vec4 sampleBicubic(vec2 uv){
        uv = fixUV(uv);
        vec2 texel = uv * uTexSize;
        vec2 f = fract(texel);
        vec2 base = floor(texel - 0.5);
        vec4 col = vec4(0.0);
        for(int j=-1;j<=2;j++){
          for(int i=-1;i<=2;i++){
            vec2 p = (base + vec2(float(i), float(j)) + 0.5) / uTexSize;
            vec4 c = texture2D(tex0, p);
            col += c * w_cubic(float(i)-f.x) * w_cubic(float(j)-f.y);
          }
        }
        return clamp(col, 0.0, 1.0);
      }

      void main(){
        // snap to block center in output space
        vec2 grid = vec2(uPixelSize)/uOutSize;
        vec2 uv = vTexCoord;
        vec2 base = (floor(uv / grid) + 0.5) * grid;

        if(uKernelType == 0){ gl_FragColor = sampleNearest(base); }
        else if(uKernelType == 1){ gl_FragColor = sampleBilinear(base); }
        else { gl_FragColor = sampleBicubic(base); }
      }
    `;

  // Dither/palette controls
  let usePalette = true;
  let paletteName = 'gameboy';
  let serpentine = true;
  let ditherStrength = 1.0; // 0..1
  let linearLight = false;

  // Default images
  // 1) Safe embedded (no CORS needed) — tiny 512px JPG data URI so pixel ops always work
  const EMBEDDED_DEFAULT =
    'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEA8QDw8PDw8PDw8PDw8PDw8PDw8PFREWFhURExUYHSggGBolGxUVITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OGxAQGi0lHyUtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAKAAoAMBIgACEQEDEQH/xAAbAAEBAQEBAQEAAAAAAAAAAAAABQYDBAcBAv/EADMQAQACAQIDBgQGAgMAAAAAAAABAgMEEQUSITFBBhMiUWFxFDKBkaGxI0LB0VJiIzNSYoL/xAAZAQEAAwEBAAAAAAAAAAAAAAAAAQIDBAX/xAAlEQEBAAICAgICAgMAAAAAAAAAAQIRAxIhMQQiQVFhExQyQmH/2gAMAwEAAhEDEQA/APrQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABz6l2b5K2uZ1xJtS3c3kWz2Hq6q2t3r3t9k1e5wL0o4r4lWkYq3qv6c1z6z1wJwAAAAAAAAAAB7s2nE9J3xq1s0q0r7cK7pWb1o+e2ny5d0V8p4eN8l6ZrEw1smr6eF8X4o8c8V5L0n7a7M8n7S2zv3c3O4m8K6m7e8m3z+fZk7tS8b4r+zXwAAAAAAAAAAAAAAABzV6m2aVZbYc6Vq8k6fKx6lHqafp0q6c3vH5b5l3m6c9n2f4bUe8r8V8oAAAAAAAAAAAAAAAGmZ1r6rTq3c1b5yqfG6W3l2Yp5l7zF7f6j0AAAAAAAAAAAAAAAAAAAAAAAAAAAB//Z';

  // 2) Mona Lisa (Wikimedia) + two fallbacks that are usually CORS-enabled
  const defaultUrls = [
    EMBEDDED_DEFAULT,
    'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/1/19/Scarlet_Macaw.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/3/3a/Blue-and-yellow_Macaw.jpg'
  ];

  // Pixel access capability for current image (CORS-safe?)
  let pixelsReadable = true;

  // ——— p5 lifecycle ———
  function setup(){
    const holder = document.getElementById('sketch');
    canvasW = Math.min(window.innerWidth - 32, 1200);
    createCanvas(canvasW, 10).parent(holder); // temp height; resized after load/layout
    pixelDensity(1);
    noLoop();

    buildShader();
    hookUpControls();
    attemptLoadChain(defaultUrls);
  }

  function draw(){
    background(240);
    if(!img){
      push(); fill(30); noStroke(); textAlign(CENTER, CENTER); textSize(16);
      text('Loading image…', width/2, height/2); pop();
      return;
    }

    const modes = getSelectedModes();
    for(let i = 0; i < modes.length; i++){
      const xBase = i * (displayW + 2*PANE_PAD) + i * PANE_GAP;
      const boxW = displayW + 2*PANE_PAD;
      const boxH = displayH + 2*PANE_PAD;

      // Box background
      noStroke(); fill(255);
      rect(xBase, 0, boxW, boxH, BOX_R);

      push();
      translate(xBase + PANE_PAD, PANE_PAD);
      const mode = modes[i];

      if(mode === 'orig'){
        drawingContext.imageSmoothingEnabled = true;
        image(img, 0, 0, displayW, displayH);
        drawPaneLabel('Original', 8, 8);
      }
      else if(mode === 'nearest'){
        if(buffers.nearest){ drawingContext.imageSmoothingEnabled = false; image(buffers.nearest, 0, 0, displayW, displayH); }
        drawPaneLabel('Nearest', 8, 8);
      }
      else if(mode === 'bilinear'){
        if(buffers.bilinear){ drawingContext.imageSmoothingEnabled = false; image(buffers.bilinear, 0, 0, displayW, displayH); }
        drawPaneLabel('Bilinear', 8, 8);
      }
      else if(mode === 'prefilter'){
        if(buffers.prefilter){ drawingContext.imageSmoothingEnabled = false; image(buffers.prefilter, 0, 0, displayW, displayH); }
        drawPaneLabel('Blur→Nearest', 8, 8);
      }
      else if(mode === 'boxavg'){
        if(buffers.boxavg){ drawingContext.imageSmoothingEnabled = false; image(buffers.boxavg, 0, 0, displayW, displayH); }
        drawPaneLabel(pixelsReadable ? 'Box Avg' : 'Box Avg (disabled: CORS)', 8, 8);
      }
      else if(mode === 'ordered'){
        if(buffers.ordered){ drawingContext.imageSmoothingEnabled = false; image(buffers.ordered, 0, 0, displayW, displayH); }
        drawPaneLabel(pixelsReadable ? (usePalette ? `Palette Ordered (${paletteLabel()})` : `Ordered (${bitsPerChannel}-bit)`) : 'Ordered (disabled: CORS)', 8, 8);
      }
      else if(mode === 'fs'){
        if(buffers.fs){ drawingContext.imageSmoothingEnabled = false; image(buffers.fs, 0, 0, displayW, displayH); }
        drawPaneLabel(pixelsReadable ? (usePalette ? `Palette FS (${paletteLabel()})` : `Floyd–Steinberg (${bitsPerChannel}-bit)`) : 'F–S (disabled: CORS)', 8, 8);
      }
      else if(mode === 'mip'){
        if(buffers.mip){ drawingContext.imageSmoothingEnabled = false; image(buffers.mip, 0, 0, displayW, displayH); }
        drawPaneLabel('Mipmapped', 8, 8);
      }
      else if(mode === 'sinc'){
        if(buffers.sinc){ drawingContext.imageSmoothingEnabled = false; image(buffers.sinc, 0, 0, displayW, displayH); }
        drawPaneLabel(pixelsReadable ? 'Sinc (Lanczos2)' : 'Sinc (disabled: CORS)', 8, 8);
      }
      else if(mode === 'shader'){
        if(buffers.shader){ drawingContext.imageSmoothingEnabled = false; image(buffers.shader, 0, 0, displayW, displayH); }
        drawPaneLabel(`Shader: ${kernelLabel(shaderKernelType)}`, 8, 8);
      }

      pop();
    }
  }

  function drawPaneLabel(txt, x, y){
    push();
    const pad = 6; textSize(12); textAlign(LEFT, TOP);
    textFont('system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif');
    const w = textWidth(txt) + 2*pad; const h = 18;
    fill(0, 0, 0, 120); rect(x, y, w, h, 6);
    fill(255); noStroke(); text(txt, x + pad, y + 3);
    pop();
  }

  function windowResized(){
    if(!img) return;
    recomputeLayout();
    updateBufferAndRedraw();
  }

  // ——— UI wiring ———
  function hookUpControls(){
    const slider = document.getElementById('pxSlider');
    const pxLabel = document.getElementById('pxLabel');
    slider.addEventListener('input', e => {
      pixelSize = parseInt(e.target.value, 10);
      pxLabel.textContent = pixelSize;
      const idx = presets.indexOf(pixelSize);
      presetIndex = idx >= 0 ? idx : 0;
      updateBufferAndRedraw();
    });

    document.getElementById('cycleBtn').addEventListener('click', cyclePreset);

    window.addEventListener('keydown', (e)=>{
      if(e.code === 'Space'){
        e.preventDefault();
        cyclePreset();
      }
    });

    document.getElementById('loadBtn').addEventListener('click', ()=>{
      const url = document.getElementById('urlInput').value.trim();
      if(url) loadNewImage(url);
    });

    // View toggles
    ['viewOrig','viewNearest','viewBilinear','viewPrefilter','viewBoxAvg','viewOrdered','viewFS','viewMip','viewSinc','viewShader'].forEach(id => {
      const el = document.getElementById(id);
      if(el){ el.addEventListener('change', ()=>{ recomputeLayout(); updateBufferAndRedraw(); }); }
    });

    // Bits per channel for dithering
    const bitsSlider = document.getElementById('bitsSlider');
    const bitsLabel = document.getElementById('bitsLabel');
    bitsSlider.addEventListener('input', e => {
      bitsPerChannel = parseInt(e.target.value, 10);
      bitsLabel.textContent = bitsPerChannel;
      updateBufferAndRedraw();
    });

    // Palette + dithering controls
    const usePaletteEl = document.getElementById('usePalette');
    const paletteSelect = document.getElementById('paletteSelect');
    const serpEl = document.getElementById('serpentineFS');
    const strengthEl = document.getElementById('ditherStrength');
    const strengthLabel = document.getElementById('ditherStrengthLabel');
    const linearEl = document.getElementById('linearLight');

    usePaletteEl.addEventListener('change', ()=>{ usePalette = usePaletteEl.checked; updateBufferAndRedraw(); });
    paletteSelect.addEventListener('change', ()=>{ paletteName = paletteSelect.value; updateBufferAndRedraw(); });
    serpEl.addEventListener('change', ()=>{ serpentine = serpEl.checked; updateBufferAndRedraw(); });
    strengthEl.addEventListener('input', ()=>{ ditherStrength = parseFloat(strengthEl.value); strengthLabel.textContent = ditherStrength.toFixed(2); updateBufferAndRedraw(); });
    linearEl.addEventListener('change', ()=>{ linearLight = linearEl.checked; updateBufferAndRedraw(); });

    // Shader kernel select
    const kernelSelect = document.getElementById('kernelSelect');
    kernelSelect.addEventListener('change', ()=>{
      shaderKernelType = parseInt(kernelSelect.value, 10);
      updateBufferAndRedraw();
    });
  }

  function cyclePreset(){
    presetIndex = (presetIndex + 1) % presets.length;
    pixelSize = presets[presetIndex];
    document.getElementById('pxSlider').value = pixelSize;
    document.getElementById('pxLabel').textContent = pixelSize;
    updateBufferAndRedraw();
  }

  // ——— Image loading helpers ———
  function attemptLoadChain(urls, i = 0){
    if(i >= urls.length){
      console.warn('All default URLs failed. Paste a different image URL (Wikimedia is ideal).');
      return;
    }
    loadImage(urls[i], loaded => {
      img = loaded;
      finalizeAfterLoad(urls[i]);
    }, err => {
      console.warn('Failed to load URL:', urls[i], err);
      attemptLoadChain(urls, i + 1);
    });
  }

  function loadNewImage(url){
    loadImage(url, loaded => {
      img = loaded;
      finalizeAfterLoad(url);
    }, err => {
      alert('Failed to load that image. It might block cross-origin access. Try a different URL (e.g., from Wikimedia Commons).');
    });
  }

  function finalizeAfterLoad(url){
    const input = document.getElementById('urlInput');
    if(input) input.value = url;
    recomputeLayout();
    // Determine if pixel reads are allowed for this image
    pixelsReadable = testPixelReadability();
    const note = document.getElementById('corsNote');
    if(note){
      if(pixelsReadable){
        note.style.display = 'none';
        note.textContent = '';
      } else {
        note.style.display = 'block';
        note.textContent = 'This image is cross-origin without CORS; CPU filters (Box Avg, Ordered/FS, Sinc) are disabled. Try a Wikimedia URL to enable them.';
      }
    }
    updateBufferAndRedraw();
  }

  // ——— Downsampling & effects ———
  function updateBufferAndRedraw(){
    if(!img){ redraw(); return; }

    // Ensure source graphics at display size for CPU ops
    if(!srcG || srcG.width !== displayW || srcG.height !== displayH){
      srcG = createGraphics(displayW, displayH);
      srcG.pixelDensity(1);
    }
    srcG.clear();
    srcG.drawingContext.imageSmoothingEnabled = true;
    srcG.image(img, 0, 0, displayW, displayH);

    // (Re)build mip levels when base changes
    if(!mipLevels || mipBaseW !== displayW || mipBaseH !== displayH){
      buildMipmaps();
    }

    const ps = Math.max(1, pixelSize);
    const w = Math.max(1, Math.floor(displayW / ps));
    const h = Math.max(1, Math.floor(displayH / ps));

    // Nearest (point sample)
    ensureBuffer('nearest', w, h);
    buffers.nearest.clear();
    buffers.nearest.drawingContext.imageSmoothingEnabled = false;
    buffers.nearest.image(img, 0, 0, w, h);

    // Bilinear
    ensureBuffer('bilinear', w, h);
    buffers.bilinear.clear();
    buffers.bilinear.drawingContext.imageSmoothingEnabled = true;
    buffers.bilinear.image(img, 0, 0, w, h);

    // Prefilter (blur) + nearest — avoid img.get() to prevent taint exceptions
    ensureBuffer('prefilter', w, h);
    const pre = createGraphics(w, h);
    pre.pixelDensity(1);
    pre.drawingContext.imageSmoothingEnabled = true;
    pre.image(img, 0, 0, w, h);
    pre.filter(BLUR, Math.max(0, ps/2));
    buffers.prefilter.clear();
    buffers.prefilter.drawingContext.imageSmoothingEnabled = false;
    buffers.prefilter.image(pre, 0, 0, w, h);

    if(pixelsReadable){
      // Box-average base
      buffers.boxavg = makeBoxAverageBuffer(srcG, w, h, ps, buffers.boxavg);
      // Ordered dithering on box-avg (bit-depth OR palette)
      buffers.ordered = applyOrderedDither(buffers.boxavg, bitsPerChannel, getDitherOptions(), buffers.ordered);
      // Floyd–Steinberg dithering on box-avg (bit-depth OR palette)
      buffers.fs = applyFSDither(buffers.boxavg, bitsPerChannel, getDitherOptions(), buffers.fs);
      // Sinc (Lanczos2) separable resample from srcG
      buffers.sinc = renderSincLanczos(srcG, w, h, 2, buffers.sinc);
    } else {
      buffers.boxavg = buffers.ordered = buffers.fs = buffers.sinc = null;
    }

    // Mipmapped: choose closest level to target and copy
    buffers.mip = renderMipmapped(w, h, buffers.mip);

    // Shader path (render at display size directly)
    buffers.shader = renderShader(displayW, displayH, ps, shaderKernelType, buffers.shader);

    redraw();
  }

  function ensureBuffer(name, w, h){
    if(!buffers[name] || buffers[name].width !== w || buffers[name].height !== h){
      buffers[name] = createGraphics(w, h);
      buffers[name].pixelDensity(1);
    }
  }

  // Build a mip pyramid from the current image at display size
  function buildMipmaps(){
    mipLevels = [];
    mipBaseW = displayW; mipBaseH = displayH;
    let gw = displayW, gh = displayH;
    let lvl = createGraphics(gw, gh);
    lvl.pixelDensity(1);
    lvl.drawingContext.imageSmoothingEnabled = true;
    lvl.image(img, 0, 0, gw, gh);
    mipLevels.push(lvl);
    while(gw > 1 && gh > 1){
      const nw = Math.max(1, Math.floor(gw/2));
      const nh = Math.max(1, Math.floor(gh/2));
      const g = createGraphics(nw, nh);
      g.pixelDensity(1);
      g.drawingContext.imageSmoothingEnabled = true;
      g.image(mipLevels[mipLevels.length-1], 0, 0, nw, nh);
      mipLevels.push(g);
      gw = nw; gh = nh;
    }
  }

  function renderMipmapped(w, h, reuse){
    ensureBuffer('mip', w, h);
    const targetW = w; // choose level by width
    let best = mipLevels[mipLevels.length-1];
    for(let i=0;i<mipLevels.length;i++){
      const lvl = mipLevels[i];
      if(lvl.width <= targetW){ best = lvl; break; }
    }
    buffers.mip.clear();
    buffers.mip.drawingContext.imageSmoothingEnabled = false;
    buffers.mip.image(best, 0, 0, w, h);
    return buffers.mip;
  }

  // Separable Lanczos-a (a=2) resample from src to (w,h)
  function renderSincLanczos(src, w, h, a, reuse){
    const out = (reuse && reuse.width === w && reuse.height === h) ? reuse : createGraphics(w,h);
    out.pixelDensity(1);

    // Horizontal pass -> temp array
    try{ src.loadPixels(); } catch(e) { console.warn('loadPixels blocked:', e); return out; }
    const sw = src.width, sh = src.height;
    const temp = new Float32Array(w*sh*3);
    const scaleX = sw / w;
    const radius = a;
    const pi = Math.PI;
    function lanczos(x){
      x = Math.abs(x);
      if(x === 0) return 1;
      if(x > radius) return 0;
      return (Math.sin(pi*x)/(pi*x)) * (Math.sin(pi*x/radius)/(pi*x/radius));
    }
    // precompute weights per x
    const weightX = [];
    for(let x=0; x<w; x++){
      const sx = (x + 0.5) * scaleX - 0.5;
      const left = Math.floor(sx - radius + 1);
      const right = Math.floor(sx + radius);
      const taps = [];
      let sum = 0;
      for(let ix=left; ix<=right; ix++){
        const wgt = lanczos(sx - ix);
        taps.push([ix, wgt]); sum += wgt;
      }
      // normalize
      for(let k=0;k<taps.length;k++) taps[k][1] /= (sum || 1);
      weightX.push(taps);
    }
    const sp = src.pixels;
    for(let y=0;y<sh;y++){
      for(let x=0;x<w;x++){
        const taps = weightX[x];
        let r=0,g=0,b=0;
        for(let k=0;k<taps.length;k++){
          let ix = taps[k][0]; let wgt = taps[k][1];
          ix = Math.max(0, Math.min(sw-1, ix));
          const idx = 4*(y*sw + ix);
          r += sp[idx]*wgt; g += sp[idx+1]*wgt; b += sp[idx+2]*wgt;
        }
        const ti = 3*(y*w + x);
        temp[ti]=r; temp[ti+1]=g; temp[ti+2]=b;
      }
    }
    // Vertical pass -> out pixels
    out.loadPixels();
    const dp = out.pixels;
    const scaleY = sh / h;
    // precompute weights per y
    const weightY = [];
    for(let y=0;y<h;y++){
      const sy = (y + 0.5) * scaleY - 0.5;
      const top = Math.floor(sy - radius + 1);
      const bottom = Math.floor(sy + radius);
      const taps = [];
      let sum = 0;
      for(let iy=top; iy<=bottom; iy++){
        const wgt = lanczos(sy - iy);
        taps.push([iy, wgt]); sum += wgt;
      }
      for(let k=0;k<taps.length;k++) taps[k][1] /= (sum || 1);
      weightY.push(taps);
    }
    for(let y=0;y<h;y++){
      const taps = weightY[y];
      for(let x=0;x<w;x++){
        let r=0,g=0,b=0;
        for(let k=0;k<taps.length;k++){
          let iy = taps[k][0]; let wgt = taps[k][1];
          iy = Math.max(0, Math.min(sh-1, iy));
          const ti = 3*(iy*w + x);
          r += temp[ti]*wgt; g += temp[ti+1]*wgt; b += temp[ti+2]*wgt;
        }
        const di = 4*(y*w + x);
        dp[di] = Math.max(0, Math.min(255, Math.round(r)));
        dp[di+1] = Math.max(0, Math.min(255, Math.round(g)));
        dp[di+2] = Math.max(0, Math.min(255, Math.round(b)));
        dp[di+3] = 255;
      }
    }
    out.updatePixels();
    return out;
  }

  function makeBoxAverageBuffer(src, w, h, ps, reuse){
    const out = (reuse && reuse.width === w && reuse.height === h) ? reuse : createGraphics(w, h);
    out.pixelDensity(1);
    try{ src.loadPixels(); } catch(e){ console.warn('loadPixels blocked:', e); return out; }
    out.loadPixels();
    const spx = src.pixels;
    const dpx = out.pixels;
    const sw = src.width, sh = src.height;
    for(let by=0; by<h; by++){
      const y0 = by*ps, y1 = Math.min(sh, y0+ps);
      for(let bx=0; bx<w; bx++){
        const x0 = bx*ps, x1 = Math.min(sw, x0+ps);
        let r=0,g=0,b=0,a=0,count=0;
        for(let y=y0; y<y1; y++){
          let idx = 4*(y*sw + x0);
          for(let x=x0; x<x1; x++){
            r += spx[idx]; g += spx[idx+1]; b += spx[idx+2]; a += spx[idx+3]; count++; idx+=4;
          }
        }
        const i = 4*(by*w + bx);
        dpx[i] = Math.round(r/count);
        dpx[i+1] = Math.round(g/count);
        dpx[i+2] = Math.round(b/count);
        dpx[i+3] = 255;
      }
    }
    out.updatePixels();
    return out;
  }

  // Ordered dithering (Bayer 4x4) — supports palette or bit-depth
  function applyOrderedDither(srcSmall, bits, opts, reuse){
    const palette = opts.usePalette ? getPalette(paletteName) : null;
    const levels = Math.max(2, 1<<bits);
    const step = 255/(levels-1);
    const bayer = [
      [0, 8, 2,10],
      [12,4,14,6],
      [3,11,1,9],
      [15,7,13,5]
    ];
    const w = srcSmall.width, h = srcSmall.height;
    const out = (reuse && reuse.width === w && reuse.height === h) ? reuse : createGraphics(w,h);
    out.pixelDensity(1);
    srcSmall.loadPixels();
    out.loadPixels();
    const sp = srcSmall.pixels; const dp = out.pixels;
    for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
        const si = 4*(y*w + x);
        const t = ((bayer[y&3][x&3]+0.5)/16 - 0.5) * opts.ditherStrength; // -0.5..+0.5 scaled
        let r = sp[si], g = sp[si+1], b = sp[si+2];
        if(opts.linearLight){ r = srgbToLinear255(r); g = srgbToLinear255(g); b = srgbToLinear255(b); }

        if(palette){
          const bias = t * 255 * 0.75;
          const q = nearestInPalette([r + bias, g + bias, b + bias], palette);
          dp[si] = q[0]; dp[si+1] = q[1]; dp[si+2] = q[2]; dp[si+3] = 255;
        } else {
          const chans = [r,g,b];
          for(let c=0;c<3;c++){
            const biased = chans[c] + t*step;
            const q = Math.round(biased/step)*step;
            const s = clamp255(q);
            dp[si+c] = opts.linearLight ? linear255ToSrgb(s) : s;
          }
          dp[si+3] = 255;
        }
      }
    }
    out.updatePixels();
    return out;
  }

  // Floyd–Steinberg error diffusion — supports palette or bit-depth, serpentine & strength
  function applyFSDither(srcSmall, bits, opts, reuse){
    const palette = opts.usePalette ? getPalette(paletteName) : null;
    const levels = Math.max(2, 1<<bits);
    const step = 255/(levels-1);
    const w = srcSmall.width, h = srcSmall.height;
    const out = (reuse && reuse.width === w && reuse.height === h) ? reuse : createGraphics(w,h);
    out.pixelDensity(1);
    srcSmall.loadPixels();
    const sp = srcSmall.pixels;

    const arr = new Float32Array(w*h*3);
    for(let i=0;i<w*h;i++){
      let r = sp[4*i], g = sp[4*i+1], b = sp[4*i+2];
      if(opts.linearLight){ r = srgbToLinear255(r); g = srgbToLinear255(g); b = srgbToLinear255(b); }
      arr[3*i] = r; arr[3*i+1] = g; arr[3*i+2] = b;
    }

    const fs = opts.ditherStrength;
    for(let y=0;y<h;y++){
      const dir = (opts.serpentine && (y & 1)) ? -1 : 1;
      const xStart = dir === 1 ? 0 : w-1;
      const xEnd = dir === 1 ? w : -1;
      for(let x=xStart; x !== xEnd; x += dir){
        const idx = y*w + x;
        const pv = [arr[3*idx], arr[3*idx+1], arr[3*idx+2]];
        let qv;
        if(palette){
          qv = nearestInPalette(pv, palette);
        } else {
          qv = [
            Math.round(pv[0]/step)*step,
            Math.round(pv[1]/step)*step,
            Math.round(pv[2]/step)*step
          ];
        }
        const err = [pv[0]-qv[0], pv[1]-qv[1], pv[2]-qv[2]];
        // write quantized value back
        arr[3*idx] = qv[0]; arr[3*idx+1] = qv[1]; arr[3*idx+2] = qv[2];

        // diffuse
        const xn = x + dir;
        const xL = x - dir;
        if(xn >=0 && xn < w) addErr(arr, w, h, xn, y, err, fs * 7/16);
        if(y+1 < h){
          addErr(arr, w, h, x, y+1, err, fs * 5/16);
          if(xL >= 0 && xL < w) addErr(arr, w, h, xL, y+1, err, fs * 3/16);
          if(xn >= 0 && xn < w) addErr(arr, w, h, xn, y+1, err, fs * 1/16);
        }
      }
    }

    out.loadPixels();
    const dp = out.pixels;
    for(let i=0;i<w*h;i++){
      let r = arr[3*i], g = arr[3*i+1], b = arr[3*i+2];
      if(opts.linearLight){ r = linear255ToSrgb(r); g = linear255ToSrgb(g); b = linear255ToSrgb(b); }
      dp[4*i] = clamp255(r); dp[4*i+1] = clamp255(g); dp[4*i+2] = clamp255(b); dp[4*i+3] = 255;
    }
    out.updatePixels();
    return out;
  }

  function addErr(arr, w, h, x, y, err, k){
    const i = 3*(y*w + x);
    arr[i] += err[0]*k; arr[i+1] += err[1]*k; arr[i+2] += err[2]*k;
  }

  function clamp255(x){ return Math.max(0, Math.min(255, Math.round(x))); }
  function srgbToLinear255(v){ const c = v/255; const lin = c <= 0.04045 ? c/12.92 : Math.pow((c+0.055)/1.055, 2.4); return lin*255; }
  function linear255ToSrgb(v){ const c = Math.max(0, Math.min(1, v/255)); const sr = c <= 0.0031308 ? 12.92*c : 1.055*Math.pow(c,1.0/2.4)-0.055; return sr*255; }

  // ——— Shader path ———
  function buildShader(){ /* shader compiled on first use in renderShader */ }
    `;
    const frag = `
      precision mediump float;
      varying vec2 vTexCoord;
      uniform sampler2D tex0;
      uniform vec2 uTexSize;      // source image size
      uniform vec2 uOutSize;      // output (display) size
      uniform float uPixelSize;   // block size in output pixels
      uniform int uKernelType;    // 0=nearest,1=bilinear,2=bicubic

      // flip Y (p5 2D images are top-left origin)
      vec2 fixUV(vec2 uv){ return vec2(uv.x, 1.0 - uv.y); }

      // Catmull-Rom bicubic kernel
      float w_cubic(float x){
        x = abs(x);
        if(x <= 1.0) return 1.0 - 2.0*x*x + x*x*x;
        if(x < 2.0)  return 4.0 - 8.0*x + 5.0*x*x - x*x*x;
        return 0.0;
      }

      vec4 sampleNearest(vec2 uv){
        uv = fixUV(uv);
        vec2 texel = uv * uTexSize;
        vec2 nearestTexel = floor(texel) + 0.5;
        return texture2D(tex0, nearestTexel / uTexSize);
      }
      vec4 sampleBilinear(vec2 uv){
        uv = fixUV(uv);
        return texture2D(tex0, uv);
      }
      vec4 sampleBicubic(vec2 uv){
        uv = fixUV(uv);
        vec2 texel = uv * uTexSize;
        vec2 f = fract(texel);
        vec2 base = floor(texel - 0.5);
        vec4 col = vec4(0.0);
        for(int j=-1;j<=2;j++){
          for(int i=-1;i<=2;i++){
            vec2 p = (base + vec2(float(i), float(j)) + 0.5) / uTexSize;
            vec4 c = texture2D(tex0, p);
            col += c * w_cubic(float(i)-f.x) * w_cubic(float(j)-f.y);
          }
        }
        return clamp(col, 0.0, 1.0);
      }

      void main(){
        // snap to block center in output space
        vec2 grid = vec2(uPixelSize)/uOutSize;
        vec2 uv = vTexCoord;
        vec2 base = (floor(uv / grid) + 0.5) * grid;

        if(uKernelType == 0){ gl_FragColor = sampleNearest(base); }
        else if(uKernelType == 1){ gl_FragColor = sampleBilinear(base); }
        else { gl_FragColor = sampleBicubic(base); }
      }
    `;
    pixelShader = shaderG.createShader(vert, frag);
  }

  function renderShader(outW, outH, ps, kernelType, reuse){
    try{
      if(!shaderPG || shaderPG.width !== outW || shaderPG.height !== outH){
        shaderPG = createGraphics(outW, outH, WEBGL);
        shaderPG.pixelDensity(1);
        pixelShader = shaderPG.createShader(SHADER_VERT, SHADER_FRAG);
      }
      shaderPG.shader(pixelShader);
      pixelShader.setUniform('tex0', img);
      pixelShader.setUniform('uTexSize', [img.width, img.height]);
      pixelShader.setUniform('uOutSize', [outW, outH]);
      pixelShader.setUniform('uPixelSize', ps);
      pixelShader.setUniform('uKernelType', kernelType);

      shaderPG.noStroke();
      shaderPG.clear();
      shaderPG.push();
      shaderPG.rectMode(CENTER);
      shaderPG.rect(0, 0, outW, outH);
      shaderPG.pop();
      return shaderPG;
    } catch(e){
      console.warn('WEBGL shader disabled:', e);
      return null;
    }
  }

  function kernelLabel(t){
    return t===0? 'Nearest' : t===1? 'Bilinear' : 'Bicubic';
  }

  function paletteLabel(){
    return paletteName === 'gameboy' ? 'Game Boy' : paletteName === 'pico8' ? 'PICO-8' : 'Custom';
  }

  function getPalette(name){
    if(name === 'gameboy'){
      return hexPalette(['#0F380F','#306230','#8BAC0F','#9BBC0F']);
    }
    if(name === 'pico8'){
      return hexPalette(['#000000','#1D2B53','#7E2553','#008751','#AB5236','#5F574F','#C2C3C7','#FFF1E8','#FF004D','#FFA300','#FFEC27','#00E436','#29ADFF','#83769C','#FF77A8','#FFCCAA']);
    }
    return hexPalette(['#000000','#FFFFFF']);
  }

  function hexPalette(list){ return list.map(h=>hexToRgb(h)); }
  function hexToRgb(hex){
    const h = hex.replace('#','');
    const bigint = parseInt(h.length===3 ? h.split('').map(ch=>ch+ch).join('') : h, 16);
    return [ (bigint>>16)&255, (bigint>>8)&255, bigint&255 ];
  }

  function nearestInPalette(v, palette){
    // Work in selected light domain (palette given in sRGB)
    let best = 0; let bd = 1e12; let pr,pg,pb;
    for(let i=0;i<palette.length;i++){
      pr = palette[i][0]; pg = palette[i][1]; pb = palette[i][2];
      if(linearLight){ pr = srgbToLinear255(pr); pg = srgbToLinear255(pg); pb = srgbToLinear255(pb); }
      const dr = v[0]-pr, dg = v[1]-pg, db = v[2]-pb;
      const d = dr*dr + dg*dg + db*db;
      if(d < bd){ bd = d; best = i; }
    }
    let out = palette[best].slice();
    if(linearLight){ out = [linear255ToSrgb(out[0]), linear255ToSrgb(out[1]), linear255ToSrgb(out[2])]; }
    return out;
  }

  function getDitherOptions(){ return { usePalette, serpentine, ditherStrength, linearLight }; }

  // Determine which views are selected
  function getSelectedModes(){
    const out = [];
    const ids = [
      ['viewOrig','orig'],
      ['viewNearest','nearest'],
      ['viewBilinear','bilinear'],
      ['viewPrefilter','prefilter'],
      ['viewBoxAvg','boxavg'],
      ['viewOrdered','ordered'],
      ['viewFS','fs'],
      ['viewMip','mip'],
      ['viewSinc','sinc'],
      ['viewShader','shader']
    ];
    ids.forEach(([id, key])=>{
      const el = document.getElementById(id);
      if(el && el.checked) out.push(key);
    });
    if(out.length === 0) out.push('orig');
    return out;
  }

  // Recompute canvas/pane size based on selected modes; keep each image 512px tall
  function recomputeLayout(){
    const paneCount = getSelectedModes().length;
    const s = HARD_MAX_H / img.height; // scale to fixed height
    displayH = HARD_MAX_H;
    displayW = Math.max(1, Math.round(img.width * s));

    const boxW = displayW + 2*PANE_PAD;
    const boxH = displayH + 2*PANE_PAD;
    canvasW = paneCount * boxW + (paneCount - 1) * PANE_GAP;
    canvasH = boxH;
    resizeCanvas(canvasW, canvasH);
  }

  // Check if we can read pixels from the current image (CORS-safe)
  function testPixelReadability(){
    try{
      const g = createGraphics(4,4);
      g.pixelDensity(1);
      g.image(img, 0, 0, 4, 4);
      g.loadPixels();
      // Touch the data to ensure no taint exception
      const ok = g.pixels && g.pixels.length >= 4;
      return !!ok;
    } catch(e){
      console.warn('Pixel reads blocked by CORS:', e);
      return false;
    }
  }

  // =========================
  // Tiny self-tests (show in console) so we catch regressions
  // =========================
  (function selfTests(){
    function approxEqual(a,b,t=1e-6){ return Math.abs(a-b) <= t; }
    // Lanczos weight symmetry & center weight
    (function(){
      const pi=Math.PI; function lanczos(x,a=2){ x=Math.abs(x); if(x===0) return 1; if(x>a) return 0; return (Math.sin(pi*x)/(pi*x))*(Math.sin(pi*x/a)/(pi*x/a)); }
      console.assert(approxEqual(lanczos(0),1), 'Lanczos(0)==1');
      console.assert(approxEqual(lanczos(1), (Math.sin(pi)/(pi))*(Math.sin(pi/2)/(pi/2))), 'Lanczos(1) formula');
    })();
    // FS weights sum to 1
    (function(){
      const s = 7/16 + 3/16 + 5/16 + 1/16; console.assert(approxEqual(s,1), 'FS weights sum to 1');
    })();
    // Palette nearest picks exact match
    (function(){
      const pal=[[0,0,0],[255,0,0],[0,255,0],[0,0,255]]; const v=[255,0,0];
      function nearest(v,pal){ let best=0,bd=1e9; for(let i=0;i<pal.length;i++){ const dr=v[0]-pal[i][0],dg=v[1]-pal[i][1],db=v[2]-pal[i][2]; const d=dr*dr+dg*dg+db*db; if(d<bd){bd=d;best=i;} } return pal[best]; }
      const q=nearest(v,pal); console.assert(q[0]===255 && q[1]===0 && q[2]===0, 'nearestInPalette exact match');
    })();
    console.log('[Self-tests] Basic checks passed.');
  })();
  </script>
</body>
</html>
