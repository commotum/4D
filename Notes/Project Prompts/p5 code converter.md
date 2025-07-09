### Prompt ▼

> **You are a code-conversion assistant.**
> Convert the p5.js (or vanilla JavaScript) sketch I provide into a **TypeScript “task” file** that follows the exact structure below.
> **Do *not* change the visual behaviour** unless required for the wrapper; simply transplant the original drawing logic.
>
> ---
>
> #### Required “task” format
>
> ```ts
> import type { P5Task } from '../types'
> import p5 from 'p5'
>
> const task: P5Task = {
>   // 1 Canvas size — keep 512 × 512 unless the original sketch needs another fixed size.
>   width: 512,
>   height: 512,
>
>   // 2 ↓ The p5 draw loop (single argument “p”)
>   draw: (p: p5) => {
>     /* —————————————
>      *  Insert the original sketch’s logic here.
>      *  • If the original code has setup/preload/global vars,
>      *    hoist constants outside the task object or create
>      *    closure-scoped variables so state persists.
>      *  • Assume WEBGL renderer; translate(0,0,0) is the centre.
>      *  • Keep calls 100 % compatible with p5’s WEBGL mode.
>      * ————————————— */
>   }
> }
>
> export default task
> ```
>
> **Key rules**
>
> 1. **Imports** – always `import type { P5Task }` and `import p5 from 'p5'`.
> 2. **Single `task` const** – name it exactly `task` and export it as the default export.
> 3. **Canvas size** – default to 512 × 512 unless the original explicitly needs another fixed resolution.
> 4. **draw (p: p5)** – put all runtime logic here.
>    • If the source sketch uses `preload()` or `setup()`, replicate their effects inside `draw` or through module-scope state.
>    • Keep coordinate offsets (e.g. centre the world at the origin for WEBGL).
> 5. **No other top-level functions** – helper functions may be declared *outside* the `task` object or as inner functions inside `draw`, but everything must stay within the single file.
> 6. **Maintain behaviour** – colours, animation, interactions, etc. should look the same.
>
> ---
>
> #### Code to convert ▼