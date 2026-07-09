# PARALLAX ENGRAM FIELD — interactive-text system spec
## "the memory looks back at you" · per-glyph MSDF · zero DOM
Produced by the bleeding-edge scour workflow wf_ed4da324-bdd (5 frontier-angle agents + novelty-skeptic + buildability-judge + synthesizer that READ THE REAL SOURCE and corrected the scour). Build v1 against THIS.

## THE CONCEPT (signature wow)
Move the cursor over a wall of black text and it becomes a 3D depth-sorted brain that leans, ripples, and focuses toward the pointer — important memories physically closer to your face. Fuses 4 techniques into ONE grammar: (a) per-glyph W-perspective/clip-z depth, (b) cursor Gaussian z-lift + tilt (diorama parallax), (c) small SDF atlas-UV domain-warp near cursor (liquid-mercury letterforms), (d) SDF-threshold weight + AA-band defocus as the data channel. v2 adds spring-physics compute pass + true 4D w-axis hover punch.

## VERIFIED ENGINE FACTS (from real source — do NOT re-derive)
- info.zw is FREE (packGlyph writes 0,0 at float offsets 14-15). out.clip.z is pinned 0.0 (msdf-text.wgsl.ts line 67) — depth pre-wired, unused.
- PARAMS_FLOATS=16 and ALL 16 lanes claimed (last 4 by live bridge). Cursor uniform requires GROWING the struct to 20 (still 16-byte aligned = 80 bytes). This is the ONE upfront cost.
- Glyph anchors live in PRE-aspect-divide NDC (item.x,item.y). Shader divides pos.x/=max(aspect,1), pos.y*=min(aspect,1). Cursor MUST be written in the SAME pre-divide space: cursor_pre.x = ndcX*max(aspect,1), cursor_pre.y = ndcY/min(aspect,1) (inverse of pickAt lines 187-193) or the field skews.

## DATA BINDING (discipline test — Math.random visibly breaks each)
| visual | real fact | source field |
| depth z (front↔back) | trust/centrality | ObservatoryNode.isCenter + inverse suppression |
| stroke weight (SDF threshold bias) | FSRS retention (live-recomputed) | retention (from stability+lastAccessed) |
| DoF blur (AA-band width) | trust % / contradiction | trust |
| brightness (bloom eligibility) | salience/activation | activation |
| cursor lift/ripple/focus | user's attention (honest: the lens, not a data fact) | cursor uniform |
| w-punch on hover | the engram being inspected | pickAt |
| recall shock on click | a real recall event | click → route pick path |
Rule: resting field = data readout; cursor = lens that perturbs it. Nothing moves that isn't a memory fact OR the user's attention.

## BUILD PLAN — v1 (ZERO new render passes)
1. **Cursor uniform (one-time cost):** types.ts PARAMS_FLOATS 16→20, extend PARAM_IDX + WGSL struct Params with cursor_x[16], cursor_y[17], cursor_vx[18], cursor_vy[19] (pre-divide NDC space + smoothed velocity). Engine writes it in the per-frame param block (one writeBuffer). Velocity = smoothed (cursor - prevCursor).
2. **Pack data into info.zw:** in packGlyph, offsets 14-15 become info.z=engramDepth (0..1 trust), info.w=engramWeight (0..1 retention). Add optional {depth?, weight?} to TextLayerItem, thread from ObservatoryNode (trust→depth, retention→weight). Backward-compat: default mid value.
3. **Vertex (vs_text, replace lines 60-67):**
   - resting depth: z = mix(0.55, 0.05, depth) (near=0.05, far=0.55)
   - cursor Gaussian: d=distance(anchor, cursor_pre); R=0.45; w=exp(-d*d/(R*R))
   - z -= w*0.30 (lift toward camera); pos += normalize(cursor-anchor)*w*0.015 (lean)
   - pulse parallax: pos += vec2(sin(time*0.6),cos(time*0.5))*((1-depth)*0.006)*pulse
   - KEEP the existing aspect transform unchanged
   - wclip = 1.0 + z (fake perspective, far shrinks); out.clip = vec4f(pos, z, wclip)
   - out.info = vec4f(info.x, info.y, w, depth) (pass cursor-influence w + depth to FS)
4. **Fragment (fs_text, ~lines 77-88):**
   - SDF domain-warp near cursor: uv += vec2(sin(uv.y*40+time*3),cos(uv.x*40+time*3)) * (cursor_w*0.010) (SMALL — heavy warp mushes MSDF)
   - weight bias: (weight-0.5)*0.18 added to (dist-0.5) → bold high-retention, thin low
   - DoF: dof=(1-depth)*(1-cursor_w); screen_range_dof = screen_range/(1+dof*3) (far+unhovered=soft)
   - glow: mix(0.6,1.35,depth) + cursor_w*0.5 → near/focused glyphs push rgb>1.0 → bloom for FREE in the existing HDR chain
5. **Hover (existing pickAt):** CPU drives picked run's info.z forward (crisp+bright hero). **Click:** one-shot recall-shock flare (per-glyph shock-phase counter, deterministic).

## v2 (trophies)
6. Spring-physics COMPUTE pass — clone node-renderer.ts recall-sim template (createComputePipeline/beginComputePass/dispatchWorkgroups already proven). Widen Glyph +1 vec4f phys lane (pos_xy, vel_xy), critically-damped spring toward anchor, cursor w injects impulse. capture_mode[11] freezes physics for ?frame=N loops.
7. True 4D w-axis hover punch: dedicated w-coord lane, xw-plane rotation + double perspective divide, hovered engram swells and passes THROUGH neighbors.
8. Deterministic demo-cursor Lissajous path under capture_mode for hands-free hero clips.
9. Velocity-directional chromatic aberration on fast sweeps (cursor_vx/vy already plumbed).

## DETERMINISM + BLOOM + ZERO-DOM
Every animated term rides params.time/frame/pulse (fixed sim clock, NOT wall clock) + per-glyph instance_index seed — never Math.random. ?frame=N&loop=1 reproduces frame-for-frame. Cursor is the only live input; for clips, freeze cursor or script a demo-cursor Lissajous under capture_mode. All glow = pushing rgb>1.0 into the existing rgba16float→bloom scene (no bloom-chain changes). Everything = glyph-instance data + 2 shader edits + 1 uniform write. No per-letter DOM, no three.js, one canvas.

## SOURCES COMBINED
W-perspective 4D→3D→2D (tesseract 6-plane rotation + W-camera divide); cursor Gaussian z-lift + magnetic tilt (Olivier Larose magnetic lineage); SDF atlas-UV domain-warp (Xor GM Shaders SDF, warp the sample not the quad, small-amplitude); SDF-threshold weight + DoF as gauge (Helsingin Sanomat Climate Crisis "type IS the gauge" + uncertainty→blur); spring physics compute (Codrops interactive text destruction, clone node-renderer.ts).
