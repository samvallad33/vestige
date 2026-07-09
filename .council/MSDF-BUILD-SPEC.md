# MSDF TEXT ENGINE — VERIFIED BUILD SPEC (trap-free, first-try-correct)

Produced by an adversarial verification workflow (5 verifiers read real source + 3 skeptics
attacked the silent-fail risks + WGSL validated under naga 30.0.0 / MSL 2.1). Every fix below
is verified against the real repo, atlas JSON, and the shipped reasoning-theater-pass.ts.
The swarm worker builds against THIS, not raw council files.

Repo: /Users/entity002/vestige · branch feat/dashboard-live-max · cd there, work in the real repo.

---

## THE 6 SILENT-FAIL TRAPS (each passes `pnpm check` and fails at runtime — resolve ALL)

1. **V-FLIP (upside-down glyphs).** atlas `yOrigin:"bottom"`; GPU texture V is top-down.
   On the CPU when packing instances: `u0=left/512, u1=right/512, v0=1-top/512, v1=1-bottom/512`.
   Verified for 'A' (atlasBounds top=373.5): `v0 = 1 - 373.5/512 = 0.2705`. Omit → every glyph
   upside down, passes check silently. GOLDEN TEST: assert the top-most vertex samples
   `1 - atlasBounds.top/512` (0.2705 for 'A') so inverted wiring fails LOUD.
   Shader mapping that matches this pack order (corner.y=0 is quad BOTTOM in +Y-up NDC):
   `let v = mix(uvMax.y, uvMin.y, corner.y);` where uvMin=(u0,v0)=glyph-top, uvMax=(u1,v1)=glyph-bottom.
   DO NOT use `out.uv = tex_off + corner*tex_ext` with tex_off=(u0,v0=glyph-top) + positive
   extent on a +Y-up quad — that double-negates the flip → upside down. (If you keep the
   additive form, set tex_off=(u0,v1=bottom) and tex_ext.y=v0-v1 NEGATIVE.)

2. **ATLAS NEVER LOADS IN THE RELEASE BINARY.** Production embeds apps/dashboard/build via
   include_dir! at compile time. The committed build/ tree is STALE (Jun 21, no msdf/). At
   runtime GET /dashboard/msdf/jetbrains-mono.png hits the SPA fallback → returns index.html
   with HTTP 200 (not 404) → `fetch().json()` throws parsing HTML, texture never creates, the
   ENTIRE text layer silently no-ops. FIX: run `pnpm --filter @vestige/dashboard build` BEFORE
   any cargo build (adapter-static copies static/msdf/ → build/msdf/). CI already does this
   (release.yml, pages.yml). Never test the embedded/release build against the stale committed
   build/ without rebuilding the dashboard first. HARDEN (separate small fix): make
   serve_dashboard_asset return a real 404 for .png/.json/.woff2 paths instead of the 200 SPA
   fallback, so a missing atlas fails loud.

3. **BASE PATH.** Fetch via `import { base } from '$app/paths'` → `${base}/msdf/jetbrains-mono.{png,json}`.
   NOT `import.meta.env.BASE_URL`, NOT hardcoded `/msdf/...`. Default base `/dashboard`
   (svelte.config.js VESTIGE_BASE_PATH ?? '/dashboard'; CI → `/vestige`). Codebase standard is
   $app/paths base everywhere. Hardcoded 404s in prod.

4. **ATLAS TEXTURE FORMAT = `rgba8unorm`, NOT `rgba8unorm-srgb`.** MTSDF channels are LINEAR
   distance values; an sRGB view corrupts the median → fuzzy/blown edges. Upload via
   fetch→blob→createImageBitmap→copyExternalImageToTexture (net-new; no existing example in repo).
   Usage: TEXTURE_BINDING | COPY_DST.

5. **BLEND = premultiplied over-blend, NOT additive.** Text is INK, not glowing energy —
   additive AA fringes go over-bright/crunchy. Use `color/alpha: {srcFactor:'one',
   dstFactor:'one-minus-src-alpha', operation:'add'}` and output premultiplied `vec4f(rgb*alpha, alpha)`.
   It STILL glows because it's drawn into the pre-bloom rgba16float HDR scene; the glow comes
   from the post chain, not the blend mode. (The reference pass uses one/one additive because
   its marks ARE energy — text is the exception.)

6. **ASPECT + ASCII + ROUTE NAME.**
   - Aspect: NDC is [-1,1] both axes but viewport isn't square. Size glyphs in NDC-Y units,
     divide X by `aspect = viewport_w/viewport_h` (params[6]/params[7]) IN THE VERTEX SHADER,
     and apply the SAME divide in pickAt or clicks miss.
   - ASCII-only atlas (95 glyphs, U+0020–U+007E). The demo `·` (U+00B7) and `…` (U+2026) are
     NOT present → they silently drop. Use `|` as separator and `...` for truncation. Filter
     any input string to 0x20–0x7E; substitute `?` (0x3F) for out-of-range codepoints.
   - `_msdftest` is a PUBLIC route at /dashboard/_msdftest (SvelteKit does NOT hide leading-
     underscore dirs — that's Remix/React-Router). Fine as a scratch name; just know it's routable.

CONFIRMED CLEAN (adversarial verdict, not a trap): the WGSL validates under naga 30.0.0 +
MSL 2.1 (the M1 Max Metal path Dawn uses), ZERO errors/warnings. Params struct = 16 f32 = 64
bytes = engine.paramsBuffer.byteLength. Single rgba16float color attachment, no depthStencil/
multisample — matches engine.ts main render pass. `in`/`out` are legal WGSL (only `input`/`output`
are reserved). Bind layout b0 uniform(V|F) / b1 read-only-storage(V) / b2 filtering-sampler(F) /
b3 float-texture(F) is a valid pairing with a filterable rgba8unorm atlas.

---

## DELIVERABLE FILES (5)

### 1. layout.ts — pure (text, opts) → GlyphInstance[]
GlyphInstance = 8 floats, 32-byte stride (matches webgpu-samples Char{texOffset,texExtent,size,offset}):
`{ x,y,w,h (quad in NDC/em from planeBounds), u,v,uw,vh (atlas UV, V ALREADY FLIPPED) }`.
Placement (verified vs real metrics, monospace advance=0.6 for ALL 95 glyphs incl space):
```
penX=0, penY=0  (baseline, +Y up)
per char: '\n' → penX=0, penY-=1.32(lineHeight); else
  g=glyphMap[cp] (fallback '?' if missing)
  if g.planeBounds: emit {x:penX+pb.left, y:pb.bottom, w:pb.right-pb.left, h:pb.top-pb.bottom, u,v,uw,vh}
  penX += 0.6 (ALWAYS, even for space/no-bounds)
```
Wrap/truncate in em units: maxChars=floor(maxWidth_em/0.6). Truncate with `...` (ASCII).
A full line box spans em [-0.30, +1.02] around baseline (descender..ascender) for pick-rect sizing.

### 2. msdf-atlas.ts — loader
fetch `${base}/msdf/jetbrains-mono.json` → glyph Map<unicode, {advance, planeBounds?, atlasBounds?}>
+ atlas{distanceRange:4,size:48,width:512,height:512} + metrics{lineHeight:1.32,ascender:1.02,descender:-0.3}.
fetch `${base}/msdf/jetbrains-mono.png` → blob → createImageBitmap → device.createTexture(rgba8unorm,
TEXTURE_BINDING|COPY_DST) → copyExternalImageToTexture. Sampler: linear/linear, clamp-to-edge.

### 3. text-layer.ts — FramePass (render-only, no compute)
Copy the reasoning-theater-pass.ts pattern: own createBindGroupLayout + createPipelineLayout
(NOT auto — paramsBuffer is engine-shared), ensurePipeline/ensureResources guards. Instance data
= flat Float32Array in a STORAGE|COPY_DST buffer read `var<storage, read>` indexed by
instance_index (NOT vertex attributes — the shipped code has zero vertex buffers). init() is
async (atlas fetch); render() no-ops until pipeline+bindGroup+glyphCount ready. setText(items)
builds the buffer + a runs[] list of NDC AABBs for picking. pickAt(ndcX,ndcY) AABB-tests runs
(apply the aspect divide). draw(6, glyphCount). Target engine.sceneFormat, premultiplied over-blend.

### 4. shaders/msdf-text.wgsl.ts — single render-only module (VERIFIED, validates clean)
struct Params (16 f32, exact engine layout) @group0@binding0 uniform;
struct Glyph {anchor_size:vec4f, quad_offset:vec4f, uv_rect:vec4f, info:vec4f, color:vec4f}
  (info.x=age_frame, info.y=reveal_span for glyph-by-glyph typewriter; NO reserved field names —
   uses info/beat/data2, avoids meta/active/filter/sample/texture/binding/common/override);
@binding1 var<storage,read> glyphs; @binding2 filtering sampler; @binding3 texture_2d<f32> atlas.
QUAD = 6-vert unit quad [0,1]². vertex: pos = anchor + quad_offset + corner*size, x/=aspect;
uv: u=mix(uvMin.x,uvMax.x,corner.x), v=mix(uvMax.y,uvMin.y,corner.y). fragment:
`fn median(c)=max(min(c.r,c.g),min(max(c.r,c.g),c.b))`; screen-space AA = fwidth(uv)*texDims →
texels/px, *distanceRange(4), px_dist=range*(dist-0.5), cov=clamp(px_dist+0.5,0,1)
[≡ smoothstep(-.5,.5,px_dist)]; reveal=clamp((frame-age_frame)/span,0,1); alpha=cov*base_a*reveal;
discard if <0.001; return vec4f(rgb*alpha, alpha).

### 5. routes/(app)/_msdftest/+page.svelte — test route (zero DOM beyond the canvas)
mounts ObservatoryCanvas demo="recall-path" seed="msdf-test-v1"; onready(engine)→ new TextLayerPass,
await init(), setText("hello | 5de3e41f | trust 51%") [note `|` not `·`], engine.addPass. Pointer
host div (transparent, paints nothing) forwards pointerdown → NDC → pass.pickAt. ObservatoryCanvas
renders exactly ONE <canvas>. (Flag: ObservatoryCanvas still has a DOM .fallback panel — later card.)

## ACCEPTANCE
Test route renders "hello | 5de3e41f | trust 51%" as glowing in-canvas MSDF text, RIGHT-SIDE UP,
crisp at any zoom, materializing glyph-by-glyph, pickable. Gate: `pnpm --filter @vestige/dashboard
check` (0 errors) + `build`. Conductor live-GPU audit: getCompilationInfo clean + screenshot shows
upright crisp glowing text (the V-flip golden test must pass).
