/**
 * Maximum-realism CRT post-process.
 *
 * The entire game is drawn as crisp low-res NES pixel art onto a source canvas;
 * this module renders it to the visible WebGL canvas as if shown on a real
 * consumer CRT television, modelling the chain of analog artifacts that make a
 * tube look like a tube:
 *
 *   - NTSC composite signal: chroma/luma crosstalk -> colour bleed + dot crawl
 *   - Aperture-grille / shadow-mask RGB phosphor triads
 *   - Per-channel beam convergence error (slight RGB misregistration)
 *   - Scanlines with a realistic gaussian beam profile (not a flat sine)
 *   - Phosphor persistence / afterglow (previous frame bleeds forward)
 *   - Bloom / phosphor glow around bright areas
 *   - Barrel (pincushion) screen curvature + rounded corners
 *   - Vignette, brightness falloff, and a faint mains-hum flicker
 *
 * Phosphor persistence needs the previous output, so we ping-pong two
 * framebuffers: pass A composites source+artifacts+persistence into a target;
 * pass B copies that target to the screen. Returns null if WebGL is missing.
 */

// Both passes use a plain fullscreen blit (no UV flips). Source orientation is
// handled once, at upload time, via UNPACK_FLIP_Y_WEBGL, so the texture already
// matches WebGL's bottom-left origin and no per-pass flipping is needed.
const VERT_CRT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = (a_pos + 1.0) * 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;
const VERT_COPY = VERT_CRT;

// ---- main CRT pass: source + previous (persistence) -> offscreen target ----
const FRAG_CRT = `
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_src;   // crisp NES composite (this frame)
uniform sampler2D u_prev;  // previous CRT output (for phosphor persistence)
uniform vec2 u_res;        // output resolution (px)
uniform vec2 u_srcRes;     // source resolution (px) -> scanline count
uniform float u_time;
uniform vec2 u_picScale;   // picture half-extent within the tube ([0..1] each axis)

// ---------------------------------------------------------------------------
// CRT brightness / calibration model.
//
// A real tube does NOT lose brightness the way a naive "multiply by scanline,
// multiply by mask, multiply by vignette" shader does. Three things must hold
// for the picture to read at a realistic luminance:
//
//  1. GAMMA / EOTF. The source is gamma-encoded (sRGB-ish). A CRT's gun has a
//     native power-law response (~2.4) — that is literally where "gamma" comes
//     from. Beam and mask attenuation must happen in LINEAR light, then the
//     result is re-encoded, or midtones get crushed and the image looks muddy.
//
//  2. ENERGY CONSERVATION. Scanlines redistribute a fixed amount of beam energy
//     into a bright centre and dark gaps; they don't delete it. We normalise the
//     scanline profile so its AVERAGE over a line is ~1.0 (a gain term offsets
//     the dark gaps) instead of just darkening everything.
//
//  3. MASK GAIN COMPENSATION. A shadow-mask / aperture-grille blocks 30-40% of
//     the beam, but tubes are calibrated with extra gun current so full white
//     still reaches the rated luminance. We divide the picture back up by the
//     mask's average transmission so the mask adds texture without dimming.
//
// On top of that a master GAIN + soft-knee tone-map gives highlights somewhere
// to bloom instead of clipping flat. Tunable constants, calibrated to a bright
// consumer set viewed in a dim room.
// ---------------------------------------------------------------------------
const float CRT_GAMMA   = 2.4;   // tube EOTF exponent
const float OUT_GAMMA   = 2.2;   // sRGB-ish re-encode
const float MASTER_GAIN = 1.55;  // overall brightness lift (post energy/mask comp)
const float BLACK_LIFT  = 0.018; // CRT black is a lifted dark gray

vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(CRT_GAMMA)); }
vec3 toGamma(vec3 c)  { return pow(max(c, 0.0), vec3(1.0 / OUT_GAMMA)); }

// Reinhard-style soft knee so values pushed past 1.0 by gain/mask/bloom roll
// off into a highlight instead of clipping to a flat plane.
vec3 tonemap(vec3 c) {
  return c / (1.0 + max(c - 0.7, 0.0));
}

// Barrel curvature of the virtual tube.
vec2 curve(vec2 uv) {
  uv = uv * 2.0 - 1.0;
  vec2 off = abs(uv.yx) / vec2(9.0, 7.0);
  uv = uv + uv * off * off;
  return uv * 0.5 + 0.5;
}

// Map a tube-space coord into the centred 4:3 picture region. Returns source UVs
// in [0,1] inside the picture, and out-of-range values outside it (so the caller
// can render those areas as dark tube with no picture).
vec2 toPicture(vec2 uv) {
  return (uv - 0.5) / u_picScale + 0.5;
}
bool inPicture(vec2 p) {
  return p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0;
}

// Sample the source as an NTSC composite: average a few horizontal taps with a
// phase-shifted weighting so chroma smears horizontally (colour bleed) and a
// time/line dependent term produces the characteristic crawling dots.
vec3 ntsc(vec2 uv) {
  vec2 px = vec2(1.0 / u_srcRes.x, 0.0);
  // luma-ish center plus smeared neighbours
  vec3 c0 = texture2D(u_src, uv).rgb;
  vec3 cl = texture2D(u_src, uv - px).rgb;
  vec3 cr = texture2D(u_src, uv + px).rgb;
  vec3 cl2 = texture2D(u_src, uv - px * 2.0).rgb;
  vec3 cr2 = texture2D(u_src, uv + px * 2.0).rgb;
  // chroma bleed: colour leaks sideways more than luma
  vec3 bleed = (cl * 0.25 + c0 * 0.5 + cr * 0.25);
  vec3 wide = (cl2 + cr2) * 0.12 + bleed * 0.76;
  // dot crawl: a faint diagonal chroma shimmer keyed to scanline + time
  float crawl = sin((uv.y * u_srcRes.y + u_time * 30.0) * 3.14159) * 0.012;
  return clamp(wide + crawl, 0.0, 1.0);
}

void main() {
  vec2 uv = curve(v_uv);

  // Off the tube -> rounded black bezel interior.
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // Map into the centred 4:3 picture region. Outside it, there is no picture
  // (the dark inner glass of the set), but the tube treatment still applies.
  vec2 pic = toPicture(uv);
  bool hasPic = inPicture(pic);

  vec3 col = vec3(0.0);
  vec3 glowLin = vec3(0.0);
  if (hasPic) {
    // Beam convergence error: sample R/G/B at slightly different offsets,
    // growing toward the edges of the picture.
    float edge = length(v_uv - 0.5);
    vec2 conv = vec2(0.0007 + edge * 0.0018, 0.0);
    col.r = ntsc(pic + conv).r;
    col.g = ntsc(pic).g;
    col.b = ntsc(pic - conv).b;

    // Work in LINEAR light from here on so beam/mask attenuation matches how a
    // real tube responds (and midtones don't get crushed).
    col = toLinear(col);

    // Bloom / phosphor glow sampled and accumulated in linear light too, so the
    // halo around bright pixels has physically-plausible energy.
    vec2 bpx = 1.6 / u_res;
    glowLin += toLinear(texture2D(u_src, pic + vec2( bpx.x, 0.0)).rgb);
    glowLin += toLinear(texture2D(u_src, pic + vec2(-bpx.x, 0.0)).rgb);
    glowLin += toLinear(texture2D(u_src, pic + vec2(0.0,  bpx.y)).rgb);
    glowLin += toLinear(texture2D(u_src, pic + vec2(0.0, -bpx.y)).rgb);
    glowLin += toLinear(texture2D(u_src, pic + bpx).rgb);
    glowLin += toLinear(texture2D(u_src, pic - bpx).rgb);
    glowLin *= 1.0 / 6.0;
  }

  // --- Scanlines, energy-conserving. A gaussian beam profile locked to source
  // rows, but NORMALISED so its average over a line-pair is ~1.0: the bright
  // centre is boosted to pay back what the dark gaps remove, instead of just
  // dimming the whole picture. ---
  float line = fract(uv.y * u_srcRes.y);
  float d = line - 0.5;
  float beam = exp(-d * d * 9.0);
  // mean of exp(-d^2*9) over d in [-0.5,0.5] is ~0.585; divide by it so the
  // line-averaged gain is 1.0 (energy conserved), then shape contrast a touch.
  float scan = mix(0.75, 1.0, beam) / 0.875;
  col *= scan;

  // --- Aperture-grille mask with GAIN COMPENSATION. The stripes attenuate, but
  // we divide the picture back up by the mask's average transmission so full
  // white still reaches full luminance (as a calibrated tube would). ---
  float mx = mod(gl_FragCoord.x, 3.0);
  vec3 mask = mx < 1.0 ? vec3(1.0, 0.70, 0.70)
            : mx < 2.0 ? vec3(0.70, 1.0, 0.70)
                       : vec3(0.70, 0.70, 1.0);
  float my = step(0.5, fract(gl_FragCoord.y / 3.0));
  mask *= mix(0.90, 1.0, my);
  // average transmission of the mask ~ (1.0+0.70+0.70)/3 * 0.95 ~ 0.76; renormalise
  col *= mask / 0.76;

  // --- Master gain, then add bloom, then tone-map so highlights roll off into a
  // glow instead of clipping flat. All in linear light. ---
  col *= MASTER_GAIN;
  col += max(glowLin - 0.30, 0.0) * 0.85; // additive phosphor bloom
  col = tonemap(col);

  // Back to display gamma.
  col = toGamma(col);

  // Phosphor persistence: blend a little of the previous output (afterglow).
  // Persistence is stronger on the green channel, like real P22 phosphors.
  vec3 prev = texture2D(u_prev, v_uv).rgb;
  vec3 persist = vec3(0.10, 0.16, 0.10);
  col = max(col, prev * persist);

  // Vignette + corner darkening — gentle, so it shades the edges without
  // dimming the centre (the old 0.9 falloff ate a lot of overall brightness).
  float vig = 1.0 - dot(v_uv - 0.5, v_uv - 0.5) * 0.45;
  col *= clamp(vig, 0.0, 1.0);

  // Faint 60Hz-ish brightness hum / flicker.
  col *= 0.99 + 0.01 * sin(u_time * 9.0 + uv.y * 5.0);

  // Tube black is a lifted dark gray, not pure black.
  col = max(col, vec3(BLACK_LIFT));

  gl_FragColor = vec4(col, 1.0);
}`;

// ---- present pass: copy the offscreen target to the screen ----
const FRAG_COPY = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_tex;
void main() { gl_FragColor = texture2D(u_tex, v_uv); }`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.error("CRT shader error:", gl.getShaderInfoLog(sh));
    return null;
  }
  return sh;
}

function program(gl, vertSrc, fragSrc) {
  const vs = compile(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragSrc);
  if (!vs || !fs) return null;
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error("CRT link error:", gl.getProgramInfoLog(p));
    return null;
  }
  return p;
}

function makeTarget(gl, w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { tex, fbo, w, h };
}

export function createCrt(glCanvas, source) {
  const gl =
    glCanvas.getContext("webgl", { antialias: false, premultipliedAlpha: false }) ||
    glCanvas.getContext("experimental-webgl");
  if (!gl) return null;

  const crtProg = program(gl, VERT_CRT, FRAG_CRT);
  const copyProg = program(gl, VERT_COPY, FRAG_COPY);
  if (!crtProg || !copyProg) return null;

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );

  function bindAttrib(prog) {
    const a = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(a);
    gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
  }

  // source texture
  const srcTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, srcTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  // ping-pong targets for persistence; (re)created on resize.
  let targets = null;
  let tw = 0, th = 0;
  function ensureTargets() {
    const w = glCanvas.width;
    const h = glCanvas.height;
    if (targets && tw === w && th === h) return;
    targets = [makeTarget(gl, w, h), makeTarget(gl, w, h)];
    tw = w; th = h;
  }
  let cur = 0;

  const uCrt = {
    src: gl.getUniformLocation(crtProg, "u_src"),
    prev: gl.getUniformLocation(crtProg, "u_prev"),
    res: gl.getUniformLocation(crtProg, "u_res"),
    srcRes: gl.getUniformLocation(crtProg, "u_srcRes"),
    time: gl.getUniformLocation(crtProg, "u_time"),
    picScale: gl.getUniformLocation(crtProg, "u_picScale"),
  };

  // The picture is a classic 4:3 region centred in the (full-viewport) tube.
  const PICTURE_ASPECT = 4 / 3;
  const uCopy = { tex: gl.getUniformLocation(copyProg, "u_tex") };

  return {
    render(time) {
      ensureTargets();
      const w = glCanvas.width, h = glCanvas.height;
      const prev = targets[cur];
      const next = targets[cur ^ 1];

      // upload this frame's crisp composite, flipping Y so the DOM canvas's
      // top-left origin matches WebGL's bottom-left texture space.
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

      // --- pass A: CRT into `next`, reading `prev` for persistence ---
      gl.bindFramebuffer(gl.FRAMEBUFFER, next.fbo);
      gl.viewport(0, 0, w, h);
      gl.useProgram(crtProg);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      bindAttrib(crtProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.uniform1i(uCrt.src, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, prev.tex);
      gl.uniform1i(uCrt.prev, 1);
      gl.uniform2f(uCrt.res, w, h);
      gl.uniform2f(uCrt.srcRes, source.width, source.height);
      gl.uniform1f(uCrt.time, time);
      // Fit a 4:3 picture inside the output: scale down the longer axis so the
      // picture keeps its aspect and is centred, with the rest as dark tube.
      const outAspect = w / h;
      let sx = 1, sy = 1;
      if (outAspect > PICTURE_ASPECT) sx = PICTURE_ASPECT / outAspect; // wide: bars L/R
      else sy = outAspect / PICTURE_ASPECT; // tall: bars top/bottom
      gl.uniform2f(uCrt.picScale, sx, sy);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      // --- pass B: present `next` to the screen ---
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, w, h);
      gl.useProgram(copyProg);
      bindAttrib(copyProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, next.tex);
      gl.uniform1i(uCopy.tex, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      cur ^= 1; // next becomes prev for the following frame
    },
  };
}
