// Small dependency-free SVG charts for the results pages.
// Colors come from CSS custom properties so light/dark both work.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => `var(${v})`;
  const pct = (v, d = 1) => (v == null || isNaN(v) ? "–" : (v * 100).toFixed(d) + "%");

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function ticks(max, n) {
    const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const out = [];
    for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }
  function legend(host, items) {
    const lg = document.createElement("div");
    lg.className = "legend";
    for (const it of items) {
      const s = document.createElement("span");
      const i = document.createElement("i");
      if (it.dash) i.className = "dash";
      else if (it.square) i.className = "sq";
      i.style.background = css(it.color);
      s.append(i, it.name);
      lg.appendChild(s);
    }
    host.appendChild(lg);
  }
  function tipBox(host) {
    const t = document.createElement("div");
    t.className = "tip";
    host.appendChild(t);
    return {
      show(html, x, y) {
        t.innerHTML = html;
        t.style.opacity = 1;
        const w = t.offsetWidth, hw = host.clientWidth;
        let left = x + 14;
        if (left + w > hw) left = Math.max(0, x - w - 14);
        t.style.left = left + "px";
        t.style.top = Math.max(0, y - 10) + "px";
      },
      hide() { t.style.opacity = 0; },
    };
  }
  const row = (color, name, val) =>
    `<div class="row"><span>${color ? `<i style="background:${css(color)}"></i>` : ""}${name}</span><span>${val}</span></div>`;

  // ---------- line chart with crosshair tooltip ----------
  // opt: {series:[{name,color,points:[[x,y]],dash,markers}], refs:[{y,label}], xMax, yMax, xLabel, yLabel, xFmt, yFmt, height}
  function lineChart(host, opt) {
    const W = 640, H = opt.height || 320, m = { l: 46, r: 120, t: 12, b: 40 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xMin = opt.xMin ?? 0, xMax = opt.xMax, yMax = opt.yMax;
    const X = (v) => m.l + ((v - xMin) / (xMax - xMin)) * iw;
    const Y = (v) => m.t + ih - (v / yMax) * ih;
    legend(host, opt.series.map((s) => ({ name: s.name, color: s.color, dash: s.legendDash }))
      .concat((opt.refs || []).map((r) => ({ name: r.label, color: "--ref", dash: true }))));
    const svg = el("svg", { class: "viz", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opt.aria || "" }, host);
    const g = el("g", { class: "grid" }, svg);
    for (const t of ticks(yMax, 5)) {
      el("line", { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t) }, g);
      el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, svg).textContent = (opt.yFmt || ((v) => v))(t);
    }
    for (const t of opt.xTicks || ticks(xMax, 6)) {
      el("text", { x: X(t), y: m.t + ih + 18, "text-anchor": "middle" }, svg).textContent = t;
    }
    el("line", { class: "axis", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }, svg);
    if (opt.xLabel) el("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = opt.xLabel;
    if (opt.yLabel) el("text", { x: 12, y: m.t + ih / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + ih / 2})` }, svg).textContent = opt.yLabel;

    for (const r of opt.refs || []) {
      el("line", { x1: m.l, x2: m.l + iw, y1: Y(r.y), y2: Y(r.y), stroke: css("--ref"), "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
    }
    const ends = [];
    for (const s of opt.series) {
      const d = s.points.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(" ");
      el("path", { d, fill: "none", stroke: css(s.color), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
        "stroke-dasharray": s.dash ? "6 4" : "none" }, svg);
      if (s.markers) for (const p of s.points)
        el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: css(s.color), stroke: css("--surface"), "stroke-width": 2 }, svg);
      const last = s.points[s.points.length - 1];
      ends.push({ y: Y(last[1]), x: X(last[0]), s });
    }
    // direct labels at line ends, nudged apart
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
    for (const e of ends) {
      const t = el("text", { x: m.l + iw + 8, y: e.y + 4, class: "lbl" }, svg);
      t.textContent = e.s.short || e.s.name;
    }

    // hover layer
    const xs = [...new Set(opt.series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
    const cross = el("line", { y1: m.t, y2: m.t + ih, stroke: css("--axis"), "stroke-width": 1, opacity: 0 }, svg);
    const dots = opt.series.map((s) => el("circle", { r: 5, fill: css(s.color), stroke: css("--surface"), "stroke-width": 2, opacity: 0 }, svg));
    const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
    const tip = tipBox(host);
    function move(ev) {
      const pt = svg.createSVGPoint();
      const src = ev.touches ? ev.touches[0] : ev;
      pt.x = src.clientX; pt.y = src.clientY;
      const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      const xv = xMin + ((loc.x - m.l) / iw) * (xMax - xMin);
      const x0 = xs.reduce((a, b) => (Math.abs(b - xv) < Math.abs(a - xv) ? b : a));
      cross.setAttribute("x1", X(x0)); cross.setAttribute("x2", X(x0)); cross.setAttribute("opacity", 1);
      let html = `<b>${(opt.xTipFmt || ((v) => v))(x0)}</b>`;
      opt.series.forEach((s, i) => {
        const p = s.points.find((q) => q[0] === x0);
        if (p) {
          dots[i].setAttribute("cx", X(p[0])); dots[i].setAttribute("cy", Y(p[1])); dots[i].setAttribute("opacity", 1);
          html += row(s.color, s.name, (opt.yFmt || ((v) => v))(p[1]));
        } else dots[i].setAttribute("opacity", 0);
      });
      for (const r of opt.refs || []) html += row(null, r.label, (opt.yFmt || ((v) => v))(r.y));
      const box = host.getBoundingClientRect();
      tip.show(html, src.clientX - box.left, src.clientY - box.top);
    }
    function leave() { cross.setAttribute("opacity", 0); dots.forEach((d) => d.setAttribute("opacity", 0)); tip.hide(); }
    hit.addEventListener("mousemove", move);
    hit.addEventListener("touchmove", move, { passive: true });
    hit.addEventListener("mouseleave", leave);
    hit.addEventListener("touchend", leave);
  }

  // ---------- 100% stacked horizontal bars ----------
  // opt: {rows:[{name, values:[..]}], parts:[{name,color}]}
  function stackBar(host, opt) {
    const W = 640, rowH = 34, gap = 14, m = { l: 112, r: 12, t: 4, b: 26 };
    const H = m.t + m.b + opt.rows.length * (rowH + gap) - gap;
    const iw = W - m.l - m.r;
    legend(host, opt.parts.map((p) => ({ name: p.name, color: p.color, square: true })));
    const svg = el("svg", { class: "viz", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opt.aria || "" }, host);
    const g = el("g", { class: "grid" }, svg);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      el("line", { x1: m.l + t * iw, x2: m.l + t * iw, y1: m.t, y2: H - m.b }, g);
      el("text", { x: m.l + t * iw, y: H - 8, "text-anchor": "middle" }, svg).textContent = t * 100 + "%";
    }
    const tip = tipBox(host);
    opt.rows.forEach((r, ri) => {
      const y = m.t + ri * (rowH + gap);
      el("text", { x: m.l - 10, y: y + rowH / 2 + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = r.name;
      const total = r.values.reduce((a, b) => a + b, 0);
      let x = m.l;
      r.values.forEach((v, i) => {
        const w = (v / total) * iw;
        const isFirst = i === 0, isLast = i === r.values.length - 1;
        const rect = el("rect", { x: x + (isFirst ? 0 : 1), y, width: Math.max(0, w - (isFirst || isLast ? 1 : 2)), height: rowH,
          rx: 4, fill: css(opt.parts[i].color) }, svg);
        if (w > 38) {
          const t = el("text", { x: x + w / 2, y: y + rowH / 2 + 4, "text-anchor": "middle", style: "fill:#fff;font-weight:600" }, svg);
          t.textContent = (v * 100).toFixed(0) + "%";
        }
        const hitR = el("rect", { x, y: y - 4, width: w, height: rowH + 8, fill: "transparent" }, svg);
        hitR.addEventListener("mousemove", (ev) => {
          const box = host.getBoundingClientRect();
          tip.show(`<b>${r.name}</b>` + row(opt.parts[i].color, opt.parts[i].name, pct(v)), ev.clientX - box.left, ev.clientY - box.top);
          rect.setAttribute("opacity", 0.85);
        });
        hitR.addEventListener("mouseleave", () => { tip.hide(); rect.setAttribute("opacity", 1); });
        x += w;
      });
    });
  }

  // ---------- CI dot plot (paired differences) ----------
  // opt: {rows:[{name, mean, lo, hi, note}], xMin, xMax}
  function ciPlot(host, opt) {
    const W = 640, rowH = 40, m = { l: 210, r: 20, t: 8, b: 30 };
    const H = m.t + m.b + opt.rows.length * rowH;
    const iw = W - m.l - m.r;
    const X = (v) => m.l + ((v - opt.xMin) / (opt.xMax - opt.xMin)) * iw;
    const svg = el("svg", { class: "viz", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opt.aria || "" }, host);
    const g = el("g", { class: "grid" }, svg);
    for (let t = opt.xMin; t <= opt.xMax + 1e-9; t += opt.step) {
      el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b }, g);
      el("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, svg).textContent = (t > 0 ? "+" : "") + t;
    }
    el("line", { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1.5 }, svg);
    const tip = tipBox(host);
    opt.rows.forEach((r, i) => {
      const y = m.t + i * rowH + rowH / 2;
      el("text", { x: m.l - 12, y: y + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = r.name;
      el("line", { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: css(r.color), "stroke-width": 2, "stroke-linecap": "round" }, svg);
      el("circle", { cx: X(r.mean), cy: y, r: 5, fill: css(r.color), stroke: css("--surface"), "stroke-width": 2 }, svg);
      const hitR = el("rect", { x: m.l, y: y - rowH / 2, width: iw, height: rowH, fill: "transparent" }, svg);
      hitR.addEventListener("mousemove", (ev) => {
        const box = host.getBoundingClientRect();
        tip.show(`<b>${r.name}</b>` + row(null, "mean diff", r.mean.toFixed(1) + "pp") + row(null, "95% CI", `[${r.lo.toFixed(1)}, ${r.hi.toFixed(1)}]`) +
          row(null, "Wilcoxon p", r.p.toFixed(3)), ev.clientX - box.left, ev.clientY - box.top);
      });
      hitR.addEventListener("mouseleave", () => tip.hide());
    });
  }

  window.Charts = { lineChart, stackBar, ciPlot, pct };
})();
