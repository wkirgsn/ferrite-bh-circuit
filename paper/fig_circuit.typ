#import "@preview/cetz:0.3.4": canvas, draw
#import "fig_law.typ": c-meas, c-rev, c-play, c-rel

// Equivalent circuit of the model. Symbols: upb-lea/Inkscape_electric_Symbols (CC0), exported to symbols/*.svg
// (vertical, 10.58 mm between lead ends, centered). Coordinates in mm.
#let circuit = canvas(length: 1mm, {
  import draw: *
  set-style(stroke: (thickness: 0.265mm, cap: "round", join: "round"))
  let h = 5.29                                          // half the lead-to-lead length of a symbol
  let part(name, pos, rot: 0deg) = content(pos, std.rotate(rot, reflow: true, image("symbols/" + name + ".svg", height: 10.85mm)))
  let v(name, x, y) = part(name, (x, y))                // vertical symbol centered at (x, y)
  let wire(..pts) = line(..pts)
  let lab(pos, body, anchor: "center") = content(pos, text(9pt, body), anchor: anchor)
  let junc(p) = circle(p, radius: 0.45, fill: black, stroke: none)
  let arrow(a, b) = line(a, b, mark: (end: ">", size: 1.2, fill: black))
  let csrc(x, y, cur: false) = {                        // controlled source: diamond with a line along (voltage) or across (current) the wire
    let r = 2.65
    line((x, y + r), (x + r, y), (x, y - r), (x - r, y), close: true)
    if cur { line((x - r, y), (x + r, y)); line((x, y + r), (x, y + 2 * r)); line((x, y - r), (x, y - 2 * r)) }
    else { line((x, y + 2 * r), (x, y - 2 * r)) }
  }
  let top = 60; let bot = 0; let right = 124
  let sx = 33; let px = 63; let jx = 106                 // branches: reversible curve, play, relaxation
  // pastel backgrounds of the element groups, in the block colors of the model diagram
  let grp(a, b, c) = rect(a, b, fill: c.lighten(86%), stroke: none, radius: 1.5)
  grp((-54, 14), (-5, 57.5), c-meas); grp((1.5, 14), (27, 57.5), c-meas)
  grp((28.5, 3), (46.3, 57.5), c-rev); grp((46.8, 3), (94.5, 57.5), c-play); grp((95, 3), (124, 57.5), c-rel)
  // terminals and rails
  circle((0, top), radius: 0.7, fill: white); circle((0, bot), radius: 0.7, fill: white)
  arrow((-1.2, top - 3), (-1.2, 3)); lab((-2.4, top / 2), $V$, anchor: "east")
  wire((0.7, top), (right, top)); wire((0.7, bot), (right, bot))
  // measurement path: R_m and L_m in series with C_m; v_B across C_m
  let mx = 8
  junc((mx, top)); wire((mx, top), (mx, 50 + h)); v("resistor", mx, 50); wire((mx, 50 - h), (mx, 38 + h)); v("inductor", mx, 38)
  wire((mx, 38 - h), (mx, 24 + h)); v("capacitor", mx, 24); wire((mx, 24 - h), (mx, bot)); junc((mx, bot))
  lab((mx + 2.4, 50), $R_m$, anchor: "west"); lab((mx + 2.0, 38), $L_m$, anchor: "west"); lab((mx + 3.4, 24), $C_m$, anchor: "west")
  arrow((mx + 11, 28), (mx + 11, 20)); lab((mx + 12, 24), $V_B$, anchor: "west")
  lab((9.5, 11.5), [measurement], anchor: "west")
  lab((21, top + 2.3), $I$ + sym.arrow.r)
  // reversible curve: nonlinear inductor
  junc((sx, top)); wire((sx, top), (sx, 30 + h)); v("inductor", sx, 30); wire((sx, 30 - h), (sx, bot)); junc((sx, bot))
  arrow((sx - 3, 25.5), (sx + 3.5, 34.5)); lab((sx + 3, 28), $cal(L)_s (psi)$, anchor: "west")
  lab((sx + 8, -3.5), [reversible curve])
  // viscous play (x6): controlled source, L_i, slider (diode bridge carrying I_c, in parallel with R_i); -L_i in parallel
  let nd = 43
  junc((px, top)); wire((px, top), (px, 50.6 + 5.3)); csrc(px, 50.6); wire((px, 50.6 - 5.3), (px, nd)); junc((px, nd))
  lab((px + 3.4, 50.6), $L_(gamma,i) thin dot(I)$, anchor: "west")
  v("inductor", px, nd - h); lab((px - 1.8, nd - h), $L_i$, anchor: "east")
  let bt = 30; let hw = 10; let bm = bt - hw; let bb = bt - 2 * hw
  wire((px, nd - 2 * h), (px, bt)); junc((px, bt)); junc((px, bb)); junc((px - hw, bm)); junc((px + hw, bm))
  wire((px, bt), (px + hw, bm), (px, bb), (px - hw, bm), close: true)
  part("diode", (px + hw / 2, bm + hw / 2), rot: 135deg)    // anode top, cathode right
  part("diode", (px + hw / 2, bm - hw / 2), rot: 45deg)     // anode bottom, cathode right
  part("diode", (px - hw / 2, bm + hw / 2), rot: 45deg)     // anode left, cathode top
  part("diode", (px - hw / 2, bm - hw / 2), rot: 135deg)    // anode left, cathode bottom
  wire((px - hw, bm), (px - h, bm)); part("current_source", (px, bm), rot: 90deg); wire((px + h, bm), (px + hw, bm))
  arrow((px + 2, bm + 4), (px - 2, bm + 4)); lab((px - hw - 0.8, bm + 2.6), $I_(c,i)$, anchor: "east")
  let rx = px + hw + 5
  wire((px, bt), (rx, bt), (rx, bm + h)); v("resistor", rx, bm); wire((rx, bm - h), (rx, bb), (px, bb))
  lab((rx + 2, bm), $R_i$, anchor: "west")
  wire((px, bb), (px, bot)); junc((px, bot))
  let nx = rx + 8
  wire((px, nd), (nx, nd), (nx, 25 + h)); v("inductor", nx, 25); wire((nx, 25 - h), (nx, bot)); junc((nx, bot))
  lab((nx + 1.8, 25), $-L_i$, anchor: "west")
  lab((px + 8, -3.5), [viscous plays $i = 1, ..., 6$])
  // relaxation (x2): controlled source, L_j, R_j; -L_j in parallel
  junc((jx, top)); wire((jx, top), (jx, 50.6 + 5.3)); csrc(jx, 50.6); wire((jx, 50.6 - 5.3), (jx, nd)); junc((jx, nd))
  lab((jx + 3.4, 50.6), $L_(gamma,j) thin dot(I)$, anchor: "west")
  v("inductor", jx, nd - h); wire((jx, nd - 2 * h), (jx, 24 + h)); v("resistor", jx, 24); wire((jx, 24 - h), (jx, bot)); junc((jx, bot))
  lab((jx - 1.8, nd - h), $L_j$, anchor: "east"); lab((jx - 2.2, 24), $R_j$, anchor: "east")
  let lx = jx + 9
  wire((jx, nd), (lx, nd), (lx, 25 + h)); v("inductor", lx, 25); wire((lx, 25 - h), (lx, bot)); junc((lx, bot))
  lab((lx + 1.8, 25), $-L_j$, anchor: "west")
  lab((jx + 4.5, -3.5), [relaxations $j = 1, 2$])
  // current sense: the recorded H is a first-order low-pass of I (controlled current source into R_h || C_h)
  let ht = 50; let hb = 24; let hx = -47; let rx2 = -36; let cx = -25
  wire((hx, ht), (cx, ht)); wire((hx, hb), (cx, hb))
  wire((hx, ht), (hx, 37 + 5.3)); wire((hx, 37 - 5.3), (hx, hb))
  csrc(hx, 37, cur: true); arrow((hx + 4, 35), (hx + 4, 39))
  lab((hx - 3.4, 37), $I$, anchor: "east")
  junc((rx2, ht)); junc((rx2, hb)); wire((rx2, ht), (rx2, 37 + h)); v("resistor", rx2, 37); wire((rx2, 37 - h), (rx2, hb))
  lab((rx2 + 2.2, 37), $R_h$, anchor: "west")
  wire((cx, ht), (cx, 37 + h)); v("capacitor", cx, 37); wire((cx, 37 - h), (cx, hb))
  lab((cx + 3.4, 37), $C_h$, anchor: "west")
  arrow((-14, 41), (-14, 33)); lab((-13, 37), $V_H$, anchor: "west")
  lab((-30, 11.5), [current sense])
  // gain scheduling: an MLP sets the element values of the material part (dashed) at every step
  let sc = luma(70)
  rect((28, -7.2), (125.2, 63.4), stroke: (paint: sc, thickness: 0.35mm, dash: "dashed"), radius: 1.5)
  rect((60, 67.5), (88, 75.5), stroke: (paint: sc, thickness: 0.3mm), fill: luma(245), radius: 1)
  content((74, 71.5), text(8pt)[scheduling MLP])
  line((38, 71.5), (60, 71.5), mark: (end: ">", size: 1.2, fill: black)); lab((37, 71.5), $u, dot(B), dot.double(B), theta.alt$, anchor: "east")
  line((74, 67.5), (74, 63.4), mark: (end: ">", size: 1.2, fill: black))
  lab((89.5, 71.5), text(8pt)[element values at every step], anchor: "west")
})
