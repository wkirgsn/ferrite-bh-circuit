#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.4"

// Block colors: measurement chain (specimen/instrument) blue, reversible curve green, hysteresis (plays) orange,
// relaxations violet, mean-field feedback red.
#let c-meas = rgb("#2f5fb3"); #let c-rev = rgb("#2e8b57"); #let c-play = rgb("#d9731a"); #let c-rel = rgb("#7a4fb5")
#let c-fb = rgb("#c0392b")
#let tint(c) = c.lighten(88%)

// Small sketches of each block's characteristic (cetz, 1 unit = 1 mm).
#let icon(c, body) = box(cetz.canvas(length: 1mm, {
  import cetz.draw: *
  set-style(stroke: (paint: c, thickness: 0.7pt, cap: "round", join: "round"))
  body
}))
#let ax = { import cetz.draw: *; line((0, 0), (9, 0), stroke: (paint: luma(150), thickness: 0.4pt)); line((0, 0), (0, 6), stroke: (paint: luma(150), thickness: 0.4pt)) }
#let ic-meas = icon(c-meas, {                 // B (dashed) lags behind the corrected u
  import cetz.draw: *; ax
  line(..range(0, 19).map(i => (i / 2, 3 + 2.4 * calc.sin(i / 2 * 0.75))))
  line(..range(0, 19).map(i => (i / 2, 3 + 2.0 * calc.sin(i / 2 * 0.75 - 0.6))), stroke: (dash: "densely-dashed"))
})
#let ic-rev = icon(c-rev, {                   // reversible curve with saturation
  import cetz.draw: *; ax
  line(..range(0, 19).map(i => (i / 2, 3 + 2.6 * calc.tanh((i / 2 - 4.5) * 0.55) + 0.05 * (i / 2 - 4.5))))
})
#let ic-play = icon(c-play, {                 // play loop (parallelogram)
  import cetz.draw: *; ax
  line((1, 0.8), (5.5, 0.8), (8, 5.2), (3.5, 5.2), close: true)
})
#let ic-lag = icon(c-meas, {                  // measured H lags behind H
  import cetz.draw: *; ax
  line((0.3, 0.6), (2, 0.6), (2, 5), (9, 5), stroke: (paint: luma(150), thickness: 0.5pt, dash: "densely-dashed"))
  line(..range(0, 15).map(i => (2 + i / 2, 0.6 + 4.4 * (1 - calc.exp(-i / 2 / 0.7)))))
})
#let ic-rel = icon(c-rel, {                   // relaxation: step input and lagging state
  import cetz.draw: *; ax
  line((0.3, 0.6), (2, 0.6), (2, 5), (9, 5), stroke: (paint: luma(150), thickness: 0.5pt, dash: "densely-dashed"))
  line(..range(0, 15).map(i => (2 + i / 2, 0.6 + 4.4 * (1 - calc.exp(-i / 2 / 1.6)))))
})

#let blk(pos, c, ic, body, name) = node(pos, grid(columns: 2, column-gutter: 2mm, align: horizon, ic, align(left, body)),
  fill: tint(c), stroke: 0.8pt + c, corner-radius: 2pt, inset: 4pt, name: name)

#let law-diagram = diagram(
  spacing: (6mm, 5mm), edge-stroke: 0.7pt,
  node((0, 0), $B, theta.alt$, stroke: none),
  edge("-|>"),
  blk((1, 0), c-meas, ic-meas, [measurement correction\ $u = B + tau_1 dot(B) + tau_2 dot.double(B)$], <u>),
  edge("-|>"),
  node((2.25, 0), $Sigma$, name: <sum1>, stroke: 0.7pt, shape: circle, inset: 2pt),
  blk((3.4, -0.85), c-play, ic-play, [6 viscous plays\ $p_i$, widths $r_i$], <plays>),
  blk((3.4, 0.85), c-rel, ic-rel, [2 relaxations\ $z_j$, rates $beta_j$], <relax>),
  blk((1, 1.7), c-rev, ic-rev, [reversible curve\ $h_s (u)$, saturation], <hs>),
  node((4.75, 0), $Sigma$, name: <sum2>, stroke: 0.7pt, shape: circle, inset: 2pt),
  edge(<sum2>, <lag>, "-|>", $H$, label-pos: 0.72),
  node((5.1, 0), none, shape: circle, radius: 1.4pt, fill: black, stroke: none, name: <tap>),
  blk((6.35, 0), c-meas, ic-lag, [measurement lag\ $eta$, current sense], <lag>),
  edge(<lag>, (8.6, 0), "-|>", $H^"m"$, label-pos: 0.28),
  edge(<sum1>, <plays>, "-|>", stroke: c-play),
  edge(<sum1>, <relax>, "-|>", stroke: c-rel),
  edge(<plays>, <sum2>, "-|>", $w_i$, stroke: c-play),
  edge(<relax>, <sum2>, "-|>", $c_j$, label-side: right, stroke: c-rel),
  edge(<u>, <hs>, "-|>", stroke: c-rev),
  edge(<hs>, (4.75, 1.7), <sum2>, "-|>", stroke: c-rev),
  edge(<tap>, (5.1, -1.75), (2.25, -1.75), <sum1>, "-|>", text(fill: c-fb)[mean-field feedback $gamma_i H_(n-1)$],
    label-pos: 0.5, stroke: (paint: c-fb, dash: "dashed", thickness: 0.8pt)),
)
