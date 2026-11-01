#figure(placement: top, scope: "parent", kind: table,
      text(size: 8pt, table(
        columns: 7, align: (left,) + (right,) * 6, stroke: none,
        table.hline(),
        table.header([*Model*], [*\#par*], [*SRE*], [*10/90*], [*50/50*], [*90/10*], [*W*]),
        table.hline(),
        [*classical circuit*], [32], [7.33 ± 0.03], [9.04], [6.85], [6.11], [12],  // 10 runs
    [without lag of $H$], [31], [7.46], [9.18], [6.95], [6.25], [12],
    [without lag of $H$, not passive], [31], [7.33 ± 0.06], [9.07], [6.84], [6.07], [11],  // 10 runs
    [$theta.alt$-independent measurement correction], [30], [7.57], [9.21], [7.10], [6.42], [11],
    [common feedback gain], [29], [7.58], [9.30], [7.08], [6.38], [12],
    [common feedback gain, $theta.alt$-independent measurement correction], [27], [7.79 ± 0.08], [9.37], [7.32], [6.67], [10],  // 10 runs
    [\+ $theta.alt$-independent feedback gain, cubic term, density slope], [24], [7.81], [9.44], [7.31], [6.68], [11],
        table.hline(),
        [RHINO-MAG GRU8 @vater2026rhino], [336], [7.77], [8.59], [8.14], [6.59], [],
        table.hline(),
      )),
      caption: [The classical circuit and smaller variants: mean test SRE in % over the 15 materials, overall and per
      scenario. W: materials (of 15) on which the model is more accurate than GRU8; ± is the standard deviation over
      training seeds where several were run. All models are trained on SRE + NERE.],
    ) <tab:path>
    