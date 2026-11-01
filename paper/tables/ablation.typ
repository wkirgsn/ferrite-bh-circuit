#figure(placement: top, scope: "parent", kind: table,
  text(size: 8pt, table(
    columns: 8, align: (left,) + (right,) * 7, stroke: none,
    table.hline(),
    table.header([*Model*], [*\#par*], [*SRE*], [*$Delta$*], [*10/90*], [*50/50*], [*90/10*], [*W*]),
    table.hline(),
    [classical circuit], [32], [7.30], [], [8.98], [6.82], [6.11], [12],
    [without saturation knee], [29], [7.91], [+0.61], [9.83], [7.43], [6.48], [8],
    [without cubic term], [30], [7.60], [+0.29], [9.32], [7.17], [6.30], [10],
    [rate-independent plays ($alpha = 1$)], [30], [10.99], [+3.68], [13.23], [10.60], [9.14], [10],
    [without relaxations], [24], [7.68], [+0.38], [9.37], [7.26], [6.42], [11],
    [without mean-field feedback], [27], [8.09], [+0.79], [10.06], [7.59], [6.62], [11],
    [without measurement correction of $B$], [28], [12.62], [+5.32], [13.10], [12.09], [12.68], [0],
    [without lag of $H$], [31], [7.46], [+0.16], [9.18], [6.95], [6.25], [12],
    table.hline(),
  )),
  caption: [Ablation of the classical circuit: each row removes one classical element, all variants trained jointly with the
  same data and seed. Mean test SRE in % over the 15 materials, overall and per scenario; $Delta$: change against the
  classical circuit; W: materials (of 15) on which the variant is more accurate than GRU8.],
) <tab:ablation>
