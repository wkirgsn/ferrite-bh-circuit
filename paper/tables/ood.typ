#figure(
  table(
    columns: 3, align: (left, right, right), stroke: none,
    table.hline(),
    table.header([*Training data*], [*Classical (32)*], [*GRU8 (336)*]),
    table.hline(),
    [25, 50 °C $arrow$ test at 70 °C], [9.3 / 5.6], [16.4 / 7.9],
    [25, 70 °C $arrow$ test at 50 °C], [7.6 / 6.8], [16.5 / 10.1],
    [$f <= 320$ kHz $arrow$ test at 500, 800 kHz], [10.2 / 5.3], [54.5 / 6.0],
    [10 % of the sequences], [6.5 / 6.4], [9.4 / 9.6],
    [2 % of the sequences], [6.5 / 6.4], [9.1 / 9.6],
    table.hline(),
  ),
  caption: [Test SRE (%) on the windows outside the training restriction, restricted / full-data model,
    mean over 3C94, N87, 3F4 and 77 and the three scenarios. GRU8: the RHINO-MAG architecture trained with the same
    pipeline.],
) <tab:ood>
