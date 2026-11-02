#figure(
  table(
    columns: 4, align: (left, right, right, right), stroke: none,
    table.hline(),
    table.header([*Training $arrow$ test*], [*Classical (32)*], [*MLP64 (5552)*], [*GRU8 (336)*]),
    table.hline(),
    [25, 50 °C $arrow$ 70 °C], [9.3 / 5.6], [11.1 / 3.8], [16.4 / 7.9],
    [25, 70 °C $arrow$ 50 °C], [7.6 / 6.8], [7.0 / 4.5], [16.5 / 10.1],
    [$<= 320$ kHz $arrow$ \ 500, 800 kHz], [10.2 / 5.3], [10.7 / 3.8], [54.5 / 6.0],
    [10 % of the data], [6.5 / 6.4], [4.2 / 4.2], [9.4 / 9.6],
    [2 % of the data], [6.5 / 6.4], [4.5 / 4.2], [9.1 / 9.6],
    table.hline(),
  ),
  caption: [Test SRE (%) on the windows outside the training restriction, restricted / full-data model,
    mean over 3C94, N87, 3F4 and 77 and the three scenarios. MLP64: the MLP64-scheduled circuit (one seed). GRU8: the
    RHINO-MAG architecture trained with the same pipeline.],
) <tab:ood>
