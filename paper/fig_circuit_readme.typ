// Standalone render of the equivalent circuit (fig_circuit.typ) for the README: paper/fig/circuit.svg,
// flattened to plain paths with svg_flatten.py (step_figures in scripts/reproduce.sh)
#import "fig_circuit.typ": circuit
#set page(width: auto, height: auto, margin: 4mm, fill: white)
#set text(font: "TeX Gyre Termes", size: 10pt)
#circuit
