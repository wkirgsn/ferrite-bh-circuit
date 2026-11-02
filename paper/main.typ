#import "@preview/charged-ieee:0.1.4": ieee

#set page(numbering: "1")

#show: ieee.with(
  title: [A Data-Driven, Gain-Scheduled Equivalent Circuit for the $B$–$H$ Behavior of Power Ferrites],
  abstract: include "sections/abstract.typ",
  authors: (
    (
      name: "Wilhelm Kirchgässner",
      organization: "Beckhoff Automation",
      email: "wk.research@web.de",
    ),
  ),
  index-terms: ("Hysteresis", "power ferrites", "play operator", "Prandtl–Ishlinskii model", "equivalent circuit", "gain scheduling", "magnetic relaxation", "system identification", "MagNet Challenge", "interpretable machine learning"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

// All figures and tables float (top or bottom of a column) so that the text flows without gaps.
#set figure(placement: auto)

#include "sections/introduction.typ"
#include "sections/related.typ"
#include "sections/data.typ"
#include "sections/model.typ"
#include "sections/observer.typ"
#include "sections/implementation.typ"
#include "sections/results.typ"
#include "sections/discussion.typ"
#include "sections/conclusion.typ"
#include "sections/acknowledgment.typ"
