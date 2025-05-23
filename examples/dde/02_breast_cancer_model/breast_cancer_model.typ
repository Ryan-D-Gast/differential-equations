#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot

#set page(width: auto, height: auto, margin: 0.5cm)

#let data = csv("target/breast_cancer_model.csv", row-type: dictionary)
#let t = data.map(x => float(x.t))
#let u1 = data.map(x => float(x.y0))
#let u2 = data.map(x => float(x.y1))
#let u3 = data.map(x => float(x.y2))

#block[
  #align(center, text(weight: "bold", size: 1.2em, "Breast Cancer Model"))
  #canvas({
    import draw: *
    plot.plot(
      legend: "inner-north-east",
      x-label: "Time",
      y-label: "Population",
      size: (12, 8),
      {
          plot.add(
              data.map(x => (float(x.t), float(x.y0))),
              label: "u1 (Proliferating)",
              style: (stroke: blue),
          )
          plot.add(
              data.map(x => (float(x.t), float(x.y1))),
              label: "u2 (Quiescent)",
              style: (stroke: red),
          )
          plot.add(
              data.map(x => (float(x.t), float(x.y2))),
              label: "u3 (Resistant)",
              style: (stroke: green),
          )
      }
    )
  })
]
