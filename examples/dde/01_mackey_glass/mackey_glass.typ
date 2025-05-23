#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot

#set page(width: auto, height: auto, margin: 0.5cm)

#let data = csv("target/mackey_glass.csv", row-type: dictionary)
#let t = data.map(x => float(x.t))
#let y = data.map(x => float(x.y0))

#block[
  #align(center, text(weight: "bold", size: 1.2em, "Mackey-Glass Model"))
  #canvas({
    import draw: *
    plot.plot(
      x-label: "Time",
      y-label: "y",
      y-min: 0,
      size: (12, 8),
      {
          plot.add(
              data.map(x => (float(x.t), float(x.y0))),
              style: (stroke: blue),
          )
      }
    )
  })
]
