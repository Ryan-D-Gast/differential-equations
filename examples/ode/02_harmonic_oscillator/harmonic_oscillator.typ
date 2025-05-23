#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot

#set page(width: auto, height: auto, margin: 0.5cm)

#let data = csv("target/harmonic_oscillator.csv", row-type: dictionary)
#let t = data.map(x => float(x.t))
#let y0 = data.map(x => float(x.y0))
#let y1 = data.map(x => float(x.y1))

#block[
  #align(center, text(weight: "bold", size: 1.2em, "Harmonic Oscillator"))
  #canvas({
    import draw: *
    plot.plot(
      legend: "inner-north-east",
      x-label: "Time",
      y-label: "Value",
      size: (12, 8),
      {
          plot.add(
              data.map(x => (float(x.t), float(x.y0))),
              label: "Position",
              style: (stroke: blue),
          )
          plot.add(
              data.map(x => (float(x.t), float(x.y1))),
              label: "Velocity",
              style: (stroke: red),
          )
      }
    )
  })
]
