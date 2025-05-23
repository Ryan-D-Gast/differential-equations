#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot

#set page(width: auto, height: auto, margin: 0.5cm)

// Load the data from a CSV file.
#let data = csv("target/sir_model.csv", row-type: dictionary)

// Extract the data into arrays.
#let t = data.map(x => float(x.t))
#let y0 = data.map(x => float(x.y0))
#let y1 = data.map(x => float(x.y1))
#let y2 = data.map(x => float(x.y2))

#block[
  #align(center, text(weight: "bold", size: 1.2em, "Susceptible-Infected-Recovered Model"))
  #canvas({
    import draw: *
    
    plot.plot(
      legend: "inner-north-east",
      x-label: "Time",
      y-label: "Population",
      y-min: 0,
      y-max: 1000,
      size: (12, 8),
      {
          plot.add(
              data.map(x => (float(x.t), float(x.y0))),
              label: "Susceptible",
              style: (stroke: blue),
          )
          plot.add(
              data.map(x => (float(x.t), float(x.y1))),
              label: "Infected",
              style: (stroke: red),
          )
          plot.add(
              data.map(x => (float(x.t), float(x.y2))),
              label: "Recovered",
              style: (stroke: green),
          )
      }
    )
  })
]