package main

import (
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	// "gonum.org/v1/plot/plotutil"
	"image/color"
	"math"
)

const target = 100

func valueIteration(theta, headProb float64) []float64 {
	V := make([]float64, target+1)
	V[target] = 1
	for delta := theta + 1; delta > theta; {
		delta = 0
		for s := 1; s < target; s++ {
			v := V[s]
			new := 0.0
			for a := 1; s+a <= target && a <= s; a++ {
				actionValue := headProb*V[s+a] + (1-headProb)*V[s-a]
				if actionValue > new {
					new = actionValue
				}
			}
			V[s] = new
			possibleDelta := math.Abs(v - new)
			if possibleDelta > delta {
				delta = possibleDelta
			}
		}
	}
	return V
}

func plotValue(values map[float64][]float64, headProb float64, colors []color.RGBA) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = fmt.Sprintf("Probability of head: %v", headProb)
	p.X.Label.Text = "Capital"
	p.Y.Label.Text = "Value estimate"
	i := 0
	for theta, V := range values {
		pts := plotter.XYs(make([]plotter.XY, target+1))
		for i := range pts {
			pts[i] = plotter.XY{float64(i), V[i]}
		}
		line, err := plotter.NewLine(pts)
		if err != nil {
			panic(err)
		}
		line.LineStyle.Color = colors[i]
		i++
		p.Add(line)
		p.Legend.Add(fmt.Sprintf("theta=%f", theta), line)
	}
	err = p.Save(300, 300, fmt.Sprintf("gambling_value_%v.png", headProb))
	if err != nil {
		panic(err)
	}
}

func getPolicy(V []float64, headProb float64) []float64 {
	policy := make([]float64, target+1)
	for s := 1; s < target; s++ {
		new := 0.0
		action := 0.0
		for a := 1; s+a <= target && a <= s; a++ {
			actionValue := headProb*V[s+a] + (1-headProb)*V[s-a]
			if actionValue > new {
				new = actionValue
				action = float64(a)
			}
		}
		policy[s] = action
	}
	return policy
}

func plotPolicy(policy []float64, headProb float64, color color.RGBA) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = fmt.Sprintf("Probability of head: %v", headProb)
	p.X.Label.Text = "Capital"
	p.Y.Label.Text = "Policy"
	pts := plotter.XYs(make([]plotter.XY, target-1))
	for i := 1; i < target; i++ {
		pts[i-1] = plotter.XY{float64(i), policy[i]}
	}
	line, err := plotter.NewLine(pts)
	if err != nil {
		panic(err)
	}
	line.LineStyle.Color = color
	p.Add(line)
	err = p.Save(300, 300, fmt.Sprintf("gambling_policy_%v.png", headProb))
	if err != nil {
		panic(err)
	}
}

func main() {
	colors := []color.RGBA{
		color.RGBA{R: 191, G: 63, B: 63, A: 255},
		color.RGBA{R: 191, G: 191, B: 63, A: 255},
		color.RGBA{R: 127, G: 191, B: 63, A: 255},
		color.RGBA{R: 63, G: 127, B: 191, A: 255},
		color.RGBA{R: 63, G: 63, B: 191, A: 255},
		color.RGBA{R: 127, G: 63, B: 191, A: 255},
		color.RGBA{R: 191, G: 63, B: 191, A: 255},
		color.RGBA{R: 191, G: 63, B: 191, A: 255},
	}
	for _, headProb := range []float64{0.25, 0.55} {
		values := make(map[float64][]float64)
		for theta := 0.1; theta > 0.000001; theta /= 10 {
			values[theta] = valueIteration(theta, headProb)
		}
		plotValue(values, headProb, colors)
		policy := getPolicy(values[0.00001], headProb)
		plotPolicy(policy, headProb, colors[0])
	}
}
