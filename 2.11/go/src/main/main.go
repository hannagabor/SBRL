package main

import (
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/vg"

	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"math"
	"math/rand"
)

const numRuns = 2000
const numArms = 10
const numSteps = 200000

type Arm struct {
	mean   float64
	stdDev float64
}

func (a *Arm) getReward() float64 {
	return rand.NormFloat64()*a.stdDev + a.mean
}

func (a *Arm) changeMean() {
	change := rand.NormFloat64() * 0.01
	a.mean += change
}

func getPlayers() []Player {
	var players []Player
	for i := -7; i < 3; i++ {
		players = append(players, NewGreedyPlayer(math.Pow(2, float64(i))))
	}
	for i := -10; i < -3; i++ {
		players = append(players, NewEpsilonGreedyPlayer(math.Pow(2, float64(i))))
	}
	for i := 4; i < 10; i++ {
		players = append(players, NewUCBPlayer(math.Pow(2, float64(i))))
	}
	for i := -15; i < -10; i++ {
		players = append(players, NewGradientBanditPlayer(math.Pow(2, float64(i))))
	}
	return players
}

func getArms() []Arm {
	arms := make([]Arm, numArms, numArms)
	for i := 0; i < numArms; i++ {
		arms[i] = Arm{stdDev: 1}
	}
	return arms
}

func simulate() []Player {
	players := getPlayers()
	arms := getArms()
	for j := 0; j < numSteps; j++ {
		for _, player := range players {
			a := arms[player.choose()]
			r := a.getReward()
			player.updateInnerState(r)
		}
		for a := range arms {
			arms[a].changeMean()
		}
	}
	return players
}

func getTicks(min, max float64) []plot.Tick {
	ticks := make([]plot.Tick, 0)
	for i := -20; i < 20; i++ {
		t := math.Pow(2, float64(i))
		if min <= t && t <= max {
			var l string
			if i < 0 {
				l = fmt.Sprintf("1/%v", math.Pow(2, float64(-i)))
			} else {
				l = fmt.Sprintf("%v", t)
			}
			ticks = append(ticks, plot.Tick{Value: t, Label: l})
		}
	}
	return ticks
}

func createPlot(avgRewards map[string]plotter.XYs) {
	p, err := plot.New()
	p.X.Scale = plot.LogScale{}
	p.X.Tick.Marker = plot.TickerFunc(getTicks)
	p.Y.Label.Text = "Average reward ove steps 100000-200000"
	p.X.Label.Text = "init_value/epsilon/c/alpha"
	p.Title.Text = "Non-statoniary parameter study"
	if err != nil {
		panic(err)
	}
	plotInp := make([]interface{}, 0, len(avgRewards))
	for name, xys := range avgRewards {
		plotInp = append(plotInp, ([]interface{}{name, xys})...)
	}
	err = plotutil.AddLinePoints(p, plotInp...)
	if err != nil {
		panic(err)
	}
	if err := p.Save(8*vg.Inch, 8*vg.Inch, "parameter_study.png"); err != nil {
		panic(err)
	}
}

func main() {
	players := getPlayers()
	sumRewards := make([]float64, len(players), len(players))

	for i := 0; i < numRuns; i++ {
		players = simulate()
		for playerID, player := range players {
			sumRewards[playerID] += getAvgReward(player)
		}
	}
	avgRewards := make(map[string]plotter.XYs)
	for id, player := range players {
		x := player.getParam()
		y := sumRewards[id] / numRuns
		avgRewards[player.getName()] = append(avgRewards[player.getName()], plotter.XY{X: x, Y: y})
	}
	createPlot(avgRewards)
}
