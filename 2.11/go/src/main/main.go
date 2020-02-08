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

const numRuns = 1
const numArms = 10
const numSteps = 5

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

type Player interface {
	choose() int
	updateInnerState(reward float64)
	getRewardSum() float64
	getParam() float64
	getName() string
}

func maxFloat(s []float64) int {
	maxValue := math.Inf(-1)
	maxId := 0
	for i, e := range s {
		if e > maxValue {
			maxValue = e
			maxId = i
		}
	}
	return maxId
}

func getAvgReward(p Player) float64 {
	return p.getRewardSum() / (numSteps / 2)
}

type GreedyPlayer struct {
	param       float64
	step        int
	name        string
	rewardSum   float64 // Sum of rewards in the second half.
	estimations []float64
	alpha       float64
	chosen      int
}

func (p *GreedyPlayer) getParam() float64 {
	return p.param
}

func (p *GreedyPlayer) getRewardSum() float64 {
	return p.rewardSum
}

func (p *GreedyPlayer) getName() string {
	return p.name
}

func (p *GreedyPlayer) choose() int {
	p.chosen = maxFloat(p.estimations)
	return p.chosen
}

func (p *GreedyPlayer) updateInnerState(reward float64) {
	p.step += 1
	if p.step > numSteps/2 {
		p.rewardSum += reward
	}
	p.estimations[p.chosen] += p.alpha * (reward - p.estimations[p.chosen])
}

func NewGreedyPlayer(initValue float64) *GreedyPlayer {
	estimations := make([]float64, numArms, numArms)
	for i := 0; i < numArms; i++ {
		estimations[i] = initValue
	}
	return &GreedyPlayer{
		param:       initValue,
		name:        "optimistic greedy",
		estimations: estimations,
		alpha:       0.1,
	}
}

func getPlayers() []Player {
	var players []Player
	for i := -2; i < 3; i++ {
		players = append(players, NewGreedyPlayer(math.Pow(2, float64(i))))
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

func simulate(player Player) float64 {
	sumAvgReward := 0.0
	for i := 0; i < numRuns; i++ {
		arms := getArms()
		for j := 0; j < numSteps; j++ {
			a := arms[player.choose()]
			r := a.getReward()
			player.updateInnerState(r)
			a.changeMean()
			sumAvgReward += getAvgReward(player)
		}
	}
	return sumAvgReward
}

func getTicks(min, max float64) []plot.Tick {
	ticks := make([]plot.Tick, 0)
	for i := -5; i < 5; i++ {
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
	// p.X.Tick.Marker = plot.LogTicks{}
	p.X.Tick.Marker = plot.TickerFunc(getTicks)
	if err != nil {
		panic(err)
	}
	for name, xys := range avgRewards {
		err = plotutil.AddLinePoints(p, name, xys)
		if err != nil {
			panic(err)
		}
	}
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "plot.png"); err != nil {
		panic(err)
	}
}

func main() {
	players := getPlayers()
	avgRewards := make(map[string]plotter.XYs)
	for _, player := range players {
		x := player.getParam()
		y := simulate(player)
		avgRewards[player.getName()] = append(avgRewards[player.getName()], plotter.XY{X: x, Y: y})
		fmt.Println(avgRewards)
	}
	createPlot(avgRewards)
}
