package main

import (
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"math"
	"strconv"
)

const maxCars = 20
const gamma = 0.9
const theta = 0.0001

type State struct {
	loc1 int
	loc2 int
}

func (state *State) toIndex() int {
	return state.loc1*(maxCars+1) + state.loc2
}

func getStates() []State {
	numStates := (maxCars + 1) * (maxCars + 1)
	states := make([](State), numStates)
	for i := 0; i <= maxCars; i++ {
		for j := 0; j <= maxCars; j++ {
			state := State{i, j}
			states[state.toIndex()] = state
		}
	}
	return states
}

func (state *State) getActions() []int {
	actions := make([]int, 0, 11)
	for i := -5; i <= 5; i++ {
		if i > 0 {
			if i <= state.loc1 && state.loc2+i <= maxCars {
				actions = append(actions, i)
			}
		} else {
			if -i <= state.loc2 && state.loc1-i <= maxCars {
				actions = append(actions, i)
			}
		}
	}
	return actions
}

func cutProb(dist *distuv.Poisson, x int, max int) float64 {
	if x == max {
		return 1 - dist.CDF(float64(x)-0.5)
	} else {
		return dist.Prob(float64(x))
	}
}

type StateAction struct {
	s State
	a int
}

type StateReward struct {
	s State
	r float64
}

type Prob float64

type StateRewardProbs map[StateReward]Prob

func getP(states []State) map[StateAction]StateRewardProbs {
	p := make(map[StateAction]StateRewardProbs)
	return1Dist := &distuv.Poisson{Lambda: 3}
	return2Dist := &distuv.Poisson{Lambda: 2}
	rent1Dist := &distuv.Poisson{Lambda: 3}
	rent2Dist := &distuv.Poisson{Lambda: 4}
	for _, s := range states {
		actions := s.getActions()
		for _, a := range actions {
			loc1 := s.loc1 - a
			loc2 := s.loc2 + a
			reward := -2 * math.Abs(float64(a))
			for rent1 := 0; rent1 <= loc1; rent1++ {
				loc1AfterRent := loc1 - rent1
				rent1Prob := cutProb(rent1Dist, rent1, loc1)
				for rent2 := 0; rent2 <= loc2; rent2++ {
					loc2AfterRent := loc2 - rent2
					rent2Prob := cutProb(rent2Dist, rent2, loc2)
					for return1 := 0; return1 <= maxCars-loc1AfterRent; return1++ {
						return1Prob := cutProb(return1Dist, return1, maxCars-loc1AfterRent)
						for return2 := 0; return2 <= maxCars-loc2AfterRent; return2++ {
							return2Prob := cutProb(return2Dist, return2, maxCars-loc2AfterRent)
							newState := State{
								loc1AfterRent + return1,
								loc2AfterRent + return2,
							}
							r := reward + 10*float64(rent1+rent2)
							prob := rent1Prob * rent2Prob * return1Prob * return2Prob
							sa := StateAction{s, a}
							sr := StateReward{newState, r}
							if p[sa] == nil {
								p[sa] = make(StateRewardProbs)
							}
							p[sa][sr] += Prob(prob)
						}
					}
				}
			}
		}
	}
	return p
}

func eval(states []State, p map[StateAction]StateRewardProbs, V []float64, pi []int, theta float64) {
	for delta := theta + 1; delta > theta; {
		delta = 0
		for _, state := range states {
			v := V[state.toIndex()]
			a := pi[state.toIndex()]
			new := 0.0
			for sr, prob := range (p[StateAction{state, a}]) {
				new += float64(prob) * (sr.r + gamma*V[sr.s.toIndex()])
			}
			V[state.toIndex()] = new
			possibleDelta := math.Abs(v - new)
			if possibleDelta > delta {
				delta = possibleDelta
			}
		}
	}
}

func improve(states []State, p map[StateAction]StateRewardProbs, V []float64, pi []int) bool {
	policyStable := true
	for _, state := range states {
		oldAction := pi[state.toIndex()]
		oldActionValue := 0.0
		for sr, prob := range (p[StateAction{state, oldAction}]) {
			oldActionValue += float64(prob) * (sr.r + gamma*V[sr.s.toIndex()])
		}
		actions := state.getActions()
		max := oldActionValue
		argmax := oldAction
		for _, a := range actions {
			actionValue := 0.0
			for sr, prob := range (p[StateAction{state, a}]) {
				actionValue += float64(prob) * (sr.r + gamma*V[sr.s.toIndex()])
			}
			if actionValue > max || (actionValue == max && math.Abs(float64(a)) < math.Abs(float64(argmax))) {
				max = actionValue
				argmax = a
			}
		}
		pi[state.toIndex()] = argmax
		if oldAction != argmax {
			policyStable = false
		}
	}
	return policyStable
}

type Plottable struct {
	pi []int
}

func (plottable Plottable) Dims() (c, r int) {
	c = maxCars + 1
	r = maxCars + 1
	return
}

func (plottable Plottable) Z(c, r int) float64 {
	s := State{r, c}
	return float64(plottable.pi[s.toIndex()])
}

func (plottable Plottable) X(c int) float64 {
	return float64(c)
}

func (plottable Plottable) Y(r int) float64 {
	return float64(r)
}

func makePlot(states []State, pi []int) {
	plottable := Plottable{pi}
	pl, err := plot.New()
	if err != nil {
		panic(err)
	}
	labels := make([]float64, 0, 11)
	for i := -5.0; i < 6; i++ {
		labels = append(labels, i)
	}
	contour := plotter.NewContour(plottable, labels, nil)
	pl.Add(contour)
	labelsMap := make(map[int]plotter.XY, 11)
	for i, state := range states {
		labelsMap[pi[i]] = plotter.XY{float64(state.loc2), float64(state.loc1)}
	}
	XYs := make([]plotter.XY, 0, 11)
	stringLabels := make([]string, 0, 11)
	for i, pos := range labelsMap {
		stringLabels = append(stringLabels, strconv.Itoa(i))
		XYs = append(XYs, pos)
	}
	ls, err := plotter.NewLabels(plotter.XYLabels{
		XYs:    XYs,
		Labels: stringLabels,
	})
	if err != nil {
		panic(err)
	}
	pl.Add(ls)
	pl.X.Label.Text = "Number of cars at the second location"
	pl.Y.Label.Text = "Number of cars at the first location"
	err = pl.Save(1000, 1000, "pi.png")
	if err != nil {
		panic(err)
	}
}

func main() {
	states := getStates()
	p := getP(states)
	V := make([]float64, len(states))
	pi := make([]int, len(states))
	policyStable := false
	for !policyStable {
		eval(states, p, V, pi, theta)
		policyStable = improve(states, p, V, pi)
	}
	makePlot(states, pi)
}
