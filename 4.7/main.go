package main

import (
	"fmt"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
)

const maxCars = 20
const gamma = 0.9
const theta = 0.1

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
		return 1 - dist.CDF(float64(x)-1)
	} else {
		return dist.Prob(float64(x))
	}
}

type StateAction struct {
	s State
	a int
}

type StateRewardProb struct {
	s    State
	r    float64
	prob float64
}

type StateRewardProbs []StateRewardProb

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
				rent1Prob := cutProb(rent1Dist, rent1, loc1)
				for rent2 := 0; rent2 <= loc2; rent2++ {
					rent2Prob := cutProb(rent2Dist, rent2, loc2)
					for return1 := 0; return1 <= maxCars-loc1; return1++ {
						return1Prob := cutProb(return1Dist, return1, maxCars-loc1)
						for return2 := 0; return2 <= maxCars-loc2; return2++ {
							return2Prob := cutProb(return2Dist, return2, maxCars-loc2)
							newState := State{
								loc1 - rent1 + return1,
								loc2 - rent2 + return2,
							}
							r := reward + 10*float64(rent1+rent2)
							prob := rent1Prob * rent2Prob * return1Prob * return2Prob
							sa := StateAction{s, a}
							srp := StateRewardProb{newState, r, prob}
							p[sa] = append(p[sa], srp)
						}
					}
				}
			}
		}
	}
	return p
}

func eval(states []State, p map[StateAction]StateRewardProbs, V []float64, pi []int, theta float64) {
	for delta := 0.0; delta < theta; {
		delta = 0
		for _, state := range states {
			v := V[state.toIndex()]
			a := pi[state.toIndex()]
			new := 0.0
			for _, srp := range (p[StateAction{state, a}]) {
				new += srp.prob * (srp.r + gamma*V[srp.s.toIndex()])
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
		for _, srp := range (p[StateAction{state, oldAction}]) {
			oldActionValue += srp.prob * (srp.r + gamma*V[srp.s.toIndex()])
		}
		actions := state.getActions()
		max := oldActionValue
		argmax := oldAction
		for _, a := range actions {
			actionValue := 0.0
			for _, srp := range (p[StateAction{state, a}]) {
				actionValue += srp.prob * (srp.r + gamma*V[srp.s.toIndex()])
			}
			if actionValue > max {
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
	fmt.Println(V)
	fmt.Println(pi)
}
