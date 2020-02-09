package main

import (
	"math"
	"math/rand"
)

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
		name:        "optimistic greedy (init value)",
		estimations: estimations,
		alpha:       0.1,
	}
}

type EpsilonGreedyPlayer struct {
	param       float64
	step        int
	name        string
	rewardSum   float64 // Sum of rewards in the second half.
	estimations []float64
	alpha       float64
	epsilon     float64
	chosen      int
}

func (p *EpsilonGreedyPlayer) getParam() float64 {
	return p.param
}

func (p *EpsilonGreedyPlayer) getRewardSum() float64 {
	return p.rewardSum
}

func (p *EpsilonGreedyPlayer) getName() string {
	return p.name
}

func (p *EpsilonGreedyPlayer) choose() int {
	r := rand.Float64()
	if r < p.epsilon {
		p.chosen = rand.Intn(numArms)
	} else {
		p.chosen = maxFloat(p.estimations)
	}
	return p.chosen
}

func (p *EpsilonGreedyPlayer) updateInnerState(reward float64) {
	p.step += 1
	if p.step > numSteps/2 {
		p.rewardSum += reward
	}
	p.estimations[p.chosen] += p.alpha * (reward - p.estimations[p.chosen])
}

func NewEpsilonGreedyPlayer(epsilon float64) *EpsilonGreedyPlayer {
	estimations := make([]float64, numArms, numArms)
	return &EpsilonGreedyPlayer{
		param:       epsilon,
		name:        "epsilon-greedy (epsilon)",
		estimations: estimations,
		alpha:       0.1,
		epsilon:     epsilon,
	}
}
