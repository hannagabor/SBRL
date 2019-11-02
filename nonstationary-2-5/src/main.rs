use rand::distributions::{Distribution, Normal, Uniform};
use gnuplot::{Figure, Caption, Color};
use std::collections::HashMap;

const NUM_ARMS: usize = 10;
const EPSILON: f64 = 0.1;
const STEPS: usize = 10000;
const RUNS: usize = 2000;

struct Arm {
    mean: f64,
    variance: f64,
    change_distribution: Normal,
}

impl Arm {
    fn new() -> Arm {
        Arm {
            mean: 0.,
            variance: 1.,
            change_distribution: Normal::new(0., 0.01),
        }
    }

    fn get_reward(&self) -> f64 {
        let normal = Normal::new(self.mean, self.variance);
        normal.sample(&mut rand::thread_rng())
    }

    fn change_mean(&mut self) {
        self.mean += self.change_distribution.sample(&mut rand::thread_rng());
    }
}

trait Player {
    fn choose(&self) -> usize;
    fn update_inner_state(&mut self, chosen_arm: usize, reward: f64, optimal:bool) {}
}

fn max_index(v: &Vec<f64>) -> usize {
    let mut max_value = 0.;
    let mut max_index: usize = 0;
    for (i, &e) in v.iter().enumerate() {
        if max_value < e {
            max_index = i;
            max_value = e;
        }
    }
    max_index
}

struct ConstantPlayer {
    estimations: Vec<f64>,
    alpha: f64,
    greedy_distribution: Uniform<f64>,
    sum_rewards: f64,
    average_rewards: Vec<f64>,
    num_optimal_choices: f64, //we want to divide it and get a float
    optimal_choice_ratio: Vec<f64>,
    step: usize,
    name: &'static str,
    color: &'static str
}

impl ConstantPlayer {
    fn new() -> ConstantPlayer {
        ConstantPlayer {
            estimations: vec![0.; NUM_ARMS],
            alpha: 0.1,
            greedy_distribution: Uniform::new(0., 1.),
            sum_rewards: 0.,
            average_rewards: vec![0.; STEPS],
            num_optimal_choices: 0.,
            optimal_choice_ratio: vec![0.; STEPS],
            step: 0,
            name: "constant",
            color: "black"
        }
    }
}

impl Player for ConstantPlayer {
    fn choose(&self) -> usize {
        if self.greedy_distribution.sample(&mut rand::thread_rng()) < EPSILON {
            Uniform::new(0, NUM_ARMS).sample(&mut rand::thread_rng())
        } else {
            max_index(&self.estimations)
        }
    }
    fn update_inner_state(&mut self, chosen_arm: usize, reward: f64, optimal: bool) {
        self.estimations[chosen_arm] += self.alpha * (reward - self.estimations[chosen_arm]);
        self.sum_rewards += reward;
        let num_steps = (self.step + 1) as f64;
        self.average_rewards[self.step] = self.sum_rewards / num_steps;
        if optimal {
            self.num_optimal_choices += 1.;
        }
        self.optimal_choice_ratio[self.step] = self.num_optimal_choices / num_steps;
        self.step += 1;
    }
}

// struct AveragePlayer {
//     estimations: Vec<f64>,
// }
//
// impl Player for AveragePlayer {
//     fn choose(&self) -> usize {
//     }
// }

fn main() {
    let mut all_results =HashMap::new();
    for _ in 0..RUNS {
        let mut arms: Vec<Arm> = vec![];
        // let average_player =  AveragePlayer::new();
        let constant_player = ConstantPlayer::new();
        // let players = vec![average_player, constant_player];
        let mut players = vec![constant_player];
        for _ in 0..NUM_ARMS {
            let arm = Arm::new();
            arms.push(arm);
        }
        for player in &mut players {
            all_results.entry(player.name).or_insert(vec![0.0; STEPS]);
            for step in 0..STEPS {
                let mut max_mean = std::f64::NEG_INFINITY;
                for a in &mut arms {
                    a.change_mean();
                    max_mean = max_mean.max(a.mean);
                }
                let chosen_arm_index = player.choose();
                let chosen_arm = &arms[chosen_arm_index];
                let reward = chosen_arm.get_reward();
                let optimal = chosen_arm.mean == max_mean;
                player.update_inner_state(chosen_arm_index, reward, optimal);
                all_results[player.name][step] += player.average_rewards.last().unwrap_or("")
            }
        }
    }

    let mut fg = Figure::new();
    let mut x = vec![];
    for i in 0..STEPS {
        x.push(i);
    }
    // println!("{:?}", player.average_rewards);
    for (player_name, results) in all_results {
        for res in &mut results{
            res = *res / RUNS;
        }
        fg.axes2d()
        .lines(&x, &results, &[Caption("average reward")]);
        fg.save_to_png("average.png", 1000, 1000).unwrap();
    }
    // match fg.save_to_png("average.png", 100, 100) {
    //     Ok(_) => {},
    //     Err(e) => println!("Error: {}", e)
    // }

}
