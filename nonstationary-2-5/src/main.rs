use rand::distributions::{Distribution, Normal, Uniform};
use gnuplot::{Figure, Caption};
use std::collections::HashMap;
use gnuplot::AxesCommon;
use std::f64::NEG_INFINITY;

const NUM_ARMS: usize = 10;
const EPSILON: f64 = 0.1;
const STEPS: usize = 2000;
const RUNS: usize = 10;


struct Arm {
    mean: f64,
    variance: f64,
    change_distribution: Normal,
}

impl Arm {
    // Arms give reward according to a normal distribution. The mean of this distribution can change.
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
    // Players can choose an arm based on their inner state.
    // Estimations are updated based on the rule Q_{n + 1} = Q_n + \alpha(R_n - Q_n).
    fn choose(&self) -> usize;
    fn update_inner_state(&mut self, chosen_arm: usize, reward: f64, optimal:bool);
}

fn max_index(v: &Vec<f64>) -> usize {
    // Used for choosing the best arm based on a vector of reward estimations.
    let mut max_value = NEG_INFINITY ;
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
    estimations: Vec<f64>, // Contains estimations of the mean rewards for diffrent arms.
    alpha: f64, // step size
    greedy_distribution: Uniform<f64>, // Used to determine if the player chooses the greedy or
    // a random action.
    sum_rewards: f64, // Sum of the already collected rewards.
    average_rewards: Vec<f64>, // Average reward/step until now for each time step.
    num_optimal_choices: f64, // Number of times when the optimal arm was choosen. It is a float,
    // because we want to divide it by step and get a float.
    optimal_choice_ratio: Vec<f64>, // Average num_optimal_choices/step until now for each time step.
    step: usize,
    name: &'static str
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
    let mut all_reward_results =HashMap::new();
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
            all_reward_results.entry(player.name).or_insert(vec![0.0; STEPS]);
            for step in 0..STEPS {
                let mut max_mean = NEG_INFINITY;
                for a in &mut arms {
                    a.change_mean();
                    max_mean = max_mean.max(a.mean); //max_mean contains the mean reward of the optimal arm
                }
                let chosen_arm_index = player.choose();
                let chosen_arm = &arms[chosen_arm_index];
                let reward = chosen_arm.get_reward();
                let optimal = chosen_arm.mean == max_mean;
                player.update_inner_state(chosen_arm_index, reward, optimal);
                // TODO: Remove unwraps.
                // println!("{:?}", player.average_rewards);
                // println!("aaaaaaaa{:?}", all_reward_results);
                all_reward_results.get_mut(player.name).unwrap()[step] += player.average_rewards[step]
            }
        }
    }

    let mut fg = Figure::new();
    let mut x = vec![];
    for i in 0..STEPS {
        x.push(i);
    }
    // println!("{:?}", player.average_rewards);
    for (player_name, mut results) in all_reward_results {
        for res in &mut results{
            *res = *res / (RUNS as f64);
        }
        // println!("{:?}", results);
        fg.axes2d()
        .lines(&x, &results, &[Caption(player_name)])
        .set_title("Average rewards", &[])
        .set_x_label("steps", &[])
        .set_y_label("average reward", &[]);
        fg.save_to_png("average.png", 1000, 1000).unwrap();
    }
    // match fg.save_to_png("average.png", 100, 100) {
    //     Ok(_) => {},
    //     Err(e) => println!("Error: {}", e)
    // }

}
