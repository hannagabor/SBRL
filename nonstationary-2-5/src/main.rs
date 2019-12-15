use rand::distributions::{Distribution, Normal, Uniform};
use gnuplot::{Figure, Caption, Color};
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

struct Player {
    // Players can choose an arm based on their inner state.
    // Estimations are updated based on the rule Q_{n + 1} = Q_n + \alpha(R_n - Q_n).
    estimations: Vec<f64>, // Contains estimations of the mean rewards for diffrent arms.
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

impl Player {
    fn new(name: &'static str) -> Player {
        Player {
            estimations: vec![0.; NUM_ARMS],
            greedy_distribution: Uniform::new(0., 1.),
            sum_rewards: 0.,
            average_rewards: vec![0.; STEPS],
            num_optimal_choices: 0.,
            optimal_choice_ratio: vec![0.; STEPS],
            step: 0,
            name: name,
        }
    }
    fn alpha(&self) -> f64 {
        // TODO: Make name an enum.
        if self.name == "constant" {
            0.1
        } else if self.name == "average"{
            let num_steps = (self.step + 1) as f64;
            1.0 / num_steps
        }
        else {
            panic!("Unknown player name!")
        }
    }

    fn choose(&self) -> usize {
        if self.greedy_distribution.sample(&mut rand::thread_rng()) < EPSILON {
            Uniform::new(0, NUM_ARMS).sample(&mut rand::thread_rng())
        } else {
            max_index(&self.estimations)
        }
    }
    fn update_inner_state(&mut self, chosen_arm: usize, reward: f64, optimal: bool) {
        self.estimations[chosen_arm] += self.alpha() * (reward - self.estimations[chosen_arm]);
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


fn main() {
    let mut all_reward_results = HashMap::new();
    // TODO: plot optimal ratio.
    for _ in 0..RUNS {
        let mut arms: Vec<Arm> = vec![];
        let constant_player = Player::new("constant");
        let average_player = Player::new("average");
        let mut players = vec![average_player, constant_player];
        for _ in 0..NUM_ARMS {
            let arm = Arm::new();
            arms.push(arm);
        }
        for player in &mut players {
            all_reward_results.entry(player.name).or_insert(vec![0.0; STEPS]); // The ith entry
            // contains the the sum of average rewards. The sum is over different runs. The average is
            // over the steps until i.
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
                all_reward_results.get_mut(player.name).unwrap()[step] += player.average_rewards[step]
            }
        }
    }


    let mut colors = HashMap::new();
    colors.insert("constant", Color("violet"));
    colors.insert("average", Color("cyan"));

    let mut fg = Figure::new();
    let mut x = vec![];
    for i in 0..STEPS {
        x.push(i);
    }
    let mut ax = fg.axes2d();
    for (player_name, mut results) in all_reward_results {
        for res in &mut results{
            *res = *res / (RUNS as f64);
        }
        ax = ax.lines(&x, &results, &[Caption(player_name), colors[player_name]])
        ;
    ax.set_title("Average rewards", &[])
    .set_x_label("steps", &[])
    .set_y_label("average reward", &[]);
    }
    fg.save_to_png("average.png", 1000, 1000).unwrap();
}
