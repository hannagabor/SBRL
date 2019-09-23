use rand::distributions::{Distribution, Normal, Uniform};

const NUM_ARMS: usize = 10;
const EPSILON: f64 = 0.1;
const STEPS: i32 = 100;

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
    fn update_estimation(&mut self, chosen_arm: usize, reward: f64) {}
}

fn max_index(v: &Vec<f64>) -> usize {
    1
}

struct ConstantPlayer {
    estimations: Vec<f64>,
    alpha: f64,
    change_distribution: Normal,
    greedy_distribution: Uniform<f64>,
}

impl ConstantPlayer {
    fn new() -> ConstantPlayer {
        ConstantPlayer {
            estimations: vec![0.; NUM_ARMS],
            alpha: 0.1,
            change_distribution: Normal::new(0., 0.1),
            greedy_distribution: Uniform::new(0., 1.),
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
    fn update_estimation(&mut self, chosen_arm: usize, reward: f64) {}
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
    let mut arms: Vec<Arm> = vec![];
    // let average_player =  AveragePlayer::new();
    let constant_player = ConstantPlayer::new();
    // let players = vec![average_player, constant_player];
    let players = vec![constant_player];
    for _ in 0..NUM_ARMS {
        let arm = Arm::new();
        arms.push(arm);
    }
    for mut player in players {
        for _ in 0..STEPS {
            for a in &mut arms {
                a.change_mean();
            }
            let chosen_arm = player.choose();
            let reward = arms[chosen_arm].get_reward();
            player.update_estimation(chosen_arm, reward);
        }
    }
}
