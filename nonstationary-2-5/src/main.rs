use rand::distributions::{Distribution, Normal};
struct Arm {
    mean: f64,
    variance: f64,
}

impl Arm {
    fn new() -> Arm {
        Arm {
            mean: 0.,
            variance: 1.,
        }
    }

    fn get_reward(&self) -> f64 {
        let normal = Normal::new(self.mean, self.variance);
        normal.sample(&mut rand::thread_rng())
    }

    fn change_mean(&mut self) {}
}

trait Player {
    fn choose(&self) -> usize;
    fn update_estimations(&mut self, arm: Arm, reward:f64){
    }
}

struct ConstantPlayer {
    estimations: Vec<f64>,
}

impl Player for ConstantPlayer {
    fn choose(&self) -> usize {
    }
}


struct AveragePlayer {
    estimations: Vec<f64>,
}

impl Player for AveragePlayer {
    fn choose(&self) -> usize {
    }
}

fn main() {
    const STEPS: i32 = 100;
    let mut arms: Vec<Arm> = vec![];
    let average_player =  AveragePlayer::new();
    let constant_player = ConstantPlayer::new();
    let players = vec![average_player, constant_player];
    for _ in 0..10 {
        let arm = Arm::new();
        arms.push(arm);
    }
    for player in players {
        for _ in 0..STEPS {
            for a in &mut arms {
                a.change_mean();
            }
        }

    }
    println!("Hello, world!");
}
