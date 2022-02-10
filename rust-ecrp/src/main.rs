use rand::prelude::*;
use rand::distributions::Uniform;
use rayon::prelude::*;

// const ALPHA: f64 = 5.0;
// const N_TRIALS: usize = 1000;
// const N_ITERS: usize = 1000;
// const ARRAY_SIZE: usize = N_ITERS;

const ALPHA: f64 = 1e-3;
const N_TRIALS: usize = 200;
const ARRAY_SIZE: usize = 0x40;
const N_ITERS: usize = 4 * ARRAY_SIZE;

const BETA_EXP_MIN: f64 = 0.0;
const BETA_EXP_MAX: f64 = 4.0;

fn sample_categorical(
    w: & Vec<f64>,
    n: f64,
    dist: & Uniform<f64>,
    rng: &mut ThreadRng
) -> usize {
    let mut x = n * dist.sample(rng);
    let mut i = 0;
    x -= w[i];
    while x > 0.0 {
        i += 1;
        x -= w[i];
    }
    return i;
}

fn sample_softmax(
    w: & Vec<f64>,
    n: f64,
    dist: & Uniform<f64>,
    rng: &mut ThreadRng
) -> usize {
    let w = w.iter().map(|x| -(-dist.sample(rng).ln()).ln() + x.log2());
    w.enumerate().reduce(|x, y| if x.1 > y.1 { x } else { y }).unwrap().0
}

fn get_entropy(w: & Vec<f64>) -> f64 {
    let n: f64 = w.iter().sum();
    -w.iter()
        .filter(|&&x| x > 0.0)
        .map(|x| (x / n).log2() * x / n).sum::<f64>()
}

// The ECRP should be in some capacity explanatory and in some capacity predictive.

fn ecrp_fixed_buf(beta: u32) -> f64 {
    // let mut weights: Vec<f64> = Vec::with_capacity(ARRAY_SIZE);
    let mut weights: Vec<f64> = vec![ALPHA; ARRAY_SIZE]; 
    let mut addend_idxs: Vec<usize> = Vec::with_capacity(beta as usize);
    addend_idxs.resize(beta as usize, 0);
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    for iter in 0..N_ITERS {
        // let mut weights_len = weights.len();
        // if weights[weights_len - 1] == 0.0 {
        //     weights[weights_len - 1] = ALPHA;
        // } else {
        //     weights.push(ALPHA);
        //     weights_len += 1;
        // }
        for b in 0..beta {
            // let idx = sample_categorical(&weights, (iter as f64) + ALPHA * ARRAY_SIZE as f64, &dist, &mut rng);
            let idx = sample_softmax(&weights, (iter as f64) + ALPHA * ARRAY_SIZE as f64, &dist, &mut rng);
            addend_idxs[b as usize] = idx;
        }
        // weights[weights_len - 1] -= ALPHA;
        addend_idxs.iter().for_each(|idx| weights[*idx] += 1.0 / (beta as f64));
    }

    get_entropy(&weights)
}

fn ecrp(beta: u32) -> f64 {
    let mut weights: Vec<f64> = Vec::with_capacity(ARRAY_SIZE);
    weights.push(0.0);
    let mut addend_idxs: Vec<usize> = Vec::with_capacity(beta as usize);
    addend_idxs.resize(beta as usize, 0);
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    for iter in 0..N_ITERS {
        let mut weights_len = weights.len();
        if weights[weights_len - 1] == 0.0 {
            weights[weights_len - 1] = ALPHA;
        } else {
            weights.push(ALPHA);
            weights_len += 1;
        }
        for b in 0..beta {
            // let idx = sample_categorical(&weights, (iter as f64) + ALPHA, &dist, &mut rng);
            let idx = sample_softmax(&weights, (iter as f64) + ALPHA, &dist, &mut rng);
            addend_idxs[b as usize] = idx;
        }
        weights[weights_len - 1] -= ALPHA;
        addend_idxs.iter().for_each(|idx| weights[*idx] += 1.0 / (beta as f64));
    }

    // println!("{:?}", &weights[10..30]);
    get_entropy(&weights)
}

fn main() {

    // let foo: [f64; 4] = [0.0, 1.0, 2.0, 4.0];
    // println!("{:?}", foo.iter().filter_map(|x| x.log2()).collect::<Vec<_>>());
    // return;

    let results: Vec<_> = (0..N_TRIALS).into_par_iter().map(|i| {
    // let results: Vec<_> = (0..N_TRIALS).map(|i| {
        let x = BETA_EXP_MIN + (BETA_EXP_MAX - BETA_EXP_MIN) * (i as f64) / (N_TRIALS as f64);
        let beta = (10.0_f64).powf(x);
        let beta_int = beta as u32;
        // (beta, ecrp(beta_int))
        (beta, ecrp_fixed_buf(beta_int))
    }).collect();
    results.iter().for_each(|(x, y)| {
        println!("{},{}", x, y);
    });
}
