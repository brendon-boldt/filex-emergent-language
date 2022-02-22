//! # Config
//!
//! `config` contains config-generating functions.  Having the functions in rust code can makes the
//! configuration process more flexible, although any changes will have to be compiled into the
//! binary.

use std::iter::Iterator;

/// Variants of the Chinese restaurant process we should use.
#[derive(Clone)]
pub enum Process {
    /// The vanilla CRP which has an infinite number of weights.
    Base,
    /// The default; a process with a fixed number of weights.
    Fixed,
}

#[derive(Clone)]
pub struct Config {
    /// The concentration parameter.
    pub alpha: f64,
    /// How many total iterations to the run the process.
    pub n_iters: usize,
    /// The size of the weights buffer.
    pub array_size: usize,
    /// The number of sample per update (i.e., the inner loop).
    pub beta: usize,
    pub process: Process,
    /// The independent variable which will be outputted for plotting purposes.
    pub ind_var: f64,
}

/// Generate a range that is uniform in log space.
fn log_range(lo: f64, hi: f64, n: i64) -> Box<dyn Iterator<Item = f64>> {
    let e_lo = lo.log2();
    let e_hi = hi.log2();
    // Divide by `n - 1` so the range is inclusive of the upper bound.
    Box::new(
        (0..n)
            .map(move |i| (2.0 as f64).powf(e_lo + (i as f64) * (e_hi - e_lo) / ((n - 1) as f64))),
    )
}

/// Retrieve a vector configurations based on the given string.
pub fn get_configs(name: String) -> Vec<Config> {
    match name.as_str() {
        "beta_infinite" => beta_infinite(),
        "beta" => beta(),
        "alpha" => alpha(),
        "alpha_infinite" => alpha_infinite(),
        "n_iters" => n_iters(),
        "n_params" => n_params(),
        _ => panic!("Could not find config gen {}", name),
    }
}

fn beta() -> Vec<Config> {
    log_range(1e0, 1e3, 1000)
        .map(|beta| Config {
            alpha: 1e-3,
            n_iters: 1e4 as usize,
            array_size: 0x40,
            beta: beta as usize,
            process: Process::Fixed,
            ind_var: beta,
        })
        .collect()
}

fn alpha() -> Vec<Config> {
    log_range(1e-3, 1e3, 1000)
        .map(|x| Config {
            alpha: x,
            n_iters: 1_000,
            array_size: 0x40,
            beta: 10,
            process: Process::Fixed,
            // alpha correlates inversely with learning rate
            ind_var: 1.0 / x,
        })
        .collect()
}

fn n_iters() -> Vec<Config> {
    log_range(1e0, 1e3, 1000)
        .map(|x| Config {
            alpha: 1e0,
            n_iters: x as usize,
            array_size: 0x40,
            beta: 5,
            process: Process::Fixed,
            ind_var: x,
        })
        .collect()
}

fn n_params() -> Vec<Config> {
    log_range(0x8 as f64, 0x100 as f64, 1000)
        .map(|x| Config {
            alpha: 5e-3 * x.floor(),
            n_iters: 10_00,
            array_size: x as usize,
            beta: 10,
            process: Process::Fixed,
            ind_var: x,
        })
        .collect()
}

fn beta_infinite() -> Vec<Config> {
    log_range(1e0, 3e1, 1000)
        .map(|beta| Config {
            alpha: 1e0,
            n_iters: 1e6 as usize,
            array_size: 0x1000,
            beta: beta as usize,
            process: Process::Base,
            ind_var: beta,
        })
        .collect()
}

fn alpha_infinite() -> Vec<Config> {
    log_range(1e-1, 1e1, 1000)
        .map(|x| Config {
            alpha: x,
            n_iters: 10_000,
            array_size: 0x1000,
            beta: 100,
            process: Process::Base,
            ind_var: x,
        })
        .collect()
}
