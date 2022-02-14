use std::iter::Iterator;

#[derive(Clone)]
pub enum Process { Base, Fixed }

#[derive(Clone)]
pub enum SampleMethod { Categorical, GumbelSoftmax }

#[derive(Clone)]
pub struct Config {
    pub alpha: f64,
    pub n_iters: usize,
    pub array_size: usize,
    pub beta: usize,
    pub sample_method: SampleMethod,
    pub process: Process,
    pub ind_var: f64,
}

fn log_range(lo: f64, hi: f64, n: i64) -> Box<dyn Iterator<Item=f64>> {
    let e_lo = lo.log2();
    let e_hi = hi.log2();
    Box::new(
        (0..(n + 1)).map(move |i| (2.0 as f64).powf(e_lo + (i as f64) * (e_hi - e_lo) / n as f64))
    )
}

pub fn get_configs(name: String) -> Vec<Config> {
    match name.as_str() {
        "beta" => beta(),
        "beta_fixed" => beta_fixed(),
        "alpha" => alpha(),
        "alpha_fixed" => alpha_fixed(),
        "n_iters" => n_iters(),
        _ => panic!("Could not find config gen {}", name),
    }
}

fn beta() -> Vec<Config> {
    log_range(1e0, 3e1, 1000).map(|beta| 
        Config {
            alpha: 1e0,
            n_iters: 1e6 as usize,
            array_size: 0x1000,
            beta: beta as usize,
            sample_method: SampleMethod::Categorical,
            process: Process::Base,
            ind_var: beta,
        }
    ).collect()
}

fn beta_fixed() -> Vec<Config> {
    log_range(1e0, 1e3, 1000).map(|beta| 
        Config {
            alpha: 1e-2,
            n_iters: 1e4 as usize,
            array_size: 0x10,
            beta: beta as usize,
            sample_method: SampleMethod::Categorical,
            process: Process::Fixed,
            ind_var: beta,
        }
    ).collect()
}

fn alpha() -> Vec<Config> {
    log_range(1e-1, 1e1, 1000).map(|x| 
        Config {
            alpha: x, 
            n_iters: 10_000,
            array_size: 0x1000,
            beta: 100,
            sample_method: SampleMethod::Categorical,
            process: Process::Base,
            ind_var: x,
        }
    ).collect()
}

fn alpha_fixed() -> Vec<Config> {
    log_range(1e-3, 1e3, 1000).map(|x| 
        Config {
            alpha: x, 
            n_iters: 10_000,
            array_size: 0x40,
            beta: 100,
            sample_method: SampleMethod::Categorical,
            process: Process::Fixed,
            ind_var: x,
        }
    ).collect()
}

fn n_iters() -> Vec<Config> {
    log_range(1e1, 1e5, 1000).map(|x| 
        Config {
            alpha: 1e-1, 
            n_iters: x as usize,
            array_size: 0x60,
            beta: 10,
            sample_method: SampleMethod::Categorical,
            process: Process::Fixed,
            ind_var: x,
        }
    ).collect()
}


