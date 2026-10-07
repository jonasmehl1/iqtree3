//! Per-site posterior sampling of the PCA coordinates with NUTS (nuts-rs).
//!
//! Mu, the branch lengths and the global rate normalization are shared and kept
//! constant while sampling. Site rates are fixed to 1. With everything shared fixed, the
//! posterior factorizes over alignment columns, so every column gets its own
//! independent 19-dimensional NUTS chain.
//!
//! Before the final sampling, Mu and the branch lengths are fitted with Monte Carlo EM,
//! starting from the two step light PMSF estimate with Mu fitted as well (see sample_internal).

use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use candle_core::{Tensor, Var};
use candle_nn::Optimizer;
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
    rand::{SeedableRng, rngs::ChaCha8Rng},
};
use phylo_grad::{
    FelsensteinTree,
    nalgebra::{SMatrix, SVector},
};
use rayon::prelude::*;

use crate::{
    MutselParams, Verbosity,
    felsenstein::{FelsensteinOp, FelsensteinWithEdgeOp},
    model::{self, calc_rate_matrix},
    optimization::{Mu, Mu_penalty, load_init_R, two_step_light_pmsf},
    pca::PCA,
    utils::tensor_full,
};

const NUM_COMPONENTS: usize = 19;
/// Number of components of the Gaussian mixture prior in data.rs
const GMM_COMPONENTS: usize = 10;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("Could not parse {}={}", name, value)),
        Err(_) => default,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HmcSettings {
    pub num_tune: u64,
    /// Draws after tuning before the effective sample size is checked for the first time
    pub num_draws: u64,
    /// Sampling continues until every pca coordinate of a site reaches this effective sample size
    pub target_ess: usize,
    /// Upper limit of draws per site, even if target_ess is not reached
    pub max_draws: u64,
    /// Number of (approximately independent) samples per site written to the npz file
    pub num_output_samples: usize,
    pub seed: u64,
}

impl HmcSettings {
    /// Reads MUTSEL_HMC_TUNE, MUTSEL_HMC_DRAWS, MUTSEL_HMC_ESS, MUTSEL_HMC_MAX_DRAWS,
    /// MUTSEL_HMC_SAMPLES and MUTSEL_HMC_SEED from the environment.
    pub fn from_env() -> HmcSettings {
        HmcSettings {
            num_tune: env_or("MUTSEL_HMC_TUNE", 400),
            num_draws: env_or("MUTSEL_HMC_DRAWS", 500),
            target_ess: env_or("MUTSEL_HMC_ESS", 100),
            max_draws: env_or("MUTSEL_HMC_MAX_DRAWS", 20000),
            num_output_samples: env_or("MUTSEL_HMC_SAMPLES", 100),
            seed: env_or("MUTSEL_HMC_SEED", 42),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SiteLogpError {
    #[error("non-finite log likelihood")]
    NonFinite,
}

impl LogpError for SiteLogpError {
    // Treated as a divergence by the sampler.
    fn is_recoverable(&self) -> bool {
        true
    }
}

/// Constants of the per-site posterior which are shared by all sites.
struct SharedModel {
    /// PCA components [19, 20], log_pi = components^T * x (up to a constant)
    components: SMatrix<f64, NUM_COMPONENTS, 20>,
    /// Log of the mutation equilibrium (diagonal of Mu)
    log_mutation_equilibrium: SVector<f64, 20>,
    /// Mu with the fixed global rate normalization already applied
    scaled_Mu: SMatrix<f64, 20, 20>,
    /// pi_reg, multiplies the log density of the prior
    prior_strength: f64,
    /// Diagonal Gaussian mixture prior on the pca coordinates (see PCA): per component
    /// ln w_k - sum_i ln(2 pi var_ki) / 2, the means and the inverse variances
    gmm_logc: SVector<f64, GMM_COMPONENTS>,
    gmm_mu: SMatrix<f64, NUM_COMPONENTS, GMM_COMPONENTS>,
    gmm_inv_var: SMatrix<f64, NUM_COMPONENTS, GMM_COMPONENTS>,
}

impl SharedModel {
    fn new(Mu: &Tensor, rate_scaling: f64, pca: &PCA, pi_reg: f64) -> SharedModel {
        let Mu_values = Mu.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let Mu = SMatrix::<f64, 20, 20>::from_row_slice(&Mu_values);
        let components = pca.components.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        // [K, 19] row-major tensors into [19, K] matrices, one column per mixture component
        let per_component = |t: &Tensor| {
            assert_eq!(t.dims(), &[GMM_COMPONENTS, NUM_COMPONENTS]);
            SMatrix::<f64, NUM_COMPONENTS, GMM_COMPONENTS>::from_column_slice(
                &t.flatten_all().unwrap().to_vec1::<f64>().unwrap(),
            )
        };
        SharedModel {
            components: SMatrix::<f64, NUM_COMPONENTS, 20>::from_row_slice(&components),
            log_mutation_equilibrium: Mu.diagonal().map(f64::ln),
            scaled_Mu: Mu * rate_scaling,
            prior_strength: pi_reg,
            gmm_logc: SVector::<f64, GMM_COMPONENTS>::from_column_slice(
                &pca.gmm_logc.flatten_all().unwrap().to_vec1::<f64>().unwrap(),
            ),
            gmm_mu: per_component(&pca.gmm_mu),
            gmm_inv_var: per_component(&pca.gmm_inv_var),
        }
    }
}

impl SharedModel {
    /// Site frequencies for the given pca coordinates
    fn pi(&self, x: &SVector<f64, NUM_COMPONENTS>) -> (SVector<f64, 20>, SVector<f64, 20>) {
        let log_pi = self.components.transpose() * x;
        let max_log_pi = log_pi.max();
        let unnormalized_pi = log_pi.map(|l| (l - max_log_pi).exp());
        (log_pi, unnormalized_pi / unnormalized_pi.sum())
    }

    /// Log prior, S and sqrt_pi for the pca coordinates x
    fn evaluate(&self, x: &SVector<f64, NUM_COMPONENTS>) -> SiteModelEval {
        // pi_reg times the log density of the Gaussian mixture prior on the pca coordinates
        // (negative of PCA::penalty_on_pca_coordinates)
        let mut exponents = SVector::<f64, GMM_COMPONENTS>::zeros();
        let mut weighted_diffs = SMatrix::<f64, NUM_COMPONENTS, GMM_COMPONENTS>::zeros();
        for k in 0..GMM_COMPONENTS {
            // (x - mu_k) / var_k
            let weighted_diff = (x - self.gmm_mu.column(k)).component_mul(&self.gmm_inv_var.column(k));
            exponents[k] = self.gmm_logc[k] - 0.5 * weighted_diff.dot(&(x - self.gmm_mu.column(k)));
            weighted_diffs.set_column(k, &weighted_diff);
        }
        let max_exponent = exponents.max();
        let responsibilities = exponents.map(|e| (e - max_exponent).exp());
        let total = responsibilities.sum();
        let log_prior = self.prior_strength * (max_exponent + total.ln());
        let grad_log_prior = -self.prior_strength * (weighted_diffs * responsibilities) / total;

        // log_pi up to an additive constant, pi = softmax(log_pi)
        let (log_pi, pi) = self.pi(x);
        let sqrt_pi = pi.map(f64::sqrt);
        let fitness = log_pi - self.log_mutation_equilibrium;

        // S_ij = sqrt(pi_i / pi_j) * Mu_ij * g(f_j - f_i), only the upper triangle is read by phylo_grad
        let mut S = SMatrix::<f64, 20, 20>::zeros();
        for i in 0..20 {
            for j in (i + 1)..20 {
                S[(i, j)] = self.scaled_Mu[(i, j)]
                    * (0.5 * (log_pi[i] - log_pi[j])).exp()
                    * fixation(fitness[j] - fitness[i]);
            }
        }

        SiteModelEval {
            log_prior,
            grad_log_prior,
            pi,
            sqrt_pi,
            fitness,
            S,
        }
    }

    /// Gradient of log_likelihood + log_prior w.r.t. x, given the gradients of the log likelihood
    /// w.r.t. the upper triangle of S and w.r.t. sqrt_pi.
    fn backpropagate(
        &self,
        eval: &SiteModelEval,
        grad_S: &SMatrix<f64, 20, 20>,
        grad_sqrt_pi: &SVector<f64, 20>,
    ) -> SVector<f64, NUM_COMPONENTS> {
        let mut grad_log_pi = SVector::<f64, 20>::zeros();
        for i in 0..20 {
            for j in (i + 1)..20 {
                let w = grad_S[(i, j)]
                    * eval.S[(i, j)]
                    * (0.5 - dx_log_fixation(eval.fitness[j] - eval.fitness[i]));
                grad_log_pi[i] += w;
                grad_log_pi[j] -= w;
            }
        }
        // d sqrt_pi_a / d log_pi_b = sqrt_pi_a * (delta_ab - pi_b) / 2
        let weighted = grad_sqrt_pi.component_mul(&eval.sqrt_pi);
        grad_log_pi += (weighted - eval.pi * weighted.sum()) * 0.5;

        eval.grad_log_prior + self.components * grad_log_pi
    }
}

struct SiteModelEval {
    log_prior: f64,
    grad_log_prior: SVector<f64, NUM_COMPONENTS>,
    pi: SVector<f64, 20>,
    sqrt_pi: SVector<f64, 20>,
    fitness: SVector<f64, 20>,
    S: SMatrix<f64, 20, 20>,
}

/// Fixation factor g(x) = x / (1 - exp(-x)), see model::GOp
fn fixation(x: f64) -> f64 {
    if x.abs() < 1e-6 {
        1.0 + x / 2.0 + x * x / 12.0
    } else {
        x / (-(-x).exp_m1())
    }
}

/// d/dx log g(x) = 1/x - 1/(exp(x) - 1)
fn dx_log_fixation(x: f64) -> f64 {
    // The closed form loses about eps / |x| to cancellation, so small arguments use the
    // Bernoulli series 1/2 - sum_k B_2k x^(2k-1) / (2k)!, whose truncation error is below 1e-20 for |x| < 0.1.
    if x.abs() < 0.1 {
        let x2 = x * x;
        0.5 - x * (1.0 / 12.0
            - x2 * (1.0 / 720.0
                - x2 * (1.0 / 30240.0 - x2 * (1.0 / 1209600.0 - x2 / 47900160.0))))
    } else {
        1.0 / x - 1.0 / x.exp_m1()
    }
}

/// Unnormalized log posterior of the PCA coordinates of a single alignment column.
/// Same model as optimization::ModelParameters with site rate 1 and fixed Mu, branch lengths and normalization.
struct SiteLogp {
    shared: Arc<SharedModel>,
    tree: Arc<FelsensteinTree<20>>,
    leaf_pl: Vec<SVector<f64, 20>>,
    pl_buffer: Vec<SVector<f64, 20>>,
}

impl HasDims for SiteLogp {
    fn dim_sizes(&self) -> std::collections::HashMap<String, u64> {
        [("unconstrained_parameter".to_string(), NUM_COMPONENTS as u64)]
            .into_iter()
            .collect()
    }
}

impl CpuLogpFunc for SiteLogp {
    type LogpError = SiteLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        NUM_COMPONENTS
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SiteLogpError> {
        let model = &*self.shared;
        let x = SVector::<f64, NUM_COMPONENTS>::from_column_slice(position);
        let eval = model.evaluate(&x);

        let num_leaves = self.leaf_pl.len();
        self.pl_buffer[..num_leaves].copy_from_slice(&self.leaf_pl);
        let result = self.tree.calculate_gradients_single_side(
            eval.S.as_view(),
            eval.sqrt_pi.as_view(),
            &mut self.pl_buffer,
        );

        let logp = result.log_likelihood + eval.log_prior;
        if !logp.is_finite() {
            return Err(SiteLogpError::NonFinite);
        }

        let grad_x = model.backpropagate(&eval, &result.grad_s, &result.grad_sqrt_pi);
        if grad_x.iter().any(|g| !g.is_finite()) {
            return Err(SiteLogpError::NonFinite);
        }
        gradient.copy_from_slice(grad_x.as_slice());

        Ok(logp)
    }

    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(array.to_vec())
    }
}

struct SiteSamples {
    /// Evenly thinned draws [num_output_samples * 19]
    thinned_draws: Vec<f64>,
    /// Posterior mean of the site frequencies over all draws
    mean_pi: SVector<f64, 20>,
    num_draws: usize,
    /// Minimum over the pca coordinates of the effective sample size of all draws
    ess: f64,
    num_divergences: usize,
    step_size: f64,
    mean_num_steps: f64,
}

/// Effective sample size of a single chain with Geyer's initial monotone sequence estimator.
/// draws: [n * dim] row-major, returns the minimum over the dimensions.
fn min_effective_sample_size(draws: &[f64], dim: usize) -> f64 {
    let n = draws.len() / dim;
    if n < 4 {
        return 0.0;
    }
    let mut min_ess = f64::INFINITY;
    let mut centered = vec![0.0; n];
    for d in 0..dim {
        let mean = (0..n).map(|t| draws[t * dim + d]).sum::<f64>() / n as f64;
        for t in 0..n {
            centered[t] = draws[t * dim + d] - mean;
        }
        let autocov = |lag: usize| -> f64 {
            centered[..n - lag]
                .iter()
                .zip(&centered[lag..])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / n as f64
        };
        let variance = autocov(0);
        if variance <= 0.0 {
            // A coordinate which does not move at all has no information
            return 0.0;
        }
        // tau = -1 + 2 * sum_k P_k, P_k = rho_{2k} + rho_{2k+1}, truncated at the first negative
        // pair and forced to be monotonically decreasing
        let mut sum_pairs = 0.0;
        let mut previous_pair = f64::INFINITY;
        let mut lag = 0;
        while lag + 1 < n {
            let pair = (autocov(lag) + autocov(lag + 1)) / variance;
            if pair <= 0.0 {
                break;
            }
            let pair = pair.min(previous_pair);
            sum_pairs += pair;
            previous_pair = pair;
            lag += 2;
        }
        let tau = (2.0 * sum_pairs - 1.0).max(1.0 / (n as f64).log10());
        min_ess = min_ess.min(n as f64 / tau);
    }
    min_ess
}

fn sample_site(logp: SiteLogp, init: &[f64], settings: &HmcSettings, site_index: usize) -> SiteSamples {
    let shared = Arc::clone(&logp.shared);

    let mut nuts_settings = DiagNutsSettings::default();
    nuts_settings.num_tune = settings.num_tune;
    nuts_settings.num_draws = settings.max_draws;
    nuts_settings.num_chains = 1;
    nuts_settings.seed = settings.seed;

    let mut rng = ChaCha8Rng::seed_from_u64(
        settings.seed ^ (site_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );

    let math = CpuMath::new(logp);
    let mut chain = nuts_settings.new_chain(site_index as u64, math, &mut rng);
    chain
        .set_position(init)
        .unwrap_or_else(|e| panic!("Site {}: could not initialize HMC chain: {:?}", site_index, e));

    let mut draws = Vec::with_capacity(settings.num_draws as usize * NUM_COMPONENTS);
    let mut num_divergences = 0;
    let mut num_steps = 0u64;
    let mut step_size;
    // Enough draws to pick num_output_samples distinct ones when thinning
    let min_draws = (settings.num_draws as usize).max(settings.num_output_samples);
    let mut next_check = min_draws.min(settings.max_draws as usize);
    let mut ess;
    loop {
        let (draw, progress) = chain
            .draw()
            .unwrap_or_else(|e| panic!("Site {}: HMC sampling failed: {:?}", site_index, e));
        if progress.tuning {
            continue;
        }
        draws.extend_from_slice(&draw);
        num_divergences += progress.diverging as usize;
        num_steps += progress.num_steps;
        step_size = progress.step_size;

        let num_draws = draws.len() / NUM_COMPONENTS;
        if num_draws < next_check {
            continue;
        }
        ess = min_effective_sample_size(&draws, NUM_COMPONENTS);
        if ess >= settings.target_ess as f64 || num_draws as u64 >= settings.max_draws {
            break;
        }
        // Extrapolate the number of draws needed, with some margin
        let needed = (num_draws as f64 * 1.2 * settings.target_ess as f64 / ess.max(1.0)) as usize;
        next_check = needed
            .clamp(num_draws + num_draws / 4 + 1, 2 * num_draws)
            .min(settings.max_draws as usize);
    }
    let num_draws = draws.len() / NUM_COMPONENTS;

    let mut thinned_draws = Vec::with_capacity(settings.num_output_samples * NUM_COMPONENTS);
    for i in 0..settings.num_output_samples {
        // Last draw of each of num_output_samples equally sized blocks
        let draw = ((i + 1) * num_draws) / settings.num_output_samples - 1;
        thinned_draws.extend_from_slice(&draws[draw * NUM_COMPONENTS..(draw + 1) * NUM_COMPONENTS]);
    }

    let mut mean_pi = SVector::<f64, 20>::zeros();
    for draw in draws.chunks_exact(NUM_COMPONENTS) {
        mean_pi += shared.pi(&SVector::<f64, NUM_COMPONENTS>::from_column_slice(draw)).1;
    }
    mean_pi /= num_draws as f64;

    SiteSamples {
        thinned_draws,
        mean_pi,
        num_draws,
        ess,
        num_divergences,
        step_size,
        mean_num_steps: num_steps as f64 / num_draws.max(1) as f64,
    }
}

fn leaf_partial_likelihoods(alignment: &[u8], site: usize, num_leaves: usize) -> Vec<SVector<f64, 20>> {
    (0..num_leaves)
        .map(|seq| {
            let residue = alignment[site * num_leaves + seq];
            if residue < 20 {
                let mut v = SVector::<f64, 20>::zeros();
                v[residue as usize] = 1.0;
                v
            } else {
                SVector::<f64, 20>::from_element(1.0)
            }
        })
        .collect()
}

/// Settings of the Monte Carlo EM for Mu and the branch lengths
#[derive(Debug, Clone, Copy)]
pub struct EmSettings {
    /// EM iterations (E-step and M-step) before the final sampling, 0 keeps the PMSF Mu and branch lengths
    pub iterations: usize,
    /// Posterior samples per site the M-step averages the log likelihood over
    pub num_samples: usize,
    /// NUTS tuning draws per site in the first E-step
    pub num_tune: u64,
    /// NUTS tuning draws per site in the later E-steps, whose chains start at the previous draws
    pub num_retune: u64,
    /// Maximum number of optimizer steps per M-step
    pub max_mstep_iterations: usize,
}

impl EmSettings {
    /// Reads MUTSEL_EM_ITERATIONS, MUTSEL_EM_SAMPLES, MUTSEL_EM_TUNE, MUTSEL_EM_RETUNE and
    /// MUTSEL_EM_MSTEP_ITERATIONS from the environment.
    pub fn from_env() -> EmSettings {
        EmSettings {
            iterations: env_or("MUTSEL_EM_ITERATIONS", 5),
            num_samples: env_or("MUTSEL_EM_SAMPLES", 8),
            num_tune: env_or("MUTSEL_EM_TUNE", 200),
            num_retune: env_or("MUTSEL_EM_RETUNE", 50),
            max_mstep_iterations: env_or("MUTSEL_EM_MSTEP_ITERATIONS", 200),
        }
    }

    /// NUTS settings of the E-step: about num_samples independent samples per site
    fn hmc_settings(&self, settings: &HmcSettings, iteration: usize) -> HmcSettings {
        HmcSettings {
            num_tune: if iteration == 0 { self.num_tune } else { self.num_retune },
            num_draws: (10 * self.num_samples).max(50) as u64,
            target_ess: self.num_samples,
            max_draws: settings.max_draws,
            num_output_samples: self.num_samples,
            seed: settings.seed.wrapping_add(iteration as u64 + 1),
        }
    }
}

/// Runs an independent NUTS chain for every site, starting at init[site]
fn sample_sites(
    shared: &Arc<SharedModel>,
    tree: &Arc<FelsensteinTree<20>>,
    alignment: &[u8],
    num_leaves: usize,
    init: &[Vec<f64>],
    settings: &HmcSettings,
    verbosity: Verbosity,
    label: &str,
) -> Vec<SiteSamples> {
    let num_sites = init.len();
    let num_nodes = tree.num_nodes();
    let start = std::time::Instant::now();
    let num_done = AtomicUsize::new(0);
    let report_every = (num_sites / 10).max(1);

    let site_samples: Vec<SiteSamples> = (0..num_sites)
        .into_par_iter()
        .map(|site| {
            let logp = SiteLogp {
                shared: Arc::clone(shared),
                tree: Arc::clone(tree),
                leaf_pl: leaf_partial_likelihoods(alignment, site, num_leaves),
                pl_buffer: vec![SVector::<f64, 20>::zeros(); num_nodes],
            };
            let samples = sample_site(logp, &init[site], settings, site);

            let done = num_done.fetch_add(1, Ordering::Relaxed) + 1;
            if verbosity.should_print(Verbosity::Min) && (done % report_every == 0 || done == num_sites) {
                println!(
                    "MUTSEL HMC {}: {}/{} sites sampled ({:.1}s)",
                    label,
                    done,
                    num_sites,
                    start.elapsed().as_secs_f64()
                );
            }
            samples
        })
        .collect();

    let total_divergences: usize = site_samples.iter().map(|s| s.num_divergences).sum();
    let sites_with_divergences = site_samples.iter().filter(|s| s.num_divergences > 0).count();
    let total_draws: usize = site_samples.iter().map(|s| s.num_draws).sum();
    let mean_steps = site_samples
        .iter()
        .map(|s| s.mean_num_steps * s.num_draws as f64)
        .sum::<f64>()
        / total_draws.max(1) as f64;
    let min_ess = site_samples.iter().map(|s| s.ess).fold(f64::INFINITY, f64::min);
    let sites_below_target = site_samples
        .iter()
        .filter(|s| s.ess < settings.target_ess as f64)
        .count();
    println!(
        "MUTSEL HMC {}: done in {:.1}s, {} draws per site on average, {} divergent transitions in {} sites, mean {:.1} leapfrog steps per draw",
        label,
        start.elapsed().as_secs_f64(),
        total_draws / num_sites.max(1),
        total_divergences,
        sites_with_divergences,
        mean_steps
    );
    println!(
        "MUTSEL HMC {}: minimum ESS over sites {:.0}, {} sites below the target ESS {} after {} draws",
        label, min_ess, sites_below_target, settings.target_ess, settings.max_draws
    );
    site_samples
}

/// log pi [num_sites, 20] (up to a constant per site) for each of the thinned draws
fn thinned_log_pi(site_samples: &[SiteSamples], num_samples: usize, pca: &PCA) -> Result<Vec<Tensor>, candle_core::Error> {
    (0..num_samples)
        .map(|k| {
            let coordinates: Vec<f64> = site_samples
                .iter()
                .flat_map(|s| s.thinned_draws[k * NUM_COMPONENTS..(k + 1) * NUM_COMPONENTS].iter().copied())
                .collect();
            let coordinates =
                Tensor::from_vec(coordinates, &[site_samples.len(), NUM_COMPONENTS], &candle_core::Device::Cpu)?;
            Ok(pca.pca_coordinates_to_log_freq(&coordinates))
        })
        .collect()
}

/// Average substitution rate over all sites and samples, without normalization
fn mean_substitution_rate(Mu: &Tensor, log_pi_samples: &[Tensor]) -> Result<f64, candle_core::Error> {
    let mut sum = 0.0;
    for log_pi in log_pi_samples {
        let (S, sqrt_pi) = calc_rate_matrix(Mu, log_pi, &tensor_full(1.0, &[]));
        sum += model::substitution_rates_tensor(&S, &sqrt_pi).mean_all()?.to_scalar::<f64>()?;
    }
    Ok(sum / log_pi_samples.len() as f64)
}

/// M-step: maximizes the log likelihood averaged over the posterior samples of the site frequencies
/// minus the penalty on Mu (as in optimization::ModelParameters),
/// with the rate normalization fixed. Returns the averaged log likelihood at the optimum.
fn m_step(
    op: &FelsensteinWithEdgeOp,
    log_R: &Var,
    init_log_R: &Tensor,
    log_branch_lengths: &Var,
    log_pi_samples: &[Tensor],
    rate_scaling: f64,
    mutsel_params: MutselParams,
    max_iterations: usize,
    verbosity: Verbosity,
) -> Result<f64, candle_core::Error> {
    const MIN_ITERATIONS: usize = 10;
    const MIN_REL_IMPROVEMENT: f64 = 1e-6;
    const NO_IMPROVE_PATIENCE: usize = 5;

    let variables = vec![log_R.clone(), log_branch_lengths.clone()];
    let mut opt = candle_nn::optim::AdamW::new(
        variables.clone(),
        candle_nn::optim::ParamsAdamW {
            lr: 0.03,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    let rate_scaling = tensor_full(rate_scaling, &[]);
    let num_samples = log_pi_samples.len() as f64;

    let mut best_objective = f64::INFINITY;
    let mut best_log_likelihood = f64::NAN;
    let mut best_values = variables
        .iter()
        .map(|v| v.as_tensor().copy())
        .collect::<Result<Vec<_>, _>>()?;
    let mut no_improve_count = 0;
    for iteration in 0..max_iterations {
        let penalty = Mu_penalty(log_R, init_log_R, mutsel_params.Mu_reg);
        // The penalty's GradStore receives the gradient of the whole objective
        let mut grads = penalty.backward()?;
        let mut summed_grads = variables
            .iter()
            .map(|v| match grads.get(v.as_tensor()) {
                Some(g) => Ok(g.clone()),
                None => v.zeros_like(),
            })
            .collect::<Result<Vec<_>, _>>()?;

        // One backward pass per sample, so only one graph over all sites is alive at a time
        let mut log_likelihood = 0.0;
        for log_pi in log_pi_samples {
            let (S, sqrt_pi) = calc_rate_matrix(&Mu(log_R), log_pi, &rate_scaling);
            let sample_log_likelihood = S.apply_op3(&sqrt_pi, &log_branch_lengths.exp()?, op.clone())?;
            log_likelihood += sample_log_likelihood.to_scalar::<f64>()? / num_samples;
            let sample_grads = sample_log_likelihood.affine(-1.0 / num_samples, 0.0)?.backward()?;
            for (sum, v) in summed_grads.iter_mut().zip(&variables) {
                if let Some(g) = sample_grads.get(v.as_tensor()) {
                    *sum = (&*sum + g)?;
                }
            }
        }

        let objective = penalty.to_scalar::<f64>()? - log_likelihood;
        if verbosity.should_print(Verbosity::Med) {
            println!(
                "MUTSEL HMC M-step iteration {}: expected log likelihood {:.3}, objective {:.3}",
                iteration, log_likelihood, objective
            );
        }
        let rel_improvement = (best_objective - objective) / best_objective.abs().max(1e-12);
        if objective < best_objective {
            best_objective = objective;
            best_log_likelihood = log_likelihood;
            best_values = variables
                .iter()
                .map(|v| v.as_tensor().copy())
                .collect::<Result<Vec<_>, _>>()?;
        }
        if iteration >= MIN_ITERATIONS {
            if rel_improvement > MIN_REL_IMPROVEMENT {
                no_improve_count = 0;
            } else {
                no_improve_count += 1;
            }
            if no_improve_count >= NO_IMPROVE_PATIENCE {
                if verbosity.should_print(Verbosity::Med) {
                    println!("MUTSEL HMC M-step: converged after {} iterations", iteration);
                }
                break;
            }
        }

        for (sum, v) in summed_grads.into_iter().zip(&variables) {
            grads.insert(v.as_tensor(), sum);
        }
        opt.step(&grads)?;
    }

    for (variable, value) in variables.iter().zip(best_values.iter()) {
        variable.set(value)?;
    }
    Ok(best_log_likelihood)
}

/// Samples the PCA coordinates of every site from its posterior and returns (S, sqrt_pi)
/// evaluated at the posterior mean of the site frequencies.
///
/// Mu, the branch lengths and the initial site positions come from the two step light PMSF procedure
/// with Mu fitted together with the branch lengths. They are then refined with Monte Carlo EM: the
/// E-step samples the sites given Mu and the branch lengths, the M-step maximizes the log likelihood
/// averaged over these samples plus the log prior of Mu. This converges to their MAP estimate with
/// the site frequencies integrated out. The rate normalization is kept fixed within each iteration and
/// afterwards reset to an average substitution rate of 1 over the posterior samples, rescaling the
/// branch lengths so that the likelihood stays the same.
pub fn sample_internal(
    parents: &[i32],
    branch_lengths: &[f64],
    alignment: &[u8],
    num_sites: usize,
    num_leaves: usize,
    mutsel_params: MutselParams,
    prior_R_file: Option<&Path>,
    verbosity: Verbosity,
    out_prefix: &str,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let settings = HmcSettings::from_env();
    let em = EmSettings::from_env();
    println!(
        "MUTSEL HMC: sampling pca coordinates per site with NUTS ({} tuning, at least {} draws, until ESS >= {} or {} draws, {} thinned samples per site, seed {}), site rates 1",
        settings.num_tune,
        settings.num_draws,
        settings.target_ess,
        settings.max_draws,
        settings.num_output_samples,
        settings.seed
    );
    println!(
        "MUTSEL HMC: Mu and branch lengths from the two step PMSF estimate refined by {} Monte Carlo EM iterations ({} samples per site, {} tuning draws in the first and {} in later iterations, at most {} M-step iterations)",
        em.iterations,
        em.num_samples,
        em.num_tune,
        em.num_retune,
        em.max_mstep_iterations
    );
    assert!(settings.num_output_samples >= 1 && (em.iterations == 0 || em.num_samples >= 1));

    let felsenstein = crate::create_felsenstein_tree(parents, branch_lengths, alignment, num_sites, num_leaves);
    let op = FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));
    let edge_op = op.into_with_edge_op();
    let pca = Arc::new(PCA::new());
    // The Mu penalty stays relative to the prior Mu
    let init_log_R = load_init_R(prior_R_file)?.log()?;

    // Starting values: PMSF site frequencies, with the branch lengths and Mu fitted to them
    let pmsf = two_step_light_pmsf(
        op,
        crate::data::UDM256,
        crate::data::UDM256_WEIGHTS,
        &Tensor::from_slice(branch_lengths, &[branch_lengths.len()], &candle_core::Device::Cpu)?.log()?,
        &init_log_R,
        true,
        mutsel_params.Mu_reg,
        verbosity,
        out_prefix,
    );
    let init_pca_tensor = pca.log_freq_to_pca_coordinates(&pmsf.site_freq.log()?);
    let average_rate = pmsf.average_rate;
    let log_R = pmsf.log_R;
    let log_branch_lengths = pmsf.log_branch_lengths;
    let log_R = Var::from_tensor(&log_R)?;
    let log_branch_lengths = Var::from_tensor(&log_branch_lengths)?;

    // Keep the normalization the branch lengths were estimated with, so they stay valid
    let mut rate_scaling = 1.0 / average_rate;
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL HMC: starting from the two step PMSF estimate, tree length {:.4}, rate normalization 1/{:.5}, max relative change of Mu from the prior {:.4}",
            log_branch_lengths.exp()?.sum_all()?.to_scalar::<f64>()?,
            average_rate,
            {
                let init_Mu = Mu(&init_log_R).detach();
                ((Mu(log_R.as_tensor()).detach() - &init_Mu)?.abs()? / &init_Mu)?
                    .max_all()?
                    .to_scalar::<f64>()?
            }
        );
    }

    let mut positions = init_pca_tensor.to_vec2::<f64>()?;
    let mut em_log_likelihood = Vec::with_capacity(em.iterations);
    let mut em_tree_length = Vec::with_capacity(em.iterations);
    for iteration in 0..em.iterations {
        let label = format!("EM iteration {}/{}", iteration + 1, em.iterations);
        let old_Mu = Mu(log_R.as_tensor()).detach();
        let shared = Arc::new(SharedModel::new(&old_Mu, rate_scaling, &pca, mutsel_params.pi_reg));
        let tree = Arc::new(FelsensteinTree::<20>::new(
            parents,
            &log_branch_lengths.exp()?.to_vec1::<f64>()?,
        ));
        let em_settings = em.hmc_settings(&settings, iteration);
        let site_samples = sample_sites(&shared, &tree, alignment, num_leaves, &positions, &em_settings, verbosity, &label);

        // Next chains start at the last draw, which is the last thinned draw
        let last = (em.num_samples - 1) * NUM_COMPONENTS;
        positions = site_samples
            .iter()
            .map(|s| s.thinned_draws[last..last + NUM_COMPONENTS].to_vec())
            .collect();

        let log_pi_samples = thinned_log_pi(&site_samples, em.num_samples, &pca)?;
        drop(site_samples);
        let log_likelihood = m_step(
            &edge_op,
            &log_R,
            &init_log_R,
            &log_branch_lengths,
            &log_pi_samples,
            rate_scaling,
            mutsel_params,
            em.max_mstep_iterations,
            verbosity,
        )?;

        let new_Mu = Mu(log_R.as_tensor()).detach();
        let new_rate_scaling = 1.0 / mean_substitution_rate(&new_Mu, &log_pi_samples)?;
        // Rescale the branch lengths so the likelihood is unchanged with the new normalization
        log_branch_lengths.set(&(log_branch_lengths.as_tensor() + (rate_scaling / new_rate_scaling).ln())?)?;
        rate_scaling = new_rate_scaling;

        let tree_length = log_branch_lengths.exp()?.sum_all()?.to_scalar::<f64>()?;
        let max_rel_Mu_change = ((&new_Mu - &old_Mu)?.abs()? / &old_Mu)?
            .max_all()?
            .to_scalar::<f64>()?;
        em_log_likelihood.push(log_likelihood);
        em_tree_length.push(tree_length);
        if verbosity.should_print(Verbosity::Min) {
            println!(
                "MUTSEL HMC {}: expected log likelihood {:.3}, tree length {:.4}, rate normalization 1/{:.5}, max relative change of Mu {:.4}",
                label,
                log_likelihood,
                tree_length,
                1.0 / rate_scaling,
                max_rel_Mu_change
            );
        }
    }

    let Mu = Mu(log_R.as_tensor()).detach();
    let shared = Arc::new(SharedModel::new(&Mu, rate_scaling, &pca, mutsel_params.pi_reg));
    let branch_lengths = log_branch_lengths.exp()?;
    let tree = Arc::new(FelsensteinTree::<20>::new(parents, &branch_lengths.to_vec1::<f64>()?));
    let site_samples = sample_sites(&shared, &tree, alignment, num_leaves, &positions, &settings, verbosity, "final");
    let rate_scaling = tensor_full(rate_scaling, &[]);

    // Tensor::write_npz does not support zip entries or files of 4 GiB or more, so pca_samples and
    // pi_samples together are kept below that by writing an evenly spaced subset of the thinned draws.
    const MAX_SAMPLE_BYTES: usize = (1 << 32) - (1 << 26);
    let bytes_per_sample = num_sites * (NUM_COMPONENTS + 20) * std::mem::size_of::<f64>();
    let num_output_samples = settings
        .num_output_samples
        .min(MAX_SAMPLE_BYTES / bytes_per_sample.max(1))
        .max(1);
    if num_output_samples < settings.num_output_samples {
        println!(
            "MUTSEL HMC: WARNING: writing only {} of the {} samples per site, the npz file would exceed 4 GiB",
            num_output_samples, settings.num_output_samples
        );
    }
    // pca_samples [num_sites, num_output_samples, 19]
    let pca_samples: Vec<f64> = site_samples
        .iter()
        .flat_map(|s| {
            (0..num_output_samples).flat_map(move |i| {
                let draw = i * settings.num_output_samples / num_output_samples;
                s.thinned_draws[draw * NUM_COMPONENTS..(draw + 1) * NUM_COMPONENTS].iter().copied()
            })
        })
        .collect();
    let pca_samples = Tensor::from_vec(
        pca_samples,
        &[num_sites, num_output_samples, NUM_COMPONENTS],
        &candle_core::Device::Cpu,
    )?;

    // Posterior mean of the site frequencies over all draws
    let mean_pi: Vec<f64> = site_samples.iter().flat_map(|s| s.mean_pi.iter().copied()).collect();
    let mean_pi = Tensor::from_vec(mean_pi, &[num_sites, 20], &candle_core::Device::Cpu)?;

    let (S, sqrt_pi) = calc_rate_matrix(&Mu, &mean_pi.log()?, &rate_scaling);

    let average_rate_mean_pi = model::substitution_rates_tensor(&S, &sqrt_pi)
        .mean_all()?
        .to_scalar::<f64>()?;
    if verbosity.should_print(Verbosity::Min) {
        println!(
            "MUTSEL HMC: average substitution rate at posterior mean frequencies {:.4}",
            average_rate_mean_pi
        );
    }

    let npz_path = format!("{}.hmc.npz", out_prefix);
    Tensor::write_npz(
        &[
            ("pca_samples", &pca_samples),
            (
                "pi_samples",
                &candle_nn::ops::softmax(
                    &pca.pca_coordinates_to_log_freq(
                        &pca_samples.reshape(&[num_sites * num_output_samples, NUM_COMPONENTS])?,
                    ),
                    1,
                )?
                .reshape(&[num_sites, num_output_samples, 20])?,
            ),
            ("init_pca_coordinates", &init_pca_tensor),
            ("mean_pi", &mean_pi),
            ("Mu", &Mu),
            ("rate_scaling", &rate_scaling),
            ("branch_lengths", &branch_lengths),
            ("em_log_likelihood", &Tensor::from_slice(&em_log_likelihood, &[em_log_likelihood.len()], &candle_core::Device::Cpu)?),
            ("em_tree_length", &Tensor::from_slice(&em_tree_length, &[em_tree_length.len()], &candle_core::Device::Cpu)?),
            (
                "ess",
                &Tensor::from_iter(site_samples.iter().map(|s| s.ess), &candle_core::Device::Cpu)?,
            ),
            (
                "num_draws",
                &Tensor::from_iter(site_samples.iter().map(|s| s.num_draws as f64), &candle_core::Device::Cpu)?,
            ),
            (
                "num_divergences",
                &Tensor::from_iter(site_samples.iter().map(|s| s.num_divergences as f64), &candle_core::Device::Cpu)?,
            ),
            (
                "step_size",
                &Tensor::from_iter(site_samples.iter().map(|s| s.step_size), &candle_core::Device::Cpu)?,
            ),
            (
                "mean_num_steps",
                &Tensor::from_iter(site_samples.iter().map(|s| s.mean_num_steps), &candle_core::Device::Cpu)?,
            ),
        ],
        Path::new(&npz_path),
    )?;
    println!("MUTSEL HMC: samples written to {}", npz_path);

    Ok((S, sqrt_pi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    const PI_REG: f64 = 0.32;
    const RATE_SCALING: f64 = 1.3;

    fn test_tree() -> FelsensteinTree<20> {
        // ((0,1),(2,3)) plus a fifth leaf at the root
        let parents = [5, 5, 6, 6, 7, 7, 7, -1];
        let branch_lengths = [0.1, 0.3, 0.2, 0.05, 0.4, 0.15, 0.25, 0.0];
        FelsensteinTree::<20>::new(&parents, &branch_lengths)
    }

    fn test_Mu() -> Tensor {
        let init_R = crate::data::load_lower_R_with_equi(crate::data::M_TXT);
        Mu(&init_R.log().unwrap()).detach()
    }

    const ALIGNMENT: [u8; 5] = [0u8, 0, 3, 20, 7];

    fn test_logp() -> SiteLogp {
        let tree = test_tree();
        SiteLogp {
            shared: Arc::new(SharedModel::new(&test_Mu(), RATE_SCALING, &PCA::new(), PI_REG)),
            leaf_pl: leaf_partial_likelihoods(&ALIGNMENT, 0, 5),
            pl_buffer: vec![SVector::<f64, 20>::zeros(); tree.num_nodes()],
            tree: Arc::new(tree),
        }
    }

    /// Reference implementation with the candle model code used by the ML optimization.
    fn candle_logp(position: &[f64]) -> (f64, Vec<f64>) {
        let pca = PCA::new();
        let x = Var::from_slice(position, &[1, NUM_COMPONENTS], &candle_core::Device::Cpu).unwrap();
        let log_pi = pca.pca_coordinates_to_log_freq(x.as_tensor());
        let (S, sqrt_pi) = calc_rate_matrix(&test_Mu(), &log_pi, &tensor_full(RATE_SCALING, &[]));
        let log_prior = pca.penalty_on_pca_coordinates(x.as_tensor(), PI_REG).neg().unwrap();

        let tree = test_tree();
        let mut felsenstein = tree.clone();
        felsenstein.bind_leaf_pl(vec![leaf_partial_likelihoods(&ALIGNMENT, 0, 5)]);
        let op = crate::felsenstein::FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));
        let ll = S.apply_op2(&sqrt_pi, op).unwrap().sum_all().unwrap();
        let logp = (ll + log_prior).unwrap();
        let grads = logp.backward().unwrap();
        let grad = grads.get(x.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        (logp.to_scalar::<f64>().unwrap(), grad)
    }

    fn test_point() -> Vec<f64> {
        (0..NUM_COMPONENTS).map(|i| 0.3 * ((i as f64) * 1.7).sin()).collect()
    }

    #[test]
    fn matches_candle_model() {
        let mut logp = test_logp();
        let x = test_point();
        let mut grad = vec![0.0; NUM_COMPONENTS];
        let value = logp.logp(&x, &mut grad).unwrap();
        let (ref_value, ref_grad) = candle_logp(&x);
        assert!((value - ref_value).abs() < 1e-13 * ref_value.abs(), "{} vs {}", value, ref_value);
        for i in 0..NUM_COMPONENTS {
            assert!(
                (grad[i] - ref_grad[i]).abs() < 1e-12 * (1.0 + ref_grad[i].abs()),
                "component {}: {} vs {}",
                i,
                grad[i],
                ref_grad[i]
            );
        }
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let mut logp = test_logp();
        let x = test_point();
        let mut grad = vec![0.0; NUM_COMPONENTS];
        let value = logp.logp(&x, &mut grad).unwrap();
        assert!(value.is_finite());

        let eps = 1e-6;
        let mut dummy = vec![0.0; NUM_COMPONENTS];
        for i in 0..NUM_COMPONENTS {
            let mut xp = x.clone();
            xp[i] += eps;
            let mut xm = x.clone();
            xm[i] -= eps;
            let fd = (logp.logp(&xp, &mut dummy).unwrap() - logp.logp(&xm, &mut dummy).unwrap()) / (2.0 * eps);
            assert!(
                (fd - grad[i]).abs() < 1e-5 * (1.0 + fd.abs()),
                "component {}: finite difference {} vs gradient {}",
                i,
                fd,
                grad[i]
            );
        }
    }

    #[test]
    fn effective_sample_size() {
        use nuts_rs::rand::RngExt;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut normal = || {
            let u1: f64 = rng.random::<f64>().max(1e-300);
            let u2: f64 = rng.random();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let n = 20000;
        // Two dimensions: iid and AR(1) with phi = 0.9, which has ESS = n (1 - phi) / (1 + phi)
        let mut draws = Vec::with_capacity(2 * n);
        let mut ar = 0.0;
        for _ in 0..n {
            ar = 0.9 * ar + normal();
            draws.push(normal());
            draws.push(ar);
        }
        let iid: Vec<f64> = draws.iter().step_by(2).copied().collect();
        let iid_ess = min_effective_sample_size(&iid, 1);
        assert!((iid_ess / n as f64 - 1.0).abs() < 0.1, "iid ESS {}", iid_ess);

        let ar_ess = min_effective_sample_size(&draws, 2);
        let expected = n as f64 * 0.1 / 1.9;
        assert!((ar_ess / expected - 1.0).abs() < 0.2, "AR(1) ESS {} vs {}", ar_ess, expected);
    }

    /// Forward mode dual number for exact directional derivatives
    #[derive(Debug, Clone, Copy)]
    struct Dual {
        v: f64,
        d: f64,
    }

    impl Dual {
        fn constant(v: f64) -> Dual {
            Dual { v, d: 0.0 }
        }
        fn exp(self) -> Dual {
            let e = self.v.exp();
            Dual { v: e, d: e * self.d }
        }
        fn exp_m1(self) -> Dual {
            Dual { v: self.v.exp_m1(), d: self.v.exp() * self.d }
        }
        fn ln(self) -> Dual {
            Dual { v: self.v.ln(), d: self.d / self.v }
        }
        fn sqrt(self) -> Dual {
            let r = self.v.sqrt();
            Dual { v: r, d: self.d / (2.0 * r) }
        }
        fn abs(self) -> Dual {
            if self.v < 0.0 { -self } else { self }
        }
    }
    impl std::ops::Add for Dual {
        type Output = Dual;
        fn add(self, o: Dual) -> Dual {
            Dual { v: self.v + o.v, d: self.d + o.d }
        }
    }
    impl std::ops::Sub for Dual {
        type Output = Dual;
        fn sub(self, o: Dual) -> Dual {
            Dual { v: self.v - o.v, d: self.d - o.d }
        }
    }
    impl std::ops::Mul for Dual {
        type Output = Dual;
        fn mul(self, o: Dual) -> Dual {
            Dual { v: self.v * o.v, d: self.d * o.v + self.v * o.d }
        }
    }
    impl std::ops::Div for Dual {
        type Output = Dual;
        fn div(self, o: Dual) -> Dual {
            Dual { v: self.v / o.v, d: (self.d * o.v - self.v * o.d) / (o.v * o.v) }
        }
    }
    impl std::ops::Neg for Dual {
        type Output = Dual;
        fn neg(self) -> Dual {
            Dual { v: -self.v, d: -self.d }
        }
    }

    /// Independent reference of the per-site model following model::calc_rate_matrix and
    /// PCA::penalty_on_pca_coordinates: returns <G, S(x)> + <b, sqrt_pi(x)> + log_prior(x)
    fn reference_surrogate(
        model: &SharedModel,
        x: &[Dual],
        G: &SMatrix<f64, 20, 20>,
        b: &SVector<f64, 20>,
    ) -> Dual {
        let c = Dual::constant;
        let log_pi: Vec<Dual> = (0..20)
            .map(|a| (0..NUM_COMPONENTS).fold(c(0.0), |acc, k| acc + c(model.components[(k, a)]) * x[k]))
            .collect();
        let max = log_pi.iter().map(|l| l.v).fold(f64::NEG_INFINITY, f64::max);
        let unnormalized: Vec<Dual> = log_pi.iter().map(|&l| (l - c(max)).exp()).collect();
        let total = unnormalized.iter().fold(c(0.0), |acc, &u| acc + u);
        let sqrt_pi: Vec<Dual> = unnormalized.iter().map(|&u| (u / total).sqrt()).collect();
        let fitness: Vec<Dual> = (0..20).map(|a| log_pi[a] - c(model.log_mutation_equilibrium[a])).collect();
        // g(d) = d / (1 - exp(-d)) = 1 + d/2 + (d/2) coth(d/2) - 1; the Taylor series is used for small |d|,
        // since the derivative of the closed form suffers from cancellation (relative error ~ eps / |d|)
        let g = |d: Dual| {
            if d.v.abs() < 1e-2 {
                let d2 = d * d;
                c(1.0) + d * c(0.5)
                    + d2 * (c(1.0 / 12.0)
                        - d2 * (c(1.0 / 720.0) - d2 * (c(1.0 / 30240.0) - d2 * c(1.0 / 1209600.0))))
            } else {
                d / -(-d).exp_m1()
            }
        };

        let mut value = c(0.0);
        for i in 0..20 {
            for j in (i + 1)..20 {
                let S_ij = sqrt_pi[i] / sqrt_pi[j] * c(model.scaled_Mu[(i, j)]) * g(fitness[j] - fitness[i]);
                value = value + c(G[(i, j)]) * S_ij;
            }
        }
        for a in 0..20 {
            value = value + c(b[a]) * sqrt_pi[a];
        }
        // pi_reg * ln sum_k exp(logc_k - sum_i (x_i - mu_ki)^2 / (2 var_ki)), shifted by a constant
        let exponents: Vec<Dual> = (0..GMM_COMPONENTS)
            .map(|k| {
                (0..NUM_COMPONENTS).fold(c(model.gmm_logc[k]), |acc, i| {
                    let diff = x[i] - c(model.gmm_mu[(i, k)]);
                    acc - diff * diff * c(0.5 * model.gmm_inv_var[(i, k)])
                })
            })
            .collect();
        let shift = exponents.iter().map(|e| e.v).fold(f64::NEG_INFINITY, f64::max);
        let total = exponents.iter().fold(c(0.0), |acc, &e| acc + (e - c(shift)).exp());
        value + c(model.prior_strength) * (total.ln() + c(shift))
    }

    #[test]
    fn backpropagation_matches_forward_mode_derivatives() {
        let model = SharedModel::new(&test_Mu(), RATE_SCALING, &PCA::new(), PI_REG);

        // Arbitrary upstream gradients
        let G = SMatrix::<f64, 20, 20>::from_fn(|i, j| ((i * 20 + j) as f64 * 0.37).sin());
        let b = SVector::<f64, 20>::from_fn(|a, _| ((a as f64) * 1.3).cos());

        // x with fitness differences exactly zero: log_pi equal to the mutation equilibrium
        let neutral = model.components * model.log_mutation_equilibrium;
        let direction: Vec<f64> = test_point();

        let mut points: Vec<Vec<f64>> = vec![];
        for scale in [0.0, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0] {
            points.push((0..NUM_COMPONENTS).map(|k| neutral[k] + scale * direction[k]).collect());
        }
        for scale in [1e-6, 1e-2, 1.0, 3.0] {
            points.push(direction.iter().map(|d| d * scale / 0.3).collect());
        }

        for x in points {
            let x_vec = SVector::<f64, NUM_COMPONENTS>::from_column_slice(&x);
            let eval = model.evaluate(&x_vec);
            let grad = model.backpropagate(&eval, &G, &b);

            let mut max_rel_error = 0.0f64;
            let max_grad = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
            for k in 0..NUM_COMPONENTS {
                let x_dual: Vec<Dual> = (0..NUM_COMPONENTS)
                    .map(|m| Dual { v: x[m], d: if m == k { 1.0 } else { 0.0 } })
                    .collect();
                let reference = reference_surrogate(&model, &x_dual, &G, &b).d;
                max_rel_error = max_rel_error.max((grad[k] - reference).abs() / max_grad);
            }
            let min_abs_fitness_diff = (0..20)
                .flat_map(|i| ((i + 1)..20).map(move |j| (i, j)))
                .map(|(i, j)| (eval.fitness[j] - eval.fitness[i]).abs())
                .fold(f64::INFINITY, f64::min);
            println!(
                "min |f_j - f_i| {:.1e}: max error relative to max |grad| {:.2e}",
                min_abs_fitness_diff, max_rel_error
            );
            assert!(max_rel_error < 1e-13, "relative error {:e}", max_rel_error);
        }
    }

    /// Full gradient including phylo_grad against a fourth order central difference
    #[test]
    fn full_gradient_matches_fourth_order_finite_differences() {
        let mut logp = test_logp();
        let mut dummy = vec![0.0; NUM_COMPONENTS];
        for x in [test_point(), test_point().iter().map(|v| v * 5.0).collect::<Vec<f64>>()] {
            let mut grad = vec![0.0; NUM_COMPONENTS];
            logp.logp(&x, &mut grad).unwrap();
            let max_grad = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
            let h = 1e-3;
            let mut max_rel_error = 0.0f64;
            for k in 0..NUM_COMPONENTS {
                let mut f = |offset: f64| {
                    let mut xs = x.clone();
                    xs[k] += offset;
                    logp.logp(&xs, &mut dummy).unwrap()
                };
                let fd = (-f(2.0 * h) + 8.0 * f(h) - 8.0 * f(-h) + f(-2.0 * h)) / (12.0 * h);
                max_rel_error = max_rel_error.max((fd - grad[k]).abs() / max_grad);
            }
            println!("full gradient vs 4th order FD: max error relative to max |grad| {:.2e}", max_rel_error);
            assert!(max_rel_error < 1e-9, "relative error {:e}", max_rel_error);
        }
    }

    #[test]
    fn logp_timing() {
        let mut logp = test_logp();
        let x = vec![0.1; NUM_COMPONENTS];
        let mut grad = vec![0.0; NUM_COMPONENTS];
        let n = 2000;
        let start = std::time::Instant::now();
        for _ in 0..n {
            logp.logp(&x, &mut grad).unwrap();
        }
        println!("logp+grad: {:.1} us", start.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
}


