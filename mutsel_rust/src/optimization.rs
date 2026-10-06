use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use candle_core::{Tensor, Var};
use candle_nn::{Optimizer, ops::softmax};
use phylo_grad::FelsensteinTree;

use crate::{
    MutselParams, Verbosity, felsenstein::{self, FelsensteinOp, FelsensteinWithEdgeOp}, model::{self, calc_rate_matrix}, pca::PCA, utils::tensor_full,
};

pub trait Optimizable {
    fn variables(&self) -> Vec<Var>;
    fn variables_names(&self) -> Vec<String>;
    fn model_name(&self) -> String;
    fn likelihood(&self) -> Tensor;
    fn penalty(&self) -> Tensor;
    fn print_state(&self) {}
}

fn calc_likelihood(
    mu: &Tensor,
    log_pi: &Tensor,
    log_branch_lengths: &Tensor,
    felsenstein_op: FelsensteinWithEdgeOp,
) -> Tensor {
    let (S, sqrt_pi) = calc_rate_matrix(mu, log_pi, &tensor_full(1.0, &[]));

    let branch_lengths = log_branch_lengths.exp().unwrap();

    let average_rate = model::substitution_rates_tensor(&S, &sqrt_pi)
        .mean_all()
        .unwrap();
    let S = S.broadcast_div(&average_rate).unwrap();

    S.apply_op3(&sqrt_pi, &branch_lengths, felsenstein_op)
        .unwrap()
        .sum_all()
        .unwrap()
}

pub struct BranchParameters {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    pub log_branch_lengths: Var,
    pub Mu: Tensor,
    pub log_pi: Tensor,
}

impl Optimizable for BranchParameters {
    fn variables(&self) -> Vec<Var> {
        vec![self.log_branch_lengths.clone()]
    }
    fn variables_names(&self) -> Vec<String> {
        vec!["log_branch_lengths".to_string()]
    }
    fn model_name(&self) -> String {
        "BranchParameters".to_string()
    }

    fn likelihood(&self) -> Tensor {
        calc_likelihood(
            &self.Mu,
            &self.log_pi,
            &self.log_branch_lengths,
            self.felsenstein_op.clone(),
        )
    }

    fn penalty(&self) -> Tensor {
        tensor_full(0.0, &[])
    }

    fn print_state(&self) {}
}

/// Smallest site frequency in the output model. read_site_model_file (mutsel_wrapper.cpp)
/// rejects frequencies <= 1e-10, and weakly penalised fits push residues absent from a
/// column far below that.
const MIN_SITE_FREQ: f64 = 1e-9;

/// Normalised log frequencies with every frequency raised to at least `min_freq`.
fn floor_log_pi(log_pi: &Tensor, min_freq: f64) -> Tensor {
    let pi = softmax(log_pi, 1).unwrap().maximum(min_freq).unwrap();
    let pi = pi.broadcast_div(&pi.sum_keepdim(1).unwrap()).unwrap();
    pi.log().unwrap()
}

pub struct ModelParameters {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    pub log_R: Var,
    /// Defines the log pi per site
    pub pca_coordinates: Var,
    pub log_branch_lengths: Var,
    pub init_log_branch_lengths: Tensor,
    pub reg_para: MutselParams,
    pub init_log_R: Tensor,
    pub pca_data: PCA,
}

impl ModelParameters {
    pub fn log_pi(&self) -> Tensor {
        self.pca_data
            .pca_coordinates_to_log_freq(&self.pca_coordinates)
    }

    pub fn calc_rate_matrix(&self) -> (Tensor, Tensor) {
        self.rate_matrix_from_log_pi(&self.log_pi())
    }

    /// The model handed back to IQ-TREE and written to the .sitemodel: frequencies are
    /// floored at MIN_SITE_FREQ before S is built, so R and pi stay consistent.
    pub fn calc_output_rate_matrix(&self) -> (Tensor, Tensor) {
        self.rate_matrix_from_log_pi(&floor_log_pi(&self.log_pi(), MIN_SITE_FREQ))
    }

    fn rate_matrix_from_log_pi(&self, log_pi: &Tensor) -> (Tensor, Tensor) {
        let (S, sqrt_pi) =
            calc_rate_matrix(&Mu(&self.log_R), log_pi, &tensor_full(1.0, &[]));
        let average_rate = model::substitution_rates_tensor(&S, &sqrt_pi)
            .mean_all()
            .unwrap();
        let S = S.broadcast_div(&average_rate).unwrap();
        (S, sqrt_pi)
    }

    pub fn save_npz(&self, path: &Path) {
        let Mu = Mu(&self.log_R.as_detached_tensor());
        let pi = softmax(&self.log_pi().detach(), 1).unwrap();

        Tensor::write_npz(
            &[
                ("Mu", &Mu),
                ("pi", &pi),
                ("init_log_R", &self.init_log_R),
                ("branch_lengths", &self.log_branch_lengths.exp().unwrap()),
            ],
            path,
        )
        .unwrap();
    }
}

impl Optimizable for ModelParameters {
    fn variables(&self) -> Vec<Var> {
        vec![
            self.log_R.clone(),
            self.pca_coordinates.clone(),
            self.log_branch_lengths.clone(),
        ]
    }

    fn variables_names(&self) -> Vec<String> {
        vec![
            "log_R".to_string(),
            "pca_coordinates".to_string(),
            "log_branch_lengths".to_string(),
        ]
    }

    fn model_name(&self) -> String {
        "ModelParameters".to_string()
    }

    fn likelihood(&self) -> Tensor {
        let (S, sqrt_pi) = self.calc_rate_matrix();
        let branch_lengths = self.log_branch_lengths.exp().unwrap();
        S.apply_op3(&sqrt_pi, &branch_lengths, self.felsenstein_op.clone())
            .unwrap()
            .sum_all()
            .unwrap()
    }

    fn penalty(&self) -> Tensor {
        let pi_penalty = self
            .pca_data
            .penalty_on_pca_coordinates(&self.pca_coordinates, self.reg_para.pi_reg);

        fn log_Mu(log_R: &Tensor) -> Tensor {
            let Mu = Mu(log_R);

            // One out the diagonal, since we only want to penalize the off-diagonal elements
            let Mu = (&Mu
                - (&Mu - 1.0).unwrap()
                    * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
            .unwrap();

            return Mu.log().unwrap();
        }

        let Mu = log_Mu(&self.log_R)
            .sub(&log_Mu(&self.init_log_R))
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap();
        let Mu_penalty = (Mu * self.reg_para.Mu_reg).unwrap();

        (pi_penalty + Mu_penalty).unwrap()
    }

    fn print_state(&self) {
    }
}

pub fn optimize(
    model: &impl Optimizable,
    min_iterations: usize,
    max_iterations: usize,
    min_rel_improvement: f64,
    no_improve_patience: usize,
    verbosity: Verbosity,
    _prefix: &str,
) {
    let variables = model.variables();
    // let variable_names = model.variables_names();
    // let mut trajectory_tensors: Vec<Vec<Tensor>> = variables
    //     .iter()
    //     .map(|variable| vec![variable.as_tensor().copy().unwrap()])
    //     .collect();
    let mut opt = candle_nn::optim::AdamW::new_lr(variables, 0.05).unwrap();
    let parameter = candle_nn::optim::ParamsAdamW {
        lr: 0.03,
        weight_decay: 0.0,
        ..Default::default()
    };
    opt.set_params(parameter);

    let variables = model.variables();

    let mut best_opt = f64::INFINITY;
    let mut best_params: Vec<Tensor> = variables
        .iter()
        .map(|variable| variable.as_tensor().copy().unwrap())
        .collect();
    let mut no_improve_count = 0;

    for iteration in 0.. {
        let neg_likelihood = model.likelihood().neg().unwrap();
        let penalty = model.penalty();
        let opt_fn = (&neg_likelihood + &penalty).unwrap();

        let current_opt = opt_fn.to_scalar::<f64>().unwrap();

        if current_opt < best_opt {
            best_params = variables
                .iter()
                .map(|variable| variable.as_tensor().copy().unwrap())
                .collect();
        }

        if verbosity.should_print(Verbosity::Med) {
            println!(
                "Iteration {}: neg Loglikelihood {:.3}, Optfn {:.3}",
                iteration,
                neg_likelihood.to_scalar::<f64>().unwrap(),
                current_opt
            );
            model.print_state();
        }

        let grads = opt_fn.backward().unwrap();
        opt.step(&grads).unwrap();

        // for (traj, new) in &mut trajectory_tensors.iter_mut().zip(variables.iter()) {
        //     traj.push(new.as_tensor().copy().unwrap());
        // }
        if iteration > min_iterations.saturating_sub(no_improve_patience) {
            let rel_improvement = (best_opt - current_opt) / best_opt.abs().max(1e-12);
            if rel_improvement > min_rel_improvement {
                no_improve_count = 0;
            } else {
                no_improve_count += 1;
            }
            if no_improve_count >= no_improve_patience {
                println!(
                    "Stopping at iteration {} (relative loss improvement {:.2e} <= {:.2e} for {} consecutive iterations)",
                    iteration, rel_improvement, min_rel_improvement, no_improve_patience
                );
                break;
            }
        }

        if iteration > max_iterations {
            println!("Reached maximum iterations ({})", max_iterations);
            break;
        }

        best_opt = current_opt.min(best_opt);
    }

    // Obtain parameters from best iteration
    for (variable, value) in variables.iter().zip(best_params.iter()) {
        variable.set(value).unwrap();
    }

    // Combine trajectory tensors into a single tensor for each variable
    // let trajectory_tensors: Vec<Tensor> = trajectory_tensors
    //     .into_iter()
    //     .map(|tensors| Tensor::stack(&tensors, 0).unwrap())
    //     .collect();
    // let filename = format!("{}.traj_{}.npz", _prefix, model.model_name());
    // Tensor::write_npz(
    //     &variable_names
    //         .iter()
    //         .zip(trajectory_tensors.iter())
    //         .collect::<Vec<_>>(),
    //     Path::new(&filename),
    // )
    // .unwrap();
}

pub fn optimize_branch_lengths(
    felsenstein_op: FelsensteinWithEdgeOp,
    log_pi: &Tensor,
    Mu: &Tensor,
    log_branch_lengths: &Tensor,
    verbosity: Verbosity,
    prefix: &str,
) -> Tensor {
    let model = BranchParameters {
        felsenstein_op,
        log_branch_lengths: Var::from_tensor(log_branch_lengths).unwrap(),
        Mu: Mu.clone(),
        log_pi: log_pi.clone(),
    };

    optimize(&model, 10, 200, 1e-6, 5, verbosity, prefix);

    model.log_branch_lengths.as_tensor().copy().unwrap()
}

pub fn two_step_light_pmsf(
    felsenstein_op: FelsensteinOp,
    categories: &[[f64; 20]],
    weights: &[f64],
    log_branch_lengths: &Tensor,
    verbosity: Verbosity,
    prefix: &str,
) -> (Tensor, Tensor) {
    let step1_site_freq = light_pmsf(
        felsenstein_op.into_with_edge_op(),
        categories,
        weights,
        log_branch_lengths,
    );

    let Mu = loadMu();

    let log_pi = step1_site_freq.log().unwrap();

    let log_branch_lengths = optimize_branch_lengths(
        felsenstein_op.into_with_edge_op(),
        &log_pi,
        &Mu,
        log_branch_lengths,
        verbosity,
        prefix,
    );

    let final_site_freq = light_pmsf(
        felsenstein_op.into_with_edge_op(),
        categories,
        weights,
        &log_branch_lengths,
    );

    (final_site_freq, log_branch_lengths)
}

pub fn light_pmsf(
    felsenstein_op: FelsensteinWithEdgeOp,
    categories: &[[f64; 20]],
    weights: &[f64],
    log_branch_lengths: &Tensor,
) -> Tensor {
    let mut likelihoods = vec![];

    let Mu = loadMu();

    for category in categories.iter() {
        let category_tensor =
            Tensor::from_vec(category.to_vec(), &[20], &candle_core::Device::Cpu).unwrap();
        let log_pi = category_tensor.log().unwrap().unsqueeze(0).unwrap();

        let (S, sqrt_pi) = model::calc_rate_matrix(&Mu, &log_pi, &tensor_full(1.0, &[]));

        let likelihood = S
            .apply_op3(
                &sqrt_pi,
                &log_branch_lengths.exp().unwrap(),
                felsenstein_op.into_fwd_op(),
            )
            .unwrap();
        likelihoods.push(likelihood);
    }

    let likelihoods = Tensor::stack(&likelihoods, 0).unwrap();
    let weights_tensor =
        Tensor::from_slice(&weights, &[weights.len()], &candle_core::Device::Cpu).unwrap();
    let log_weights_tensor = weights_tensor.log().unwrap().unsqueeze(1).unwrap();

    let weighted_likelihoods = (likelihoods.broadcast_add(&log_weights_tensor)).unwrap();

    let posteriors = candle_nn::ops::softmax(&weighted_likelihoods, 0).unwrap();

    let category_tensor = Tensor::from_vec(
        categories.iter().flatten().copied().collect(),
        &[categories.len(), 20],
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let site_freq = posteriors.t().unwrap().matmul(&category_tensor).unwrap();

    site_freq
}

/// Calculates the mutation rate matrix from the log_R variable. This is a lower triangular matrix with the exchanabilities and the equilibirum in the diagonal. Both as in log space.
/// Returns Mut and the equilibrium frequencies in the diagonal. (not log space)
pub fn Mu(log_parameter: &Tensor) -> Tensor {
    let parameter = log_parameter.exp().unwrap();
    let diagonal = (&parameter
        * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
    .unwrap();
    let off_diagonal = (parameter - &diagonal).unwrap();

    // Mutation equilibrium should stay fixed to neutral
    let diagonal = diagonal.detach();

    let pi = diagonal
        .broadcast_div(&diagonal.sum_all().unwrap())
        .unwrap();
    let pi_vec = pi.sum(1).unwrap();
    let sqrt_pi = pi_vec.sqrt().unwrap();
    let sqrt_pi_inv = sqrt_pi.recip().unwrap();

    let S = (&off_diagonal + off_diagonal.t().unwrap()).unwrap();

    // Q = diag(sqrt_pi_inv) * S * diag(sqrt_pi)
    let Q = sqrt_pi_inv
        .unsqueeze(1)
        .unwrap()
        .broadcast_mul(&S)
        .unwrap()
        .broadcast_mul(&sqrt_pi.unsqueeze(0).unwrap())
        .unwrap();
    let Q = (&Q
        - &Q * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
    .unwrap();

    let row_sum = Q.sum(1).unwrap();
    let sum = row_sum.dot(&pi_vec).unwrap();
    let Q = Q.broadcast_div(&sum).unwrap();

    // The scaling of Q does not matter, since we always scale it so the subsitution rate over all rate averages to 1.0.

    // diagonal is zero here, but they are not used anyway
    (Q + pi).unwrap()
}

// Used with the MutSel model.
fn loadMu() -> Tensor {
    let R_lower = crate::data::load_lower_R_with_equi(crate::data::M_TXT);
    let log_R = R_lower.log().unwrap();
    Mu(&log_R)
}

/// Returns the optimal S, sqrt_pi and rate parameters.
/// S: [L, 20, 20], sqrt_pi: [L, 20]
/// rate parameters is either alpha shape: [1] or [2 * num_cat] for free rate model (first num_cat weights, second num_cat are rates)
pub fn optimize_internal(
    felsenstein: FelsensteinTree<20>,
    distances: &[f64],
    mutsel_params: super::MutselParams,
    prior_R_file: Option<&Path>,
    verbosity: Verbosity,
    out_prefix: &str,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let op = felsenstein::FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));

    // Lower triangular matrix with zeros everywhere else
    let init_R = if let Some(prior_R_file) = prior_R_file {
        let file_content = std::fs::read_to_string(prior_R_file)?;
        crate::data::load_lower_R_with_equi(&file_content)
    } else {
        crate::data::load_lower_R_with_equi(crate::data::M_TXT)
    };

    let log_branch_lengths =
        Tensor::from_slice(distances, &[distances.len()], &candle_core::Device::Cpu)?.log()?;

    // Do our lightweight PMSF procedure for initialization:
    let (site_freq, log_branch_lengths) = two_step_light_pmsf(
        op.clone(),
        crate::data::UDM256,
        crate::data::UDM256_WEIGHTS,
        &log_branch_lengths,
        verbosity,
        out_prefix,
    );

    // Variable which gets optimized
    let log_R = Var::from_tensor(&init_R.log()?)?;
    let init_log_pi = site_freq.log()?;

    let init_log_R = log_R.detach().copy().unwrap();

    let pca = PCA::new(19);

    let pca_coordinates = pca.log_freq_to_pca_coordinates(&init_log_pi);

    let model = ModelParameters {
        felsenstein_op: op.into_with_edge_op(),
        log_R,
        pca_coordinates: Var::from_tensor(&pca_coordinates).unwrap(),
        log_branch_lengths: Var::from_tensor(&log_branch_lengths).unwrap(),
        init_log_branch_lengths: log_branch_lengths.detach().copy().unwrap(),
        reg_para: mutsel_params,
        init_log_R,
        pca_data: pca,
    };

    optimize(&model, 100, 500, 1e-6, 5, verbosity, out_prefix);

    let (S, sqrt_pi) = model.calc_output_rate_matrix();

    if verbosity.should_print(Verbosity::Med) {
        model.save_npz(Path::new(&format!("{}.mutsel.npz", out_prefix)));
    }

    Ok((S, sqrt_pi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_log_pi_lifts_tiny_frequencies_only() {
        // Site 0 has one residue at exp(-40) ~ 4e-18, site 1 is uniform.
        let mut values = vec![0.0; 40];
        values[3] = -40.0;
        let log_pi = Tensor::from_vec(values, &[2, 20], &candle_core::Device::Cpu).unwrap();

        let pi = floor_log_pi(&log_pi, MIN_SITE_FREQ).exp().unwrap();
        let pi = pi.to_vec2::<f64>().unwrap();

        for row in &pi {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(row.iter().all(|&p| p > 1e-10));
        }
        assert!((pi[0][3] - MIN_SITE_FREQ).abs() < 1e-15);
        assert!((pi[0][0] - 1.0 / 19.0).abs() < 1e-9);
        assert!(pi[1].iter().all(|&p| (p - 0.05).abs() < 1e-15));
    }
}
