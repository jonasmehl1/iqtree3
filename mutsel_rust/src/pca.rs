use candle_core::Tensor;

pub struct PCA {
    /// The components are per row
    pub components: Tensor,
    /// The diagonal Gaussian mixture prior: [K, 19] means and inverse variances, and per
    /// component `ln w_k - sum_i ln(2 pi var_ki) / 2` as [1, K].
    pub gmm_mu: Tensor,
    pub gmm_inv_var: Tensor,
    pub gmm_logc: Tensor,
}

impl PCA {
    pub fn new() -> PCA {
        let components: Vec<f64> = super::data::PCA_COMPONENTS
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        let components =
            Tensor::from_vec(components, &[20, 20], &candle_core::Device::Cpu).unwrap();

        let weight: Vec<f64> = super::data::PCA_GMM_WEIGHT
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();
        let gmm_mu: Vec<f64> = super::data::PCA_GMM_MU
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();
        let gmm_var: Vec<f64> = super::data::PCA_GMM_VAR
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();
        let k = weight.len();
        let gmm_mu = Tensor::from_vec(gmm_mu, &[k, 19], &candle_core::Device::Cpu).unwrap();
        let gmm_var = Tensor::from_vec(gmm_var, &[k, 19], &candle_core::Device::Cpu).unwrap();
        let log_weight = Tensor::from_vec(weight, &[k], &candle_core::Device::Cpu)
            .unwrap()
            .log()
            .unwrap();
        let gmm_logc = (log_weight
            - gmm_var
                .affine(2.0 * std::f64::consts::PI, 0.0)
                .unwrap()
                .log()
                .unwrap()
                .sum(1)
                .unwrap()
                .affine(0.5, 0.0)
                .unwrap())
        .unwrap()
        .unsqueeze(0)
        .unwrap();

        PCA {
            components: components.narrow(0, 0, 19).unwrap(),
            gmm_mu,
            gmm_inv_var: gmm_var.recip().unwrap(),
            gmm_logc,
        }
    }

    pub fn log_freq_to_pca_coordinates(&self, data: &Tensor) -> Tensor {
        data.matmul(&self.components.transpose(0, 1).unwrap())
            .unwrap()
    }

    pub fn pca_coordinates_to_log_freq(&self, pca_coordinates: &Tensor) -> Tensor {
        pca_coordinates.matmul(&self.components).unwrap()
    }

    /// `-strength * sum_sites ln sum_k w_k N(x; mu_k, diag(var_k))`.
    pub fn penalty_on_pca_coordinates(&self, pca_coordinates: &Tensor, strength: f64) -> Tensor {
        // [num_sites, K]: ln w_k + ln N(x; mu_k, diag(var_k)).
        let exponent = pca_coordinates
            .unsqueeze(1)
            .unwrap()
            .broadcast_sub(&self.gmm_mu.unsqueeze(0).unwrap())
            .unwrap()
            .sqr()
            .unwrap()
            .broadcast_mul(&self.gmm_inv_var.unsqueeze(0).unwrap())
            .unwrap()
            .sum(2)
            .unwrap()
            .affine(-0.5, 0.0)
            .unwrap()
            .broadcast_add(&self.gmm_logc)
            .unwrap();
        // Far from every component all exponentials underflow, so the largest is pulled out.
        // Detached, since the identity holds for any constant.
        let top = exponent.max_keepdim(1).unwrap().detach();
        let log_density = exponent
            .broadcast_sub(&top)
            .unwrap()
            .exp()
            .unwrap()
            .sum_keepdim(1)
            .unwrap()
            .log()
            .unwrap()
            .add(&top)
            .unwrap();
        log_density.sum_all().unwrap().affine(-strength, 0.0).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_returns_19_components() {
        let pca = PCA::new();
        let data = Tensor::from_vec(
            (0..40).map(|x| x as f64 * 0.01).collect(),
            &[2, 20],
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let coords = pca.log_freq_to_pca_coordinates(&data);

        assert_eq!(coords.dims(), &[2, 19]);
    }

    /// `-ln sum_k w_k prod_i N(x_i; mu_ki, var_ki)` for one site, straight from data.rs.
    fn reference_penalty(x: &[f64]) -> f64 {
        let parse = |text: &str| -> Vec<f64> {
            text.split_whitespace().map(|s| s.parse::<f64>().unwrap()).collect()
        };
        let weight = parse(super::super::data::PCA_GMM_WEIGHT);
        let mu = parse(super::super::data::PCA_GMM_MU);
        let var = parse(super::super::data::PCA_GMM_VAR);
        let density: f64 = (0..weight.len())
            .map(|k| {
                weight[k]
                    * x.iter()
                        .enumerate()
                        .map(|(i, x)| {
                            let (m, v) = (mu[k * 19 + i], var[k * 19 + i]);
                            (-(x - m).powi(2) / (2.0 * v)).exp()
                                / (2.0 * std::f64::consts::PI * v).sqrt()
                        })
                        .product::<f64>()
            })
            .sum();
        -density.ln()
    }

    #[test]
    fn gmm_weights_sum_to_one() {
        let total: f64 = super::super::data::PCA_GMM_WEIGHT
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .sum();
        assert!((total - 1.0).abs() < 1e-8, "weights sum to {}", total);
    }

    #[test]
    fn penalty_matches_direct_mixture_density() {
        let pca = PCA::new();
        let rows: Vec<Vec<f64>> = (0..3)
            .map(|site| {
                (0..19)
                    .map(|i| -1.5 + 0.2 * i as f64 + 0.7 * site as f64)
                    .collect()
            })
            .collect();
        let coords =
            Tensor::from_vec(rows.concat(), &[3, 19], &candle_core::Device::Cpu).unwrap();

        let penalty = pca
            .penalty_on_pca_coordinates(&coords, 2.0)
            .to_scalar::<f64>()
            .unwrap();
        let expected: f64 = 2.0 * rows.iter().map(|row| reference_penalty(row)).sum::<f64>();

        assert!(
            (penalty - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "{} against {}",
            penalty,
            expected
        );
    }

    #[test]
    fn penalty_stays_finite_far_from_every_component() {
        let pca = PCA::new();
        let far = Tensor::full(25.0, (1, 19), &candle_core::Device::Cpu).unwrap();
        let coords = candle_core::Var::from_tensor(&far).unwrap();

        let penalty = pca.penalty_on_pca_coordinates(&coords, 1.0);
        let value = penalty.to_scalar::<f64>().unwrap();
        let grad = penalty.backward().unwrap().get(&coords).unwrap().clone();
        let grad = grad.flatten_all().unwrap().to_vec1::<f64>().unwrap();

        assert!(value.is_finite() && value > 1e3, "penalty far out was {}", value);
        assert!(grad.iter().all(|g| g.is_finite() && *g > 0.0), "gradient {:?}", grad);
    }
}
