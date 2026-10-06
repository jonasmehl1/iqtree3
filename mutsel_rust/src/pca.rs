use candle_core::Tensor;

pub struct PCA {
    /// The components are per row
    pub components: Tensor,
    pub mean: Tensor,
    /// The diagonal Gaussian mixture prior, marginalised to the first `num_components`
    /// coordinates: [K, num_components] means and inverse variances, and per component
    /// `ln w_k - sum_i ln(2 pi var_ki) / 2` as [1, K].
    gmm_mu: Tensor,
    gmm_inv_var: Tensor,
    gmm_logc: Tensor,
    pub num_components: usize,
}

impl PCA {
    pub fn new(num_components: usize) -> PCA {
        let components: Vec<f64> = super::data::PCA_COMPONENTS
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        let mean_values = super::data::PCA_MEAN
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        let components =
            Tensor::from_vec(components, &[20, 20], &candle_core::Device::Cpu).unwrap();
        let mean = Tensor::from_vec(mean_values, &[19], &candle_core::Device::Cpu).unwrap();

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

        // A diagonal mixture marginalises by dropping the coordinates that are not optimised.
        let gmm_mu = gmm_mu.narrow(1, 0, num_components).unwrap();
        let gmm_var = gmm_var.narrow(1, 0, num_components).unwrap();
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
            mean,
            gmm_mu,
            gmm_inv_var: gmm_var.recip().unwrap(),
            gmm_logc,
            num_components,
        }
    }

    pub fn log_freq_to_pca_coordinates(&self, data: &Tensor) -> Tensor {
        let pca_coordinates = data
            .matmul(&self.components.transpose(0, 1).unwrap())
            .unwrap();
        pca_coordinates.narrow(1, 0, self.num_components).unwrap()
    }

    pub fn pca_coordinates_to_log_freq(&self, pca_coordinates: &Tensor) -> Tensor {
        let pad_means = self
            .mean
            .narrow(0, self.num_components, 19 - self.num_components)
            .unwrap();

        let full_pca_coordinates = if self.num_components == 19 {
            pca_coordinates.clone()
        } else {
            Tensor::cat(&[pca_coordinates, &pad_means.unsqueeze(0).unwrap()], 1).unwrap()
        };
        let log_freq = full_pca_coordinates.matmul(&self.components).unwrap();
        log_freq
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

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
    }

    #[test]
    fn projection_returns_requested_number_of_components() {
        let pca = PCA::new(4);
        let data = Tensor::from_vec(
            (0..40).map(|x| x as f64 * 0.01).collect(),
            &[2, 20],
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let coords = pca.log_freq_to_pca_coordinates(&data);

        assert_eq!(coords.dims(), &[2, 4]);
    }

    #[test]
    fn projection_returns_requested_number_of_components_19() {
        let pca = PCA::new(19);
        let data = Tensor::from_vec(
            (0..40).map(|x| x as f64 * 0.01).collect(),
            &[2, 20],
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let coords = pca.log_freq_to_pca_coordinates(&data);

        assert_eq!(coords.dims(), &[2, 19]);
    }
    #[test]
    fn inverse_fills_missing_components_with_means() {
        let pca = PCA::new(3);
        let coords =
            Tensor::from_vec(vec![0.2, -0.3, 0.5], &[1, 3], &candle_core::Device::Cpu).unwrap();

        let reconstructed = pca.pca_coordinates_to_log_freq(&coords);

        let expected_tail = pca.mean.narrow(0, 3, 16).unwrap().unsqueeze(0).unwrap();
        let full_coords = Tensor::cat(&[&coords, &expected_tail], 1).unwrap();
        let expected = full_coords.matmul(&pca.components).unwrap();

        assert_eq!(reconstructed.dims(), &[1, 20]);
        assert!(max_abs_diff(&reconstructed, &expected) < 1e-12);
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
        for num_components in [3, 10, 19] {
            let pca = PCA::new(num_components);
            let rows: Vec<Vec<f64>> = (0..3)
                .map(|site| {
                    (0..num_components)
                        .map(|i| -1.5 + 0.2 * i as f64 + 0.7 * site as f64)
                        .collect()
                })
                .collect();
            let coords = Tensor::from_vec(
                rows.concat(),
                &[3, num_components],
                &candle_core::Device::Cpu,
            )
            .unwrap();

            let penalty = pca
                .penalty_on_pca_coordinates(&coords, 2.0)
                .to_scalar::<f64>()
                .unwrap();
            let expected: f64 = 2.0 * rows.iter().map(|row| reference_penalty(row)).sum::<f64>();

            assert!(
                (penalty - expected).abs() < 1e-9 * expected.abs().max(1.0),
                "{} components: {} against {}",
                num_components,
                penalty,
                expected
            );
        }
    }

    #[test]
    fn penalty_stays_finite_far_from_every_component() {
        let pca = PCA::new(19);
        let far = (pca.mean.unsqueeze(0).unwrap() + 25.0).unwrap();
        let coords = candle_core::Var::from_tensor(&far).unwrap();

        let penalty = pca.penalty_on_pca_coordinates(&coords, 1.0);
        let value = penalty.to_scalar::<f64>().unwrap();
        let grad = penalty.backward().unwrap().get(&coords).unwrap().clone();
        let grad = grad.flatten_all().unwrap().to_vec1::<f64>().unwrap();

        assert!(value.is_finite() && value > 1e3, "penalty far out was {}", value);
        assert!(grad.iter().all(|g| g.is_finite() && *g > 0.0), "gradient {:?}", grad);
    }
}
