import os
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.optimize import linear_sum_assignment
from diff_models import diff_CSDI

class CSDI_base(nn.Module):
    """
    Single-level tile-based correlation (128x128) via Cholesky.
    Time-dependent blending of Gaussian vs. Blue noise.
    "Standard rectification" using Hungarian assignment in training.
    """

    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        # 1) Basic model config
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        # Side info dimension
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim,
            embedding_dim=self.emb_feature_dim
        )

        # 2) Diffusion Model
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"]**0.5,
                config_diff["beta_end"]**0.5,
                self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device)
            .unsqueeze(1).unsqueeze(1)
        )

        # 3) Load Precomputed Blue Noise Covariance Matrix (Σ)
        self.blue_noise_chol_path = config["model"].get("blue_noise_chol_path", "blue_noise_chol_matrix.pt")
        self.L_chol_small = self.load_blue_noise_cholesky(self.blue_noise_chol_path).to(self.device)

        # 4) Gamma(t) schedule
        self.gamma_start = config["model"].get("gamma_start", 0.0)
        self.gamma_end   = config["model"].get("gamma_end",   3.0)
        self.gamma_tau   = config["model"].get("gamma_tau",   0.2)

        # 5) Standard rectified mapping using Hungarian
        self.use_rectified_mapping = config["model"].get("use_rectified_mapping", True)
        self.rectify_lambda = config["model"].get("rectify_lambda", 1.0)  # Partial blend factor

    # ------------------------------------------------
    # Load Precomputed Blue Noise Cholesky Matrix
    # ------------------------------------------------
    def load_blue_noise_cholesky(self, cov_path):
        """
        Load or compute Cholesky decomposition of the empirical blue noise covariance matrix.
        """
        if os.path.exists(cov_path):
            print(f"[INFO] Loading precomputed blue noise Cholesky factor from '{cov_path}'")
            L_chol = torch.load(cov_path)
        else:
            raise FileNotFoundError(f"[ERROR] Blue noise covariance matrix '{cov_path}' not found!")

        return L_chol

    # ------------------------------------------------
    # Generate Blue Noise from Precomputed Cholesky Matrix
    # ------------------------------------------------
    def generate_blue_noise_tile(self, B):
        """
        Generate blue noise using the precomputed Cholesky factor.
        """
        D_tile = self.L_chol_small.shape[0]  # Ensure correct dimensionality
        z = torch.randn(B, D_tile, device=self.device)  # Standard Gaussian noise
        blue_noise = torch.matmul(self.L_chol_small, z.T).T  # Apply blue noise transformation
        return blue_noise.view(B, int(math.sqrt(D_tile)), int(math.sqrt(D_tile)))

    # Gamma(t) schedule for transitioning from Gaussian to Blue Noise
    def get_gamma(self, t_tensor):
        x = self.gamma_start + (self.gamma_end - self.gamma_start) * ((t_tensor / self.num_steps) ** self.gamma_tau)
        gamma = torch.sigmoid(x)
        return gamma
    
    def rectify_mapping(self, data_batch, noise_batch):
        B, K, L = data_batch.shape

        data_flat = data_batch.reshape(B, -1)
        noise_flat = noise_batch.reshape(B, -1)

        # cost => shape(B,B)
        cost = torch.cdist(data_flat, noise_flat, p=2.0)**2
        cost_np = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        # reorder noise 
        hungarian_noise = noise_batch[col_ind]  # shape(B, K, L)

        if self.rectify_lambda >= 1.0:
            # full rectification
            return hungarian_noise
        else:
            # partial blend
            return self.rectify_lambda * hungarian_noise + (1.0 - self.rectify_lambda)*noise_batch

    # time embedding, etc.
    def time_embedding(self, pos, d_model=128):
        B,L = pos.shape
        pe = torch.zeros(B,L,d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1.0/torch.pow(
            10000.0,
            torch.arange(0,d_model,2, device=self.device)/d_model
        )
        pe[:,:,0::2] = torch.sin(position*div_term)
        pe[:,:,1::2] = torch.cos(position*div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask)*observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed*sample_ratio)
            topk_indices = rand_for_mask[i].topk(num_masked).indices
            rand_for_mask[i][topk_indices] = -1

        cond_mask = (rand_for_mask>0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            if self.target_strategy=="mix" and np.random.rand()>0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i]*for_pattern_mask[i-1]
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask*test_pattern_mask

    def get_side_info(self, observed_tp, cond_mask):
        B,K,L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim, device=self.device)
        )
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0,3,2,1)
        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    # ------------------------------------------------
    # Forward Noising Process: Transition from Gaussian to Blue Noise
    # ------------------------------------------------
    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if is_train != 1:
            t = torch.full((B,), float(set_t), device=self.device)
        else:
            t = torch.randint(0, self.num_steps, (B,), device=self.device).float()

        current_alpha = self.alpha_torch[t.long()]

        # Generate noise
        noise_gauss = torch.randn(B, K, L, device=self.device)  # White noise
        noise_blue = self.generate_blue_noise_tile(B)  # Precomputed blue noise

        gamma_vals = self.get_gamma(t).view(B, 1, 1)

        # Transition from Gaussian to Blue Noise over time
        noise = gamma_vals * noise_gauss + (1.0 - gamma_vals) * noise_blue

        # Apply Hungarian correction if needed
        if self.use_rectified_mapping and is_train == 1:
            noise = self.rectify_mapping(observed_data, noise)

        # Forward noising process
        noisy_data = (current_alpha ** 0.5) * observed_data + ((1.0 - current_alpha) ** 0.5) * noise

        t_int = t.long()
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t_int)

        # Compute loss
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    # ------------------------------------------------
    # Reverse Process: Use Blue Noise to Restore Details
    # ------------------------------------------------
    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L, device=self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                t_val = torch.tensor([float(t)], device=self.device)
                gamma_val = self.get_gamma(t_val)
                gamma_t = gamma_val.view(1, 1, 1)

                noise_gauss = torch.randn(B, K, L, device=self.device)
                noise_blue = self.generate_blue_noise_tile(B)
                noise_blend = gamma_t * noise_gauss + (1.0 - gamma_t) * noise_blue

                if self.use_rectified_mapping:
                    noise_blend = self.rectify_mapping(observed_data, noise_blend)

                predicted = self.diffmodel(self.set_input_to_diffmodel(current_sample, observed_data, cond_mask), side_info, t_val)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise_blend

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

class CSDI_RNA(CSDI_base):
    def __init__(self, config, device, target_dim=100):
        super(CSDI_RNA, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        # Permute for (Batch, Genes, Time) format
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)

        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)



    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
