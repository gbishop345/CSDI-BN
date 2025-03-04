import os
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.optimize import linear_sum_assignment
from diff_models import diff_CSDI

class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        # 1) Basic model config
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        # side info dimension
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

        # 3) correlation => Cholesky
        self.tile_k = 35
        self.tile_l = 48
        self.rho_feat = config["model"].get("rho_feat", 0.5)
        self.rho_time = config["model"].get("rho_time", 0.9)
        self.cov_save_path = config["model"].get("cov_save_path", "cov_matrix.pt")

        # Build or load single tile factor
        self.L_chol_small = self.create_covariance_matrix_2d(
            K=self.tile_k,
            L_max=self.tile_l,
            rho_feat=self.rho_feat,
            rho_time=self.rho_time,
            save_path=self.cov_save_path
        ).to(self.device)

        # 4) gamma(t)
        self.gamma_start = config["model"].get("gamma_start", 0.0)
        self.gamma_end   = config["model"].get("gamma_end",   3.0)
        self.gamma_tau   = config["model"].get("gamma_tau",   0.2)

        # 5) Standard rectified mapping using Hungarian
        self.use_rectified_mapping = config["model"].get("use_rectified_mapping", True)
        self.rectify_lambda = config["model"].get("rectify_lambda", 1.0)  # partial blend factor


    # ------------------------------------------------
    # Build or load covariance => store Cholesky
    # ------------------------------------------------
    def create_covariance_matrix_2d(self, K, L_max, rho_feat=0.5, rho_time=0.9, save_path="cov_matrix.pt"):
        if os.path.exists(save_path):
            print(f"[INFO] Loading precomputed CHOLESKY factor from '{save_path}'")
            L_chol = torch.load(save_path)
            return L_chol

        print(f"[INFO] Building param-based tile covariance {K}x{L_max}")
        f_indices = torch.arange(K).repeat_interleave(L_max)
        t_indices = torch.arange(L_max).repeat(K)
        dist_f = (f_indices[:, None] - f_indices[None, :]).abs()
        dist_t = (t_indices[:, None] - t_indices[None, :]).abs()
        cov = (rho_feat ** dist_f) * (rho_time ** dist_t)
        cov = cov.float()

        L_chol = torch.linalg.cholesky(cov)
        torch.save(L_chol, save_path)
        return L_chol

    # ------------------------------------------------
    # Generate a single tile
    # ------------------------------------------------
    def generate_correlated_noise_tile(self, B):
        D_tile = self.tile_k * self.tile_l
        z = torch.randn(B, D_tile, device=self.device)
        correlated = z @ self.L_chol_small[:D_tile, :D_tile].T
        return correlated.view(B, self.tile_k, self.tile_l)

    # ------------------------------------------------
    # Tiled approach if too big
    # ------------------------------------------------
    def generate_correlated_noise_2d_tiled(self, B, K, L):
        big_noise = torch.zeros(B, K, L, device=self.device)
        num_tiles_k = (K + self.tile_k -1)//self.tile_k
        num_tiles_l = (L + self.tile_l -1)//self.tile_l

        for i in range(num_tiles_k):
            for j in range(num_tiles_l):
                start_k = i*self.tile_k
                end_k = min(start_k+self.tile_k, K)
                start_l = j*self.tile_l
                end_l = min(start_l+self.tile_l, L)

                tile_noise = self.generate_correlated_noise_tile(B)
                cropped_tile = tile_noise[:, :end_k-start_k, :end_l-start_l]
                big_noise[:, start_k:end_k, start_l:end_l] = cropped_tile

        return big_noise

    def generate_correlated_noise_2d(self, B, K, L):
        if K<= self.tile_k and L<= self.tile_l:
            tile_noise = self.generate_correlated_noise_tile(B)
            return tile_noise[:, :K, :L]
        else:
            return self.generate_correlated_noise_2d_tiled(B,K,L)

    # gamma(t)
    def get_gamma(self, t_tensor):
        x = self.gamma_start + (self.gamma_end - self.gamma_start)*((t_tensor/self.num_steps)**self.gamma_tau)
        gamma = torch.sigmoid(x)
        return gamma

    # ------------------------------------------------
    # Standard rectification: Hungarian assignment
    # ------------------------------------------------
    def rectify_mapping(self, data_batch, noise_batch):
        B, K, L = data_batch.shape

        data_flat = data_batch.reshape(B, -1)
        noise_flat = noise_batch.reshape(B, -1)

        # cost => shape(B,B)
        cost = torch.cdist(data_flat, noise_flat, p=2.0)**2
        cost_np = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        # reorder noise => noise_batch[col_ind]
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

    # calc_loss => time-based blend
    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B,K,L = observed_data.shape
        if is_train != 1:
            t = torch.full((B,), float(set_t), device=self.device)
        else:
            t = torch.randint(0, self.num_steps, (B,), device=self.device).float()

        current_alpha = self.alpha_torch[t.long()]

        # 1) standard Gauss vs correlated
        noise_gauss = torch.randn(B,K,L, device=self.device)
        noise_corr = self.generate_correlated_noise_2d(B,K,L)

        # 2) gamma(t)
        gamma_vals = self.get_gamma(t).view(B,1,1)
        noise = gamma_vals*noise_gauss + (1.0-gamma_vals)*noise_corr

        # 3) standard rectification
        if self.use_rectified_mapping and is_train == 1:
            noise = self.rectify_mapping(observed_data, noise)

        # 4) forward noising
        noisy_data = (current_alpha**0.5)*observed_data + (1.0 - current_alpha)**0.5*noise

        t_int = t.long()
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t_int)

        # 5) loss
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted)*target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum()/(num_eval if num_eval>0 else 1)
        return loss

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):
            loss_t = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss_t.detach()
        return loss_sum/self.num_steps

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            return noisy_data.unsqueeze(1)
        else:
            cond_obs = (cond_mask*observed_data).unsqueeze(1)
            noisy_target = ((1-cond_mask)*noisy_data).unsqueeze(1)
            return torch.cat([cond_obs, noisy_target], dim=1)

    # Imputation => reverse prorcess
    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B,K,L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L, device=self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                t_val = torch.tensor([float(t)], device=self.device)
                gamma_val = self.get_gamma(t_val)
                gamma_t = gamma_val.view(1,1,1)

                noise_gauss = torch.randn(B,K,L, device=self.device)
                noise_corr = self.generate_correlated_noise_2d(B,K,L)
                noise_blend = gamma_t*noise_gauss + (1.0-gamma_t)*noise_corr

                if self.use_rectified_mapping:
                   noise_blend = self.rectify_mapping(observed_data, noise_blend)

                if self.is_unconditional:
                    diff_input = cond_mask*observed_data + (1-cond_mask)*current_sample
                    diff_input = diff_input.unsqueeze(1)
                else:
                    cond_obs = (cond_mask*observed_data).unsqueeze(1)
                    noisy_target = ((1-cond_mask)*current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)

                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t], device=self.device))

                coeff1 = 1 / self.alpha_hat[t]**0.5
                coeff2 = (1 - self.alpha_hat[t])/(1 - self.alpha[t])**0.5
                current_sample = coeff1*(current_sample - coeff2*predicted)

                if t>0:
                    sigma = ((1.0 - self.alpha[t-1])/(1.0 - self.alpha[t])*self.beta[t])**0.5
                    current_sample += sigma*noise_blend

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _
        ) = self.process_data(batch)

        if is_train==0:
            cond_mask = gt_mask
        elif self.target_strategy!="random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train==1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):
                target_mask[i, ..., :cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp
    

class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


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
