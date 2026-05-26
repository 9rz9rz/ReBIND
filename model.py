import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
)
from utility import get_closs

class BlockOrthogonalFeatureTransform(nn.Module):
    def __init__(self, dim=768, block_dim=64):
        super().__init__()
        assert dim % block_dim == 0

        self.dim = dim
        self.block_dim = block_dim
        self.num_blocks = dim // block_dim

        self.A_raw = nn.Parameter(
            torch.zeros(self.num_blocks, block_dim, block_dim)
        )

    def matrix(self):
        A_raw = self.A_raw.float()
        A = A_raw - A_raw.transpose(-1, -2)

        I = torch.eye(
            self.block_dim,
            device=A.device,
            dtype=A.dtype
        ).unsqueeze(0)

        Q = torch.linalg.solve(
            I + 0.5 * A,
            I - 0.5 * A
        )

        return Q.to(self.A_raw.dtype)

    def forward_with_q(self, x, Q):
        B, N, D = x.shape
        assert D == self.dim

        x_blk = x.reshape(B, N, self.num_blocks, self.block_dim)

        z_blk = torch.einsum(
            "bngd,gdk->bngk",
            x_blk,
            Q
        )

        return z_blk.reshape(B, N, D)

    def inverse_with_q(self, z, Q):
        B, N, D = z.shape
        assert D == self.dim

        z_blk = z.reshape(B, N, self.num_blocks, self.block_dim)

        x_blk = torch.einsum(
            "bngd,gkd->bngk",
            z_blk,
            Q
        )

        return x_blk.reshape(B, N, D)


def sinkhorn_log_domain(p, q, cost, reg=0.05, niter=100, tol=1e-3):
    B, M, _ = cost.shape
    device = cost.device

    def lse(A):
        max_A = A.max(dim=-1, keepdim=True)[0]
        return torch.log(torch.exp(A - max_A).sum(dim=-1, keepdim=True) + 1e-10) + max_A

    u = torch.zeros(B, M, device=device)
    v = torch.zeros(B, M, device=device)

    for _ in range(niter):
        u_prev = u
        M = (-cost + u.unsqueeze(2) + v.unsqueeze(1)) / reg 
        u = reg * (torch.log(p) - lse(M).squeeze(-1)) + u 

        M = (-cost + u.unsqueeze(2) + v.unsqueeze(1)) / reg
        v = reg * (torch.log(q) - lse(M.transpose(-2, -1)).squeeze(-1)) + v

        if (u - u_prev).abs().max() < tol:
            break

    M_final = (-cost + u.unsqueeze(2) + v.unsqueeze(1)) / reg
    pi = M_final.exp()
    return pi


def partial_ot(cost, rho=0.8, reg=0.05, niter=100):
    B, M, _ = cost.shape
    device = cost.device

    xi = 1e2 * cost.max()
    A = cost.max()

    p = torch.ones(B, M, device=device) / M
    q = torch.ones(B, M, device=device) / M

    C_aug = torch.cat([cost, xi * torch.ones(B, M, 1, device=device)], dim=2)  
    C_aug = torch.cat([C_aug, xi * torch.ones(B, 1, M + 1, device=device)], dim=1)  
    C_aug[:, -1, -1] = 2 * xi + A

    dustbin_mass = (p.sum(dim=1) - rho).unsqueeze(1) 
    p_aug = torch.cat([p, dustbin_mass], dim=1)  
    q_aug = torch.cat([q, dustbin_mass], dim=1)  

    pi_aug = sinkhorn_log_domain(p_aug, q_aug, C_aug, reg=reg, niter=niter)
    pi = pi_aug[:, :-1, :-1]
    return pi



class MaskPredictor(nn.Module):
    def __init__(self, d_model_img=384, d_model_txt=768, nhead=4, num_blocks=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.text_proj = nn.Linear(d_model_txt, d_model_img)
        self.ca = nn.MultiheadAttention(embed_dim=d_model_img, num_heads=nhead,
                                        batch_first=True)
        self.norm = nn.LayerNorm(d_model_img)
        self.mlp = nn.Sequential(
            nn.Linear(d_model_img, d_model_img // 2),
            nn.GELU(),
            nn.Linear(d_model_img // 2, num_blocks),
        )

    def forward(self, ref_latent, text_embeds, text_attention_mask):
        text_features = self.text_proj(text_embeds)
        key_padding_mask = (text_attention_mask == 0)
        attn_out, _ = self.ca(query=ref_latent, key=text_features,
                              value=text_features, key_padding_mask=key_padding_mask)
        fused = self.norm(ref_latent + attn_out)
        logit_dynamic = self.mlp(fused)
        mask_static = torch.sigmoid(-logit_dynamic) 
        mask_dynamic = 1.0 - mask_static
        return mask_static, mask_dynamic, logit_dynamic


@registry.register_model("cirmodel")
class CIRModel(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        lrm=1.0,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.d2t_proj = nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

        self.ref_transform = None
        self._Q = None

        self.swap_fused_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.swap_tgt_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.swap_alpha = 0.0 
        self.lrm = lrm

        self.swap_warmup_epochs = None
        self.inference_balance = None
        self.inn_rho = None 
        self.inn_sinkhorn_reg = None
        self.inn_sinkhorn_iter = None
        self.inn_block_dim = None
        self.similarity_option = None
        self.aux_loss_option = None

    def build_inn(self, inn_block_dim):
        D = self.Qformer.config.hidden_size 
        assert D % inn_block_dim == 0, f"hidden_size {D} must be divisible by inn_block_dim {inn_block_dim}"
        self.inn_block_dim = inn_block_dim
        self.ref_transform = BlockOrthogonalFeatureTransform(dim=D, block_dim=inn_block_dim)
        self.ref_transform = self.ref_transform.to(
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype)
        self.inn_block_num = D // inn_block_dim
        self.mask_predictor = MaskPredictor(d_model_img=D, d_model_txt=D, nhead=4, num_blocks=self.inn_block_num)
        self.mask_predictor = self.mask_predictor.to(
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype)

    def _map_forward(self, x):
        Q = self._Q.to(device=x.device, dtype=x.dtype)
        return self.ref_transform.forward_with_q(x, Q)

    def _map_reverse(self, x):
        Q = self._Q.to(device=x.device, dtype=x.dtype)
        return self.ref_transform.inverse_with_q(x, Q)
    
    def _tokens_to_block_units(self, x):
        """
        x: (B, N, D)
        return: (B, N*G, block_dim)
        """
        B, N, D = x.shape
        G = self.ref_transform.num_blocks
        block_dim = self.ref_transform.block_dim

        assert D == G * block_dim

        return x.reshape(B, N, G, block_dim).reshape(B, N * G, block_dim)


    def _block_mask_to_feature_mask(self, mask_blk):

        block_dim = self.ref_transform.block_dim

        return mask_blk.repeat_interleave(
            repeats=block_dim,
            dim=-1,
        )

    def _compute_swap_infonce(self, q_latent, k_latent, labels, pn_loss, aux_loss_option):
        q_rev = self._map_reverse(q_latent)    
        k_rev = self._map_reverse(k_latent)    
        z_q = F.normalize(self.vision_proj(q_rev).mean(dim=1), dim=-1)  
        z_k = F.normalize(self.vision_proj(k_rev).mean(dim=1), dim=-1)
        sim = torch.matmul(z_q, z_k.T)                                  
        return self.robust_infoNCE(sim, labels, pn_loss)

    @torch.no_grad()
    def vit_encode(self, image):
        return self.visual_encoder(image)

    # Image Encoder
    def encode_image(self, image_embeds, query_tokens=None, ln=True):
        if ln:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(image_embeds)
        if query_tokens is None:
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )
        image_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return image_output.last_hidden_state
    
    def encode_fusion(self, F_image, text_tokens, 
                      no_image=False, diff_embeds=None, clean_label=None, pn_loss=None):
        bs = text_tokens.input_ids.shape[0]
        image_atts = torch.ones(F_image.shape[:-1], dtype=torch.long).to(
            F_image.device
        )
        attention_mask = torch.cat([image_atts, text_tokens.attention_mask], dim=1)
        if diff_embeds is not None:
            diff_atts = torch.ones(diff_embeds.shape[:-1], dtype=torch.long).to(
                F_image.device
            )
            attention_mask = torch.cat([image_atts, diff_atts], dim=1)
        assert F_image.shape[:-1] == (bs, 32)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=F_image,
            attention_mask=attention_mask,
            return_dict=True,
            no_img=no_image,
            image_diff=diff_embeds,
            clean_label=clean_label,
            pn_loss=pn_loss,
        )
        if diff_embeds is not None:
            fusion_output, lsa = fusion_output
        token_num = 0 if no_image else 32
        res = F.normalize(self.text_proj(fusion_output.last_hidden_state[:, token_num, :]), dim=-1)
        res = (res, lsa) if diff_embeds is not None else res
        return res

    def encode_fusion_tokens(self, F_image, text_tokens):
        bs = F_image.shape[0]
        image_atts = torch.ones(F_image.shape[:-1], dtype=torch.long).to(F_image.device)
        attention_mask = torch.cat([image_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=F_image,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return fusion_output.last_hidden_state[:, :32, :]

    @torch.no_grad()
    def per_loss(self, reference_embeds, target_embeds, captions):
        F_r = self.encode_image(reference_embeds)
        F_t = self.encode_image(target_embeds)
        sim_i2t = self.inference(F_r, F_t, captions, self.similarity_option)
        loss = - (sim_i2t / self.temp).log_softmax(1).diag()
        return loss, sim_i2t.diag()

    def robust_infoNCE(self, scores, labels, pn_loss):
        eps=1e-7
        self.temp.data = torch.clamp(self.temp.data, min=1e-2)
        scores = scores.float()
        i2t = (scores / self.temp).softmax(1)
        i2t = torch.clamp(i2t, min=eps, max=1-eps)
        target=torch.arange(scores.shape[0]).to(scores.device)
        clean_mask = labels.to(bool)
        noise_mask = ~clean_mask
        ploss = get_closs(i2t[clean_mask], target[clean_mask], pn_loss['positive_loss'])
        nloss = get_closs(i2t[noise_mask], target[noise_mask], pn_loss['negative_loss'])
        trade_off = pn_loss['trade_off']
        return trade_off * ploss + (1 - trade_off) * nloss

    def forward(self, samples, labels=None, pn_loss=None, warmup=None):
        image_embeds = samples["image"]
        target_embeds = samples["target"]
        text = samples["text_input"]
        image_embeds = self.ln_vision(image_embeds)
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        text_tokens = text_tokens.to(image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        F_reference = self.encode_image(image_embeds, query_tokens, ln=False)
        target_embeds = self.ln_vision(target_embeds)
        F_target = self.encode_image(target_embeds, query_tokens, ln=False)

        text_outputs = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_embeds = text_outputs.last_hidden_state

        self._Q = self.ref_transform.matrix()
        ref_latent = self._map_forward(F_reference) 
        tgt_latent = self._map_forward(F_target) 

        B, N, D = ref_latent.shape
        G = self.ref_transform.num_blocks 
        block_dim = self.ref_transform.block_dim 
        M = N * G 

        ref_units = ref_latent.reshape(B, N, G, block_dim).reshape(B, M, block_dim)
        tgt_units = tgt_latent.reshape(B, N, G, block_dim).reshape(B, M, block_dim)
        
        ref_norm = F.normalize(ref_units, dim=-1)
        tgt_norm = F.normalize(tgt_units, dim=-1) 
        
        sim = torch.matmul(ref_norm, tgt_norm.transpose(-2, -1))
        cost = 1.0 - sim
        with torch.cuda.amp.autocast(enabled=False):
            Pi = partial_ot(cost.float(), rho=self.inn_rho,
                            reg=self.inn_sinkhorn_reg, niter=self.inn_sinkhorn_iter)
            Pi = Pi.to(ref_latent.dtype)

        sum_tgt_mass = Pi.sum(dim=1)  
        sum_ref_mass = Pi.sum(dim=2)
        
        mask_ref_static_ot = torch.clamp(sum_ref_mass * M / self.inn_rho, 0.0, 1.0)
        mask_tgt_static_ot = torch.clamp(sum_tgt_mass * M / self.inn_rho, 0.0, 1.0)
        
        mask_ref_static_ot = mask_ref_static_ot.reshape(B, N, G)
        mask_tgt_static_ot = mask_tgt_static_ot.reshape(B, N, G)

        mask_ref_dynamic_ot = 1.0 - mask_ref_static_ot
        mask_tgt_dynamic_ot = 1.0 - mask_tgt_static_ot

        loss_dict = {}
        loss_dict["lortho"] = torch.tensor(0.0, device=image_embeds.device)

        mask_ref_static_pred, mask_ref_dynamic_pred, logit_dynamic = self.mask_predictor(
            ref_latent,
            text_embeds,
            text_tokens.attention_mask,
        )

        with torch.cuda.amp.autocast(enabled=False):
            lmask = F.binary_cross_entropy_with_logits(
                (-logit_dynamic).float(), mask_ref_static_ot.float())
        loss_dict["lmask"] = lmask

        alpha = self.swap_alpha

        mask_ref_static_use = (
            alpha * mask_ref_static_pred
            + (1.0 - alpha) * mask_ref_static_ot
        )

        mask_ref_dynamic_use = 1.0 - mask_ref_static_use

        mask_ref_static_full = self._block_mask_to_feature_mask(mask_ref_static_use)
        mask_ref_dynamic_full = 1.0 - mask_ref_static_full

        mask_tgt_static_full = self._block_mask_to_feature_mask(mask_tgt_static_ot)
        mask_tgt_dynamic_full = 1.0 - mask_tgt_static_full

        ref_static = ref_latent * mask_ref_static_full
        ref_dynamic = ref_latent * mask_ref_dynamic_full

        tgt_static = tgt_latent * mask_tgt_static_full
        tgt_dynamic = tgt_latent * mask_tgt_dynamic_full

        dynamic_fused = self.encode_fusion_tokens(
            ref_dynamic, text_tokens)   

        fused_main = ref_static + dynamic_fused      
        fused_t2r  = tgt_static + dynamic_fused  
        fused_r2t  = ref_static + tgt_dynamic     
        fused_key2 = tgt_static + dynamic_fused   

        F_ref_orig = F_reference
        F_tgt_orig = F_target

        F_reference = self._map_reverse(ref_latent) 
        F_target = self._map_reverse(tgt_latent) 

        z_target = F.normalize(self.vision_proj(F_target), dim=-1)

        fused_main_rev = self._map_reverse(fused_main)
        z_query = F.normalize(self.vision_proj(fused_main_rev[:, 0, :]), dim=-1)
        z_tgt_tok = F.normalize(self.vision_proj(F_tgt_orig), dim=-1)
        if self.similarity_option == 'default':
            sim_main = torch.matmul(
                z_query.unsqueeze(1).unsqueeze(1), z_tgt_tok.permute(0, 2, 1)
            ).squeeze()
            sim_main, _ = sim_main.max(-1)
        elif self.similarity_option == 'n2nmean':
            bsz = fused_main_rev.shape[0]
            q = F.normalize(self.vision_proj(fused_main_rev), dim=-1) 
            q = q.reshape(bsz, -1)
            k = F.normalize(self.vision_proj(F_tgt_orig), dim=-1) 
            k = k.reshape(bsz, -1)
            sim_main = torch.mm(q, k.T) / 32.0
        elif self.similarity_option == 'mean121':
            q = F.normalize(self.vision_proj(fused_main_rev), dim=-1).mean(dim=1)
            k = F.normalize(self.vision_proj(F_tgt_orig), dim=-1).mean(dim=1)
            sim_main = torch.mm(q, k.T)
        elif self.similarity_option == 'modified_original':
            sim_main = torch.einsum("qd,ntd->qnt", z_query, z_tgt_tok).max(dim=-1).values
        else:
            raise ValueError(f"Invalid similarity_option {self.similarity_option} passed to model.")

        lswap_main = self.robust_infoNCE(sim_main, labels, pn_loss)

        lswap_aux1 = self._compute_swap_infonce(fused_t2r, fused_r2t, labels, pn_loss, self.aux_loss_option)
        lswap_aux2 = self._compute_swap_infonce(fused_r2t, fused_key2, labels, pn_loss, self.aux_loss_option)

        loss_dict["lswap_main"] = lswap_main
        loss_dict["lswap_aux1"] = lswap_aux1
        loss_dict["lswap_aux2"] = lswap_aux2
        return loss_dict

    @torch.no_grad()
    def inference(self, F_reference, F_target, text, similarity_option):
        assert similarity_option is not None, "similarity_option must not be none"

        text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(F_reference.device)

        z_target = F.normalize(self.vision_proj(F_target), dim=-1) 

        text_outputs = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_embeds = text_outputs.last_hidden_state 

        self._Q = self.ref_transform.matrix()
        ref_latent = self._map_forward(F_reference)     

        mask_static_blk, mask_dynamic_blk, _ = self.mask_predictor(
            ref_latent,
            text_embeds,
            text_tokens.attention_mask,
        )  

        
        mask_static_full = self._block_mask_to_feature_mask(mask_static_blk)
        mask_dynamic_full = 1.0 - mask_static_full

        ref_static = ref_latent * mask_static_full
        ref_dynamic = ref_latent * mask_dynamic_full
        dynamic_fused = self.encode_fusion_tokens(
            ref_dynamic, text_tokens)  

        fused = ref_static + dynamic_fused    
        F_fused = self._map_reverse(fused)    
        z_rm_new = F.normalize(self.vision_proj(F_fused[:, 0, :]), dim=-1) 

        sim_new = torch.einsum("qd,ntd->qnt", z_rm_new, z_target).max(dim=-1).values
        return sim_new

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        max_txt_len = cfg.get("max_txt_len", 32)
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
