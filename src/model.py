# model.py
import math

import torch
import torch.nn as nn
import torchvision.models as models

# import timm
from timm.models.vision_transformer import VisionTransformer

import config


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim).to(config.DEVICE)
        position = (
            torch.arange(0, max_len, dtype=torch.float).to(config.DEVICE).unsqueeze(1)
        )
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        ).to(config.DEVICE)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, embedding_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embedding_dim)
        x = x + self.pe[: x.size(0), :]
        return x


class StateEncoder(nn.Module):
    """Encodes past states and intent into a feature vector."""

    def __init__(self):
        super().__init__()

        layers = []
        num_layers = config.INTENT_LAYERS
        for i in range(num_layers):
            in_dim = (
                config.INTENT_CLASSES
                if i == 0
                else config.INTENT_EMBEDDING_DIM * (num_layers - i + 1)
            )
            out_dim = config.INTENT_EMBEDDING_DIM * (num_layers - i)
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.INTENT_LAYERS_DROPOUT))

        self.intent_encoder = nn.Sequential(*layers)

        layers = []
        num_layers = config.STATE_LAYERS
        for i in range(num_layers):
            in_dim = (
                config.NUM_PAST_STATES
                if i == 0
                else config.STATE_EMBEDDING_DIM * (num_layers - i + 1)
            )
            out_dim = config.STATE_EMBEDDING_DIM * (num_layers - i)
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.STATE_LAYERS_DROPOUT))

        self.past_state_encoder = nn.Sequential(*layers)

    def forward(self, batch):
        intent = batch[
            config.ProcessedDataColumn.INTENT
        ]  # shape: (B, NUM_INTENT_CLASSES)
        intent_embedding = self.intent_encoder(intent)

        # Flatten past states sequence for MLP
        past_states = batch[
            config.ProcessedDataColumn.PAST_STATES
        ]  # shape: (B, 3, 16, 2)
        past_states_flat = past_states.view(past_states.shape[0], -1)
        past_embedding = self.past_state_encoder(past_states_flat)

        # Concatenate state embeddings
        state_embedding = torch.cat(
            [past_embedding, intent_embedding], dim=1
        )  # (B, embedding_dim)

        batch[config.ProcessedDataColumn.STATE_EMBEDDING] = state_embedding

        return state_embedding


class ImageEncoder(nn.Module):
    """Encodes multiple camera images into a single feature vector."""

    def __init__(self, dry_run=False):
        super().__init__()

        self.dry_run = dry_run

        # Use a pre-trained ResNet, remove the final classification layer
        # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.backbone = nn.Sequential(
        #     *list(resnet.children())[:-1]
        # )  # Remove final fc layer
        # num_features = resnet.fc.in_features  # Get feature size from original fc layer

        squeeze_net = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            *list(squeeze_net.children())[:-1][0],  # Remove final classifier
            nn.AdaptiveAvgPool2d(
                (config.IMAGE_RESNET_FINAL_MAP_DIM, config.IMAGE_RESNET_FINAL_MAP_DIM)
            ),
        )
        token_dim = squeeze_net.classifier[
            1
        ].in_channels  # Get feature size from original classifier

        # Pretrained ViT
        # self.vit = timm.create_model(
        #     "vit_base_patch16_224",
        #     pretrained=True,
        #     num_classes=0,
        #     img_size=16,
        #     in_chans=512,
        # )
        # self.fusion_layer = nn.Linear(
        #     768 * config.NUM_CAMERAS, config.IMAGE_EMBEDDING_DIM
        # )

        self.vit = VisionTransformer(
            img_size=(
                config.IMAGE_RESNET_FINAL_MAP_DIM,
                config.IMAGE_RESNET_FINAL_MAP_DIM,
            ),  # Critical for pos_embed size
            patch_size=1,  # Each feature vector from CNN is a "patch"
            in_chans=token_dim,  # Input channels to ViT's patch_embed
            num_classes=0,
            embed_dim=token_dim,  # Dimension of tokens fed to Transformer blocks
            depth=config.VIT_DEPTH,  # Number of Transformer blocks
            num_heads=config.VIT_NUM_HEADS,  # Number of attention heads
            mlp_ratio=config.VIT_MLP_RATIO,  # Ratio of hidden dim to embed dim in MLP
            qkv_bias=True,
            drop_rate=0.1,
            patch_drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=lambda channels: nn.LayerNorm(
                channels, eps=1e-6
            ),  # Standard LayerNorm
            # The default `embed_layer=PatchEmbed` will create a Conv2d projection.
            # For patch_size=1, this proj is nn.Conv2d(cnn_out_chans, cnn_out_chans, kernel_size=1, stride=1)
            # This 1x1 convolution is fine and can learn a useful per-token linear transformation.
        )
        self.vit.patch_embed = nn.Identity()

        if config.USE_IMAGE_FUSION_TRANSFORMER:
            # Transformer fusion
            self.fusion_positional_encoder = PositionalEncoding(
                embedding_dim=token_dim, max_len=config.NUM_CAMERAS
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=config.IMAGE_FUSION_TRANSFORMER_HEADS,
                dim_feedforward=config.IMAGE_FUSION_TRANSFORMER_FF_DIM,
                batch_first=True,
                device=config.DEVICE,
            )
            self.fusion_transformer_encoder = nn.TransformerEncoder(
                encoder_layer, config.IMAGE_FUSION_TRANSFORMER_LAYERS
            )
            fused_dim = token_dim
        else:
            fused_dim = token_dim * config.NUM_CAMERAS

        # Project output to desired embedding dim
        self.final_layer = nn.Linear(fused_dim, config.IMAGE_EMBEDDING_DIM)

    def forward(self, batch):
        batch_size = batch[config.ProcessedDataColumn.INTENT].shape[0]

        image_features = []
        for column in config.DataColumn:
            if not column.name.startswith("IMAGE"):
                continue
            image = batch[config.ProcessedDataColumn.IMAGES][
                column
            ]  # shape: (B, C, H, W)

            if config.EARLY_STATE_FUSION:
                # Add as additional row in image, to all channels
                intent = batch[
                    config.ProcessedDataColumn.INTENT
                ]  # shape: (B, NUM_INTENT_CLASSES)
                past_states = batch[
                    config.ProcessedDataColumn.PAST_STATES
                ]  # shape: (B, 3, 16, 2)
                past_states_flat = past_states.view(batch_size, -1)
                state = torch.cat([intent, past_states_flat], dim=1)
                _, C, H, W = image.shape
                early_state_fusion_features = torch.zeros(
                    (batch_size, C, 1, W), dtype=image.dtype, device=image.device
                )
                early_state_fusion_features[:, :, :, 0 : state.shape[-1]] = (
                    state.reshape(batch_size, 1, 1, -1)
                )
                image = torch.cat([image, early_state_fusion_features], dim=2)

            conv_features = self.backbone(image)  # (B, C, H', W'), [64, 512, 16, 16]

            tokens = conv_features.flatten(2).transpose(
                1, 2
            )  # (B, N_tokens, C), [64, 256, 512]
            # Add positional encoding
            pos_embed = self.vit.pos_embed[
                :, : tokens.size(1), : tokens.size(2)
            ]  # (1, N_tokens, C), [1, 256, 512]
            tokens = tokens + pos_embed  # (B, N_tokens, C), [64, 256, 512]
            tokens = self.vit.patch_drop(tokens)  # (B, N_tokens, C), [64, 256, 512]
            # Pass through transformer blocks
            tokens = self.vit.blocks(tokens)  # (B, N_tokens, C), [64, 256, 512]
            tokens = self.vit.norm(tokens)  # (B, N_tokens, C), [64, 256, 512]
            # Pool tokens (mean)
            pooled = tokens.mean(dim=1)  # (B, C), [64, 512]
            image_features.append(pooled)

            # # Pretrained ViT
            # tokens = conv_features
            # vit_features = self.vit(tokens)
            # image_features.append(vit_features)

            if self.dry_run:
                break

        stacked_features = torch.stack(
            image_features, dim=1
        )  # (B, num_cameras, C), [64, 8, 512]
        if self.dry_run:
            stacked_features = torch.repeat_interleave(
                stacked_features, config.NUM_CAMERAS, dim=1
            )

        # Simple Averaging Fusion #
        # image_embedding = stacked_features.mean(dim=1)

        # Transformer Fusion Layer #
        if config.USE_IMAGE_FUSION_TRANSFORMER:
            # PositionalEncoding expects (seq_len, batch_size, d_model), then convert
            # back for TransformerEncoder (batch_first=True)
            image_embedding_with_pos = self.fusion_positional_encoder(
                stacked_features.transpose(0, 1)
            ).transpose(0, 1)
            transformer_output = self.fusion_transformer_encoder(
                image_embedding_with_pos
            )  # (batch_size, seq_len + 1, embedding_dim)
            # This is analogous to using the [CLS] token in BERT for sequence-level classification
            # tasks, where the model learns to aggregate the fused information into this specific
            # token's representation. Alternative is to take the mean
            image_embedding = transformer_output[:, 0, :]  # (batch_size, embedding_dim)
        else:
            image_embedding = stacked_features.view(batch_size, -1)

        image_embedding = self.final_layer(image_embedding)

        batch[config.ProcessedDataColumn.IMAGE_EMBEDDING] = image_embedding

        return image_embedding


class TrajectoryDecoder(nn.Module):
    def __init__(self, combined_embedding_dim):
        super().__init__()

        # Decoder MLP to predict trajectory points
        layers = []
        num_layers = config.TRAJECTORY_DECODER_LAYERS
        for i in range(num_layers):
            in_dim = (
                combined_embedding_dim
                if i == 0
                else config.TRAJECTORY_DECODER_EMBEDDING_DIM * (num_layers - i + 1)
            )
            out_dim = config.TRAJECTORY_DECODER_EMBEDDING_DIM * (num_layers - i)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.TRAJECTORY_DECODER_LAYERS_DROPOUT))

        num_future_states = config.NUM_FUTURE_STATES
        layers.append(
            nn.Linear(config.TRAJECTORY_DECODER_EMBEDDING_DIM, num_future_states)
        )

        self.trajectory_decoder = nn.Sequential(*layers)

    def forward(self, batch):
        combined_embedding = batch[config.ProcessedDataColumn.COMBINED_EMBEDDING]
        flat_output = self.trajectory_decoder(
            combined_embedding
        )  # (B, num_future_states)

        # Reshape trajectory: (B, future_steps, future_dim)
        start_idx = 0
        end_idx = config.NUM_FUTURE_STATES
        predicted_trajectory = flat_output[:, start_idx:end_idx]
        predicted_trajectory = predicted_trajectory.view(
            -1, config.NUM_FUTURE_SECONDS * config.HERTZ, config.FUTURE_STATE_DIM
        )

        return predicted_trajectory


class GMMTrajectoryDecoder(nn.Module):
    def __init__(self, combined_embedding_dim):
        super().__init__()

        # Decoder MLP to predict trajectory points
        layers = []
        num_layers = config.GMM_TRAJECTORY_DECODER_LAYERS
        for i in range(num_layers):
            in_dim = (
                combined_embedding_dim
                if i == 0
                else config.GMM_TRAJECTORY_DECODER_EMBEDDING_DIM * (num_layers - i + 1)
            )
            out_dim = config.GMM_TRAJECTORY_DECODER_EMBEDDING_DIM * (num_layers - i)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.GMM_TRAJECTORY_DECODER_LAYERS_DROPOUT))

        num_future_states = config.GMM_COMPONENTS + (
            config.NUM_FUTURE_STATES * config.GMM_COMPONENTS * 2
        )  # mu and ln_var
        layers.append(
            nn.Linear(config.GMM_TRAJECTORY_DECODER_EMBEDDING_DIM, num_future_states)
        )

        self.gmm_trajectory_decoder = nn.Sequential(*layers)

    def forward(self, batch):
        combined_embedding = batch[config.ProcessedDataColumn.COMBINED_EMBEDDING]
        flat_output = self.gmm_trajectory_decoder(
            combined_embedding
        )  # (B, num_future_states)

        start_idx = 0
        end_idx = start_idx + config.GMM_COMPONENTS
        predicted_trajectory_gmm_coef = flat_output[:, start_idx:end_idx]
        # (B, GMM_COMPONENTS)

        start_idx = end_idx
        end_idx = start_idx + (config.GMM_COMPONENTS * config.NUM_FUTURE_STATES)
        predicted_trajectory_gmm_mu = flat_output[:, start_idx:end_idx]
        predicted_trajectory_gmm_mu = predicted_trajectory_gmm_mu.view(
            -1,
            config.GMM_COMPONENTS,
            config.NUM_FUTURE_SECONDS * config.HERTZ,
            config.FUTURE_STATE_DIM,
        )  # (B, GMM_COMPONENTS, future_steps, future_dim)

        start_idx = end_idx
        end_idx = start_idx + (config.GMM_COMPONENTS * config.NUM_FUTURE_STATES)
        predicted_trajectory_gmm_ln_var = flat_output[:, start_idx:end_idx]
        predicted_trajectory_gmm_ln_var = predicted_trajectory_gmm_ln_var.view(
            -1,
            config.GMM_COMPONENTS,
            config.NUM_FUTURE_SECONDS * config.HERTZ,
            config.FUTURE_STATE_DIM,
        )  # (B, GMM_COMPONENTS, future_steps, future_dim)

        predicted_trajectory_gmm = (
            predicted_trajectory_gmm_coef,
            predicted_trajectory_gmm_mu,
            predicted_trajectory_gmm_ln_var,
        )

        return predicted_trajectory_gmm


class MNLLTrajectoryDecoder(nn.Module):
    def __init__(self, combined_embedding_dim):
        super().__init__()

        # Decoder MLP to predict trajectory points
        layers = []
        num_layers = config.MNLL_TRAJECTORY_DECODER_LAYERS
        for i in range(num_layers):
            in_dim = (
                combined_embedding_dim
                if i == 0
                else config.MNLL_TRAJECTORY_DECODER_EMBEDDING_DIM * (num_layers - i + 1)
            )
            out_dim = config.MNLL_TRAJECTORY_DECODER_EMBEDDING_DIM * (num_layers - i)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.MNLL_TRAJECTORY_DECODER_LAYERS_DROPOUT))

        num_future_states = config.MNLL_MODES + (
            config.NUM_FUTURE_STATES * config.MNLL_MODES
        )

        layers.append(
            nn.Linear(config.MNLL_TRAJECTORY_DECODER_EMBEDDING_DIM, num_future_states)
        )

        self.mnll_trajectory_decoder = nn.Sequential(*layers)

    def forward(self, batch):
        combined_embedding = batch[config.ProcessedDataColumn.COMBINED_EMBEDDING]
        flat_output = self.mnll_trajectory_decoder(
            combined_embedding
        )  # (B, num_future_states)

        start_idx = 0
        end_idx = start_idx + config.MNLL_MODES
        predicted_trajectory_mnll_mode_prob = flat_output[:, start_idx:end_idx]
        # (B, MNLL_MODES)

        start_idx = end_idx
        end_idx = start_idx + (config.MNLL_MODES * config.NUM_FUTURE_STATES)
        predicted_trajectory_mnll_traj = flat_output[:, start_idx:end_idx]
        predicted_trajectory_mnll_traj = predicted_trajectory_mnll_traj.view(
            -1,
            config.MNLL_MODES,
            config.NUM_FUTURE_SECONDS * config.HERTZ,
            config.FUTURE_STATE_DIM,
        )  # (B, MNLL_MODES, future_steps, future_dim)

        predicted_trajectory_mnll = (
            predicted_trajectory_mnll_mode_prob,
            predicted_trajectory_mnll_traj,
        )

        return predicted_trajectory_mnll


class EndToEndDriver(nn.Module):
    """Combines image and state features to predict future trajectory."""

    def __init__(self, dry_run=False):
        super().__init__()

        self.dry_run = dry_run

        self.state_encoder = StateEncoder()
        self.image_encoder = ImageEncoder(dry_run=dry_run)

        combined_embedding_dim = (
            config.INTENT_EMBEDDING_DIM
            + config.STATE_EMBEDDING_DIM
            + config.IMAGE_EMBEDDING_DIM
        )

        self.trajectory_decoder = TrajectoryDecoder(combined_embedding_dim)
        if config.GMM_TRAJECTORY_DECODER_ENABLED:
            self.gmm_trajectory_decoder = GMMTrajectoryDecoder(combined_embedding_dim)
        if config.MNLL_TRAJECTORY_DECODER_ENABLED:
            self.mnll_trajectory_decoder = MNLLTrajectoryDecoder(combined_embedding_dim)

    def forward(self, batch):
        # Encode inputs
        state_embedding = self.state_encoder(batch)  # (B, state_embedding_dim)
        image_embedding = self.image_encoder(batch)  # (B, image_embedding_dim)

        # Concatenate features
        combined_embedding = torch.cat(
            [image_embedding, state_embedding], dim=1
        )  # (B, total_embedding_dim)
        batch[config.ProcessedDataColumn.COMBINED_EMBEDDING] = combined_embedding

        # Decode trajectory
        predicted_trajectory = self.trajectory_decoder(
            batch
        )  # (B, future_steps, future_dim)
        batch[config.PredictedDataColumn.FUTURE_STATES] = predicted_trajectory

        if config.GMM_TRAJECTORY_DECODER_ENABLED:
            predicted_trajectory_gmm = self.gmm_trajectory_decoder(batch)
            batch[config.PredictedDataColumn.FUTURE_STATES_GMM] = (
                predicted_trajectory_gmm
            )
        else:
            batch[config.PredictedDataColumn.FUTURE_STATES_GMM] = None

        if config.MNLL_TRAJECTORY_DECODER_ENABLED:
            predicted_trajectory_mnll = self.mnll_trajectory_decoder(batch)
            batch[config.PredictedDataColumn.FUTURE_STATES_MNLL] = (
                predicted_trajectory_mnll
            )
        else:
            batch[config.PredictedDataColumn.FUTURE_STATES_MNLL] = None

        return batch
