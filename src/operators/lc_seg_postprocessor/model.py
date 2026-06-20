import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class MLP(nn.Module):
    def __init__(self, input_dim=256, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: [B, C_in, H, W] -> [B, H*W, C_in]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class GraphConv(nn.Module):
    """
    Simple Graph Convolutional Layer.
    Refines class embeddings based on the adjacency matrix.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [B, num_nodes, in_features]
        # adj: [num_nodes, num_nodes]
        # Message passing: out = Adj * X * W
        support = self.linear(x) # [B, num_nodes, out_features]
        # Batch matrix multiplication: [B, N, N] @ [B, N, F] -> [B, N, F]
        # Since adj is constant across batch, we can use a simpler approach
        out = torch.matmul(adj, support)
        return out


class GCNRefinementModule(nn.Module):
    """
    GCN-based class feature refinement with adjusted edge weights.
    """
    def __init__(self, num_nodes, embed_dim, dag_pairs):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        # Build adjacency with smaller reverse edges
        adj = torch.eye(num_nodes)
        for u, v in dag_pairs:
            adj[u, v] = 0.8   # forward stronger
            adj[v, u] = 0.2   # reverse weaker

        # Row-normalize
        row_sum = adj.sum(1, keepdim=True)
        adj = adj / row_sum
        self.register_buffer('adj', adj)

        self.gcn1 = GraphConv(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.gcn2 = GraphConv(embed_dim, embed_dim)

    def forward(self, class_features):
        x = self.gcn1(class_features, self.adj)
        x = self.relu(x)
        x = self.gcn2(x, self.adj)
        return x


class MultiHeadGCNFormer(nn.Module):
    """
    SegFormer-style model with GCN Refinement on class-specific features.
    Supports Grad-CAM via activations/gradients hooks.
    """
    def __init__(self, dag_pairs, head_groups, encoder_name="efficientnet-b5", embed_dim=256, global_context_alpha=0.1, num_classes=16):
        super().__init__()
        self.dag_pairs = dag_pairs
        self.head_groups = head_groups
        self.num_classes = num_classes
        self.global_context_alpha = global_context_alpha   # ✅ add this
        
        # 1. Encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=None,
        )
        enc_channels = self.encoder.out_channels[1:] # Level 1-5
        
        # 2. Decoder (SegFormer-like)
        # We produce class-specific features for ALL classes first
        self.levels = [1, 2, 3, 4]
        self.mlps = nn.ModuleList([
            MLP(enc_channels[i], embed_dim) for i in self.levels
        ])
        
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(self.levels), embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        
        # Global class prototypes (learned)
        self.class_prototypes = nn.Parameter(torch.randn(self.num_classes, embed_dim))
        
        # 3. GCN Refinement
        self.gcn_refiner = GCNRefinementModule(self.num_classes, embed_dim, dag_pairs)
        
        # 4. Prediction heads
        self.heads = nn.ModuleList()
        for group in head_groups:
            # Each head predicts masks for its group + 1 background
            self.heads.append(nn.Conv2d(embed_dim, len(group) + 1, 1))

        # Grad-CAM support
        self.activations = [None] * len(head_groups)
        self.gradients = [None] * len(head_groups)

    def _save_gradient_hook(self, head_idx):
        def hook(grad):
            self.gradients[head_idx] = grad
        return hook

    def forward(self, x):
        B, _, H, W = x.shape
        features = self.encoder(x)[1:]
        
        H_feat, W_feat = features[self.levels[0]].shape[2:]
        fused = []
        for i, level_idx in enumerate(self.levels):
            feat = features[level_idx]
            mlp_out = self.mlps[i](feat)
            mlp_out = mlp_out.transpose(1, 2).reshape(B, -1, feat.shape[2], feat.shape[3])
            if feat.shape[2] != H_feat or feat.shape[3] != W_feat:
                mlp_out = F.interpolate(mlp_out, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            fused.append(mlp_out)

        spatial_features = self.linear_fuse(torch.cat(fused, dim=1))

        # --- Adjusted Global Context Weighting ---
        global_context = F.adaptive_avg_pool2d(spatial_features, (1, 1)).squeeze(-1).squeeze(-1) # [B, embed_dim]
        #alpha = 0.1  # reduce global context impact
        class_features = self.class_prototypes.unsqueeze(0) * (1 - self.global_context_alpha) + global_context.unsqueeze(1) * self.global_context_alpha

        # #attention pooling
        # attn = (spatial_features.flatten(2).transpose(1,2) @ class_features.transpose(1,2)).softmax(dim=1)
        # context = attn.transpose(1,2) @ spatial_features.flatten(2).transpose(1,2)
        refined_class_features = self.gcn_refiner(class_features)

        head_outputs = []
        for i, head in enumerate(self.heads):
            out = head(spatial_features)
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            self.activations[i] = out
            if out.requires_grad:
                out.register_hook(self._save_gradient_hook(i))
            head_outputs.append(out)

        # --- Adjusted correlation computation ---
        corr_scores = []
        for u, v in self.dag_pairs:
            x_u = refined_class_features[:, u] - refined_class_features[:, u].mean(dim=1, keepdim=True)
            x_v = refined_class_features[:, v] - refined_class_features[:, v].mean(dim=1, keepdim=True)
            score = F.cosine_similarity(x_u, x_v, dim=1)
            corr_scores.append(score)
        correlation_scores = torch.stack(corr_scores, dim=1)

        return head_outputs, correlation_scores
    
    def get_gradcam_data(self, global_class_idx):
        for h_idx, group in enumerate(self.head_groups):
            if global_class_idx in group:
                return self.activations[h_idx], self.gradients[h_idx]
        return None, None
