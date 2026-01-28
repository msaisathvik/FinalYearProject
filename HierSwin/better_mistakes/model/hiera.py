import torch
import torch.nn as nn
import timm

class CrossLevelAttentionFusion(nn.Module):
    def __init__(self, high_level_dim, low_level_dim, num_heads=4):
        super(CrossLevelAttentionFusion, self).__init__()
        self.query = nn.Linear(high_level_dim, low_level_dim)
        self.key = nn.Linear(low_level_dim, low_level_dim)
        self.value = nn.Linear(low_level_dim, low_level_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=low_level_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(low_level_dim)
        self.ffn = nn.Sequential(
            nn.Linear(low_level_dim, low_level_dim * 4),
            nn.ReLU(),
            nn.Linear(low_level_dim * 4, low_level_dim)
        )
        self.norm2 = nn.LayerNorm(low_level_dim)

    def forward(self, high_level_features, low_level_features):
        # high_level_features: [B, H_dim] -> [B, 1, H_dim] (as query)
        # low_level_features: [B, L_dim] (we might need to treat this as sequence or just project)
        
        # For simplicity in this classification context where features are often pooled:
        # We will treat the high level prediction/embedding as a context to refine the low level feature.
        # But wait, usually attention is Q(Target) attending to K,V(Source).
        # Here we want Level 1 (Coarse) to guide Level 2 (Fine).
        # So Q = Level 1 embedding, K, V = Backbone Features (before final pooling if possible, or just the feature vector)
        
        # Let's assume input features are already pooled vectors for simplicity in this first version, 
        # or we can project them to sequence format if we had spatial maps.
        # Given the SwinT output is usually a vector (num_classes) or embedding (num_features), let's work with embeddings.
        
        # If inputs are 1D vectors [B, Dim]:
        q = self.query(high_level_features).unsqueeze(1) # [B, 1, L_dim]
        k = self.key(low_level_features).unsqueeze(1)    # [B, 1, L_dim]
        v = self.value(low_level_features).unsqueeze(1)  # [B, 1, L_dim]
        
        attn_out, _ = self.multihead_attn(q, k, v)
        
        # Residual connection + Norm
        out = self.norm(k + attn_out) # Add to original features (K)
        
        # FFN
        out = out + self.ffn(out)
        out = self.norm2(out)
        
        return out.squeeze(1)

class HieRA(nn.Module):
    def __init__(self, num_classes_l1, num_classes_l2, num_classes_l3, pretrained=True):
        super(HieRA, self).__init__()
        # Backbone
        self.backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=pretrained, num_classes=0)
        self.feature_dim = 1536 # Swin-Large feature dim
        
        # Level 1 (Coarse) Classifier
        self.head_l1 = nn.Linear(self.feature_dim, num_classes_l1)
        
        # CLAF for Level 2
        self.claf_l2 = CrossLevelAttentionFusion(self.feature_dim, self.feature_dim)
        self.head_l2 = nn.Linear(self.feature_dim, num_classes_l2)
        
        # CLAF for Level 3
        self.claf_l3 = CrossLevelAttentionFusion(self.feature_dim, self.feature_dim)
        self.head_l3 = nn.Linear(self.feature_dim, num_classes_l3)

    def forward(self, x):
        # Extract features
        features = self.backbone(x) # [B, 1536]
        
        # Level 1 Prediction
        logits_l1 = self.head_l1(features)
        
        # Level 2 Prediction (Guided by Level 1 context - here using features as context proxy)
        # Ideally we use the L1 logits or embedding. Let's use the features refined by L1 "intent" if we had a separate embedding.
        # For now, let's self-attend: "Given the global features, attend to relevant parts for L2"
        # A better design: Use L1 logits to weight features? 
        # Let's stick to the proposed CLAF: L1 features guiding L2.
        # Since we share the backbone, "L1 features" are just 'features'. 
        # Let's make it distinct:
        # L1 uses raw backbone features.
        # L2 uses backbone features refined by L1 context.
        
        # Refine features for L2
        # We treat 'features' as both source and target, but we could inject L1 logits info.
        # Let's inject L1 logits into the query for L2.
        # Q = Linear(logits_l1) -> projected to feature dim
        # K, V = features
        
        # Re-implementing CLAF usage logic inside forward:
        
        # L2
        # Project L1 logits to feature dim to act as a "Guide"
        l1_guide = torch.matmul(torch.softmax(logits_l1, dim=1), self.head_l1.weight) # [B, Feature_Dim] - weighted sum of class prototypes
        features_l2 = self.claf_l2(l1_guide, features)
        logits_l2 = self.head_l2(features_l2)
        
        # L3
        # Guide by L2
        l2_guide = torch.matmul(torch.softmax(logits_l2, dim=1), self.head_l2.weight)
        features_l3 = self.claf_l3(l2_guide, features) # Or features_l2? Let's use original features to avoid drift, guided by L2.
        logits_l3 = self.head_l3(features_l3)
        
        return logits_l1, logits_l2, logits_l3, features
