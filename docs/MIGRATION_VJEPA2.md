# Plan de Migration V-JEPA 1 → V-JEPA 2

## Objectif
Migrer de V-JEPA 1 (CC BY-NC) vers V-JEPA 2 (MIT) pour permettre la commercialisation.

## Références
- **V-JEPA 1** : `legacy-vjepa/` (CC BY-NC, non-commercial)
- **V-JEPA 2** : `legacy-vjepa2/` (MIT, commercial OK)
- **Notre code** : `rjepa/jepa/` (à migrer)
- **Backup** : `backups/vjepa1_migration_20251123/`

## Mapping des Modules

| Notre Module (V-JEPA 1) | Source V-JEPA 2 | Action |
|-------------------------|-----------------|--------|
| `modules.py` (Block, Attention, MLP) | `src/models/utils/modules.py` | REMPLACER |
| `pos_embs.py` | `src/models/utils/pos_embs.py` | ADAPTER 1D |
| `step_transformer.py` | `src/models/vision_transformer.py` | ADAPTER 1D |
| `step_predictor.py` | `src/models/predictor.py` | ADAPTER 1D |
| `multiblock1d.py` | `src/masks/` | ADAPTER 1D |
| `vjepa_adapted/` | (supprimer) | SUPPRIMER |

## Modules à NE PAS Changer (Code Propre)
Ces modules sont notre code original, pas de V-JEPA :
- `model.py` : Assemblage R-JEPA (notre architecture)
- `trainer.py` : Training loop (notre code)
- `losses.py` : Loss functions (notre code)
- `maskers.py` : Masking strategies (notre adaptation)
- `dataset.py` : Data loading (notre code)
- `service.py` : API FastAPI (notre code)
- `client.py` : HTTP client (notre code)
- `encoder.py` : Wrapper (garder ou fusionner)
- `predictor.py` : Wrapper (garder ou fusionner)

## Ordre de Migration (Module par Module)

### Phase 1 : Modules de Base (sans casser les dimensions)
```
1. modules.py     : Block, Attention, MLP (fondation)
2. pos_embs.py    : Positional embeddings 1D
```

### Phase 2 : Encoder
```
3. step_transformer.py : Adapter VisionTransformer → StepTransformer
```

### Phase 3 : Predictor
```
4. step_predictor.py : Adapter Predictor pour 1D
```

### Phase 4 : Masking
```
5. multiblock1d.py : Adapter masking strategies
```

### Phase 5 : Cleanup
```
6. Supprimer vjepa_adapted/
7. Mettre à jour les imports
8. Tests de régression
```

## Checklist par Module

### 1. modules.py
- [ ] Copier `legacy-vjepa2/src/models/utils/modules.py`
- [ ] Garder : Block, Attention, MLP, DropPath
- [ ] Supprimer : fonctions spécifiques vidéo (rotate_queries_or_keys, etc.)
- [ ] Changer header : "MIT License" + "Adapted for R-JEPA"
- [ ] Tester : `python -c "from rjepa.jepa.modules import Block"`

### 2. pos_embs.py
- [ ] Copier `legacy-vjepa2/src/models/utils/pos_embs.py`
- [ ] Adapter : `get_2d_sincos_pos_embed` → `get_1d_sincos_pos_embed`
- [ ] Tester : `python -c "from rjepa.jepa.pos_embs import get_1d_sincos_pos_embed"`

### 3. step_transformer.py
- [ ] Baser sur `legacy-vjepa2/src/models/vision_transformer.py`
- [ ] Garder : notre interface (input_dim, max_seq_len, etc.)
- [ ] Remplacer : les blocks internes par V-JEPA 2
- [ ] Tester : forward pass avec dummy data

### 4. step_predictor.py
- [ ] Baser sur `legacy-vjepa2/src/models/predictor.py`
- [ ] Adapter pour séquences 1D
- [ ] Tester : forward pass avec masked data

### 5. multiblock1d.py
- [ ] Réviser `legacy-vjepa2/src/masks/`
- [ ] Garder notre logique 1D
- [ ] Mettre à jour header licence

## Validation

### Tests Unitaires
```bash
pytest tests/test_jepa_model.py -v
pytest tests/test_maskers.py -v
```

### Test de Compatibilité Checkpoint
```python
# Charger ancien checkpoint avec nouveau code
checkpoint = torch.load("checkpoint-epoch-39.pth")
model = ReasoningJEPA(...)  # Nouveau code
model.load_state_dict(checkpoint["model"], strict=False)
# Si ça charge → dimensions compatibles !
```

### Test de Régression
```bash
# Comparer loss sur même batch
python scripts/test_migration_regression.py
```

## Risques et Mitigations

| Risque | Mitigation |
|--------|------------|
| Checkpoints incompatibles | Backup fait, peut réentraîner |
| Perte de fonctionnalités | Migration module par module, tests à chaque étape |
| Bugs subtils | Tests de régression, comparaison loss |

## Timeline Estimée
- Phase 1 (modules.py, pos_embs.py) : 2-3h
- Phase 2 (step_transformer.py) : 3-4h
- Phase 3 (step_predictor.py) : 2-3h
- Phase 4 (multiblock1d.py) : 1-2h
- Phase 5 (cleanup, tests) : 2-3h
- **Total : 1-2 jours**

## Détails d'Implémentation

### Phase 1.1 : modules.py (Code Final)

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license.
# Adapted for R-JEPA (1D reasoning sequences)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    """Stochastic Depth per sample (V-JEPA 2)"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """Standard attention (V-JEPA 2 style, simplified for 1D)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0,
                 proj_drop=0.0, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """Transformer block (V-JEPA 2 with DropPath)"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

### Phase 1.2 : pos_embs.py (Code Final)

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license.

import numpy as np

def get_1d_sincos_pos_embed(embed_dim, seq_len, cls_token=False):
    """
    1D sinusoidal positional embeddings for reasoning sequences.

    Args:
        embed_dim: output dimension for each position
        seq_len: int of the sequence length (number of steps)
    Returns:
        pos_embed: [seq_len, embed_dim]
    """
    grid = np.arange(seq_len, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: array of positions [M,]
    returns: [M, embed_dim]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
```

### Phase 2 : step_transformer.py (Changements Clés)

**Principaux changements vs VisionTransformer:**
1. Supprimer `PatchEmbed` (on reçoit déjà des latents LLM)
2. Ajouter `input_proj = Linear(input_dim, embed_dim)`
3. Utiliser `get_1d_sincos_pos_embed` au lieu de 2D/3D
4. Simplifier `forward()` pour séquences 1D

### Phase 3 : step_predictor.py (Changements Clés)

**Principaux changements vs VisionTransformerPredictor:**
1. Utiliser `get_1d_sincos_pos_embed`
2. Simplifier la gestion des masks pour 1D
3. Garder `predictor_embed`, `mask_tokens`, `predictor_proj`

## Scripts de Validation

### scripts/test_migration_phase1.py
```python
#!/usr/bin/env python
"""Test Phase 1 migration (modules.py, pos_embs.py)"""
import torch
from rjepa.jepa.modules import Block, Attention, MLP, DropPath
from rjepa.jepa.pos_embs import get_1d_sincos_pos_embed

# Test MLP
mlp = MLP(256, 1024, 256)
x = torch.randn(2, 10, 256)
y = mlp(x)
assert y.shape == x.shape, f"MLP shape mismatch: {y.shape}"
print("✅ MLP OK")

# Test Attention
attn = Attention(256, num_heads=8)
y = attn(x)
assert y.shape == x.shape, f"Attention shape mismatch: {y.shape}"
print("✅ Attention OK")

# Test Block
block = Block(256, num_heads=8, drop_path=0.1)
y = block(x)
assert y.shape == x.shape, f"Block shape mismatch: {y.shape}"
print("✅ Block OK")

# Test pos_embs
pos = get_1d_sincos_pos_embed(256, 10)
assert pos.shape == (10, 256), f"pos_embed shape mismatch: {pos.shape}"
print("✅ pos_embs OK")

print("\n🎉 Phase 1 migration validated!")
```

## Post-Migration
1. Changer LICENSE du projet en Apache 2.0
2. Mettre à jour README avec nouvelle licence
3. Relancer training si checkpoints incompatibles
4. Mettre à jour site web avec sections Research + Service
