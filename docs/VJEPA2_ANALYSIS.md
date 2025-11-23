# Analyse V-JEPA 2 vs R-JEPA Actuel

## Résumé Exécutif

**Bonne nouvelle**: V-JEPA 2 inclut déjà `get_1d_sincos_pos_embed()` dans pos_embs.py!
Cela simplifie énormément la migration pour R-JEPA (séquences 1D).

## Comparaison Module par Module

### 1. modules.py

| Composant | V-JEPA 1 (actuel) | V-JEPA 2 (nouveau) | Action |
|-----------|-------------------|-------------------|--------|
| MLP | ✓ Identique | ✓ Identique | COPIER |
| SwiGLUFFN | ✗ Absent | ✓ Nouveau | AJOUTER (optionnel) |
| DropPath | ✗ Absent | ✓ Nouveau (stochastic depth) | AJOUTER |
| Attention | Simple (retourne x, attn) | Plus riche (is_causal, use_sdpa, attn_mask) | REMPLACER |
| RoPEAttention | ✗ Absent | ✓ Nouveau (rotary pos embed) | AJOUTER (optionnel) |
| Block | Simple | + drop_path, wide_silu, use_rope | REMPLACER |

**Changement majeur**: Notre `Attention.forward()` retourne `(x, attn)` mais V-JEPA 2 retourne juste `x`.
→ Il faut adapter les appelants ou modifier le module.

### 2. pos_embs.py

| Fonction | V-JEPA 1 | V-JEPA 2 | R-JEPA besoin |
|----------|----------|----------|---------------|
| get_1d_sincos_pos_embed | ✗ | ✓ PRÉSENT ! | ✅ PARFAIT |
| get_2d_sincos_pos_embed | ✓ | ✓ | Non utilisé |
| get_3d_sincos_pos_embed | ✓ | ✓ | Non utilisé |
| get_1d_sincos_pos_embed_from_grid | ✓ | ✓ | ✅ Base |

**Excellente nouvelle**: V-JEPA 2 a déjà la fonction 1D dont on a besoin!

### 3. vision_transformer.py → step_transformer.py

| Composant | V-JEPA 2 | R-JEPA Adaptation |
|-----------|----------|-------------------|
| PatchEmbed | Conv2D/Conv3D patches | SUPPRIMER (on reçoit déjà des latents) |
| input_proj | Absent | AJOUTER: Linear(llm_hidden, embed_dim) |
| pos_embed | 2D/3D sincos | → 1D sincos |
| blocks | Block avec RoPE optionnel | GARDER |
| norm | LayerNorm | GARDER |
| interpolate_pos_encoding | Pour images | SIMPLIFIER pour séquences |

### 4. predictor.py → step_predictor.py

| Composant | V-JEPA 2 | R-JEPA Status |
|-----------|----------|---------------|
| predictor_embed | Linear(embed_dim, pred_dim) | ✅ Compatible |
| mask_tokens | ParameterList (num_mask_tokens) | ✅ Compatible |
| predictor_pos_embed | 2D/3D → 1D | À ADAPTER |
| predictor_blocks | Block[] | ✅ Compatible |
| predictor_proj | Linear(pred_dim, embed_dim) | ✅ Compatible |
| forward logic | apply_masks, sorting | À SIMPLIFIER pour 1D |

## Fichiers à Créer/Modifier

### Phase 1: Base (pas de breaking changes)

```
rjepa/jepa/modules.py       # REMPLACER entièrement (MIT license)
rjepa/jepa/pos_embs.py      # COPIER + garder get_1d_sincos_pos_embed
```

### Phase 2: Encoder

```
rjepa/jepa/step_transformer.py  # ADAPTER de vision_transformer.py
  - Supprimer PatchEmbed
  - Ajouter input_proj
  - Utiliser get_1d_sincos_pos_embed
```

### Phase 3: Predictor

```
rjepa/jepa/step_predictor.py    # ADAPTER de predictor.py
  - Simplifier pour 1D
  - Garder mask_tokens, predictor_embed, predictor_proj
```

## Code V-JEPA 2 à Copier Directement

### pos_embs.py (ligne 60-72) - PARFAIT POUR R-JEPA

```python
def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
```

### modules.py - Attention simplifiée (sans RoPE, pour R-JEPA)

```python
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        # ... (lignes 390-434 de V-JEPA 2 modules.py)
```

## Estimation Temps Migration

| Phase | Durée | Complexité |
|-------|-------|------------|
| 1. modules.py + pos_embs.py | 1-2h | Basse (copier + adapter imports) |
| 2. step_transformer.py | 2-3h | Moyenne (supprimer PatchEmbed, adapter forward) |
| 3. step_predictor.py | 2-3h | Moyenne (simplifier pour 1D) |
| 4. Tests + validation | 2-3h | Basse (vérifier dimensions) |
| **Total** | **7-11h** | |

## Risques Identifiés

1. **Attention return signature**: V-JEPA 2 Attention ne retourne pas `attn`, juste `x`
   - **Solution**: Adapter les appelants ou ajouter paramètre optionnel

2. **Block forward signature**: V-JEPA 2 Block accepte plus de paramètres (mask, attn_mask, T, H_patches, W_patches)
   - **Solution**: Garder interface simple pour 1D (juste x)

3. **Checkpoints incompatibles**: Les noms de paramètres peuvent différer
   - **Solution**: Mapper les clés si nécessaire, ou réentraîner (datasets intacts)

## Recommandation

**Approche incrémentale**:
1. Commencer par pos_embs.py (aucun risque, copie directe)
2. Puis modules.py avec tests unitaires
3. Puis step_transformer.py
4. Puis step_predictor.py
5. Valider avec un forward pass complet
6. Tester chargement checkpoint (strict=False si besoin)

## Validation Finale

```python
# Test rapide après migration
import torch
from rjepa.jepa.model import ReasoningJEPA

# Créer modèle avec nouveau code V-JEPA 2
model = ReasoningJEPA(
    input_dim=4096,      # Qwen3-8B hidden size
    embed_dim=2048,      # R-JEPA internal dim
    depth_encoder=6,
    depth_predictor=4,
    num_heads=16,
)

# Test forward pass
H = torch.randn(2, 10, 4096)  # [batch, steps, llm_hidden]
output = model(H)
print(f"Loss: {output['loss'].item():.4f}")
print("✅ Migration successful!")
```
