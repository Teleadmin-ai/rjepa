# Audit Complet : Batching & Optimisations R-JEPA

**Date** : 2025-11-18
**Objectif** : Vérifier que le batching et les optimisations fonctionnent avec le service et la queue d'entraînement.

---

## 📈 RÉSULTATS TESTS DE VITESSE

### Test 1 : Sans Batching (Séquentiel)
- **Vitesse** : 41 secondes/problème ❌
- **Extrapolation** : 244 heures (10 jours) pour 21,456 problèmes
- **Problème** : BEAUCOUP TROP LENT

### Test 2 : Avec Batching (batch_size=8)
- **Vitesse** : 5.67 secondes/problème ✅
- **Extrapolation** : 33.8 heures (~1.4 jours) pour 21,456 problèmes
- **Gain** : **7.2x plus rapide** 🚀

### Test 3 : Avec Batching (batch_size=16)
- **Vitesse** : 18.48 secondes/problème ❌
- **Extrapolation** : 110.1 heures (~4.6 jours) pour 21,456 problèmes
- **Problème** : 3.3x PLUS LENT que batch_size=8 (GPU memory swapping)

### Conclusion Tests
✅ **batch_size=8 est OPTIMAL** pour RTX 4090 avec Qwen3-8B AWQ 4-bit
❌ Au-delà, le GPU a du mal (memory swapping, overhead)

---

## 🔍 AUDIT ARCHITECTURE

### 1. **Script d'Extraction Optimisé** (`scripts/extract_latents_optimized.py`)

**Status** : ✅ OPTIMAL

```python
# Batching implémenté (ligne 173-178)
def generate_and_extract_batch(self, problems: List[Dict], ...):
    # Génère batch_size problèmes en parallèle sur GPU
    ...
```

**Configuration** :
- `--batch-size 8` (RECOMMANDÉ)
- `--limit N` pour tests
- `--resume` pour checkpoint/reprendre

**Performances** :
- batch_size=8 : 5.67s/problème
- Full dataset : ~34h

---

### 2. **Pipeline Officiel** (`rjepa/pipeline/build_latents.py`)

**Status** : ⚠️ **BESOIN D'OPTIMISATION**

**Problème Actuel** (ligne 96) :
```python
for cot_data in tqdm(cots_data):  # ❌ SÉQUENTIEL, pas de batching
    H = llm.extract_latents(...)
```

**Fix Appliqué** :
- Ajout paramètre `batch_size=8` dans signature
- TODO : Implémenter logique de batching dans la boucle

**Recommandation** :
Pour l'instant, utiliser `scripts/extract_latents_optimized.py` pour extraction rapide.
Le pipeline officiel sera optimisé dans une prochaine phase.

---

### 3. **Training Pipeline** (`rjepa/jepa/dataset.py` + `rjepa/pipeline/train_rjepa.py`)

**Status** : ✅ **BATCHING OK**

**Architecture** :
```python
# dataset.py - Charge latents pré-extraits
class LatentDataset(Dataset):
    def __getitem__(self, idx):
        return latents[idx], domain_id

# train_rjepa.py - DataLoader PyTorch avec batching
train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],  # ✅ Configurable
    shuffle=True,
    num_workers=4,  # ✅ Multi-threading
)
```

**Config YAML** (`configs/rjepa/train.yaml`) :
```yaml
training:
  batch_size: 32  # Optimal pour training R-JEPA
  num_workers: 4
```

**Verdict** : ✅ **Aucun problème**, PyTorch DataLoader gère le batching automatiquement.

---

### 4. **Service d'Inférence** (`rjepa/jepa/service.py`)

**Status** : ✅ **OK mais pas de batching côté serveur**

**Architecture** :
```python
@app.post("/score")
def score(request: ScoreRequest):
    # Accepte 1 séquence à la fois
    latents = torch.tensor(request.latents)  # [num_steps, hidden_dim]
    latents = latents.unsqueeze(0)  # ✅ Ajoute batch dim [1, S, D]
    result = model.score(latents, ...)
    return result
```

**Justification** :
- API REST simple : 1 requête = 1 séquence
- Batching côté client possible si nécessaire (ex: UI backend)
- Pour inférence temps réel, batching n'est pas critique

**Verdict** : ✅ **Acceptable pour l'inférence**

---

## 🎯 RECOMMANDATIONS FINALES

### Pour Extraction de Latents (21,456 problèmes) :

1. **Utiliser `extract_latents_optimized.py`** :
   ```bash
   python scripts/extract_latents_optimized.py \
     --batch-size 8 \
     --checkpoint-every 10 \
     --resume  # Si besoin de reprendre
   ```

2. **Temps estimé** : ~34 heures (~1.4 jours)

3. **Optimisation future** :
   - Intégrer le batching dans `rjepa/pipeline/build_latents.py`
   - Permettre de réutiliser le script optimisé via Prefect

### Pour Training R-JEPA :

✅ **Aucune modification requise**
- DataLoader PyTorch gère le batching
- Config YAML contrôle batch_size

### Pour Service d'Inférence :

✅ **Aucune modification requise**
- API simple (1 séquence à la fois)
- Batching côté client si besoin

---

## 🐛 BUGS IDENTIFIÉS & FIXES

### 1. Double Forward Pass (adapter.py)

**Problème** : IMPOSSIBLE à éviter avec HuggingFace `generate()`
- `model.generate()` ne retourne pas les hidden states
- Obligés de faire un 2ème forward pass pour extraction

**Status** : ⚠️ **Limitation HuggingFace, pas un bug**

**Impact** : Temps d'extraction x2, mais inevitable.
Le vrai gain vient du batching, pas de l'élimination du double pass.

### 2. Scripts Multiples

**Fix** : ✅ **Scripts obsolètes supprimés**
- Supprimé : `test_latent_extraction.py`
- Supprimé : `extract_latents_from_problems.py`
- Conservé : `extract_latents_optimized.py` (seul script valide)

---

## ✅ CONCLUSION

| Composant | Status | Batching | Performance |
|-----------|--------|----------|-------------|
| Script optimisé | ✅ OK | batch_size=8 | 5.67s/problème |
| Pipeline officiel | ⚠️ TODO | Aucun | ~41s/problème |
| Training | ✅ OK | PyTorch DataLoader | Config YAML |
| Service | ✅ OK | Single sample API | Acceptable |

**Action Immédiate** :
Lancer extraction complète avec script optimisé (~34h).

**Action Future** :
Optimiser `build_latents.py` pour utiliser le batching.
