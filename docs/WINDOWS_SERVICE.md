# R-JEPA Windows Service Setup

Guide pour faire tourner R-JEPA en tant que service Windows (démarrage automatique au boot).

## 🎯 Qu'est-ce qu'un service Windows ?

Un service Windows permet de faire tourner une application en arrière-plan de façon permanente, même sans être connecté. Les avantages :

- ✅ **Démarrage automatique** au boot de Windows
- ✅ **Redémarrage automatique** en cas de crash
- ✅ **Exécution en arrière-plan** (pas besoin de terminal ouvert)
- ✅ **Logging centralisé** dans des fichiers
- ✅ **Gestion via Services Windows** (services.msc)

## 📦 Services disponibles

### 1. **Student LLM Server** (`RJEPA-StudentLLM`)
Serveur FastAPI qui expose Qwen3-8B sur GPU pour :
- Génération de CoT structurés
- Extraction de latents

**Utilisation** : Permet à d'autres composants (UI, pipeline) d'appeler le LLM via HTTP

### 2. **Latent Extraction** (`RJEPA-LatentExtraction`)
Pipeline d'extraction continue qui surveille les nouveaux datasets et génère automatiquement les latents.

**Utilisation** : Traitement batch automatique des nouveaux problèmes

### 3. **Continuous Training** (`RJEPA-ContinuousTraining`)
Boucle d'entraînement continue qui re-entraîne R-JEPA chaque nuit avec les nouvelles données.

**Utilisation** : Apprentissage continu du système (amélioration progressive)

## 🚀 Installation

### Prérequis
- ✅ Windows 11 (ou 10)
- ✅ Droits administrateur
- ✅ Python venv configuré (`.venv`)
- ✅ GPU NVIDIA avec CUDA 12.1+

### Étape 1 : Installer NSSM

Le script PowerShell télécharge et installe automatiquement **NSSM (Non-Sucking Service Manager)**, un outil qui transforme n'importe quel exécutable en service Windows.

### Étape 2 : Installer un service

Ouvrir PowerShell **en tant qu'administrateur** :

```powershell
# Se placer dans le projet
cd C:\Users\teleadmin\world-txt-model

# Installer le service Student LLM
.\scripts\setup_windows_service.ps1 -Service student-llm -Install

# Ou installer tous les services d'un coup
.\scripts\setup_windows_service.ps1 -Service all -Install
```

### Étape 3 : Démarrer le service

```powershell
# Démarrer Student LLM
.\scripts\setup_windows_service.ps1 -Service student-llm -Start

# Vérifier le statut
.\scripts\setup_windows_service.ps1 -Service student-llm -Status
```

Le service va :
1. Charger le modèle Qwen3-8B sur cuda:0
2. Démarrer le serveur FastAPI sur port 8000
3. Logger dans `logs/student-llm/service.log`

### Étape 4 : Vérifier que ça marche

```bash
# Test HTTP
curl http://localhost:8000/health

# Devrait retourner:
# {"status":"ok","model":"Qwen/Qwen3-8B","hidden_size":4096,...}
```

## 📊 Gestion des services

### Voir le statut

```powershell
.\scripts\setup_windows_service.ps1 -Service student-llm -Status
```

### Arrêter un service

```powershell
.\scripts\setup_windows_service.ps1 -Service student-llm -Stop
```

### Redémarrer un service

```powershell
# Arrêter puis redémarrer
.\scripts\setup_windows_service.ps1 -Service student-llm -Stop
.\scripts\setup_windows_service.ps1 -Service student-llm -Start
```

### Désinstaller un service

```powershell
.\scripts\setup_windows_service.ps1 -Service student-llm -Uninstall
```

## 📝 Logs

Les logs de chaque service sont dans :

```
logs/
├─ student-llm/
│   └─ service.log          ← Logs du serveur LLM
├─ latent-extraction/
│   └─ service.log          ← Logs de l'extraction
└─ training/
    └─ service.log          ← Logs du training continu
```

Pour voir les logs en temps réel :

```powershell
# PowerShell
Get-Content -Path ".\logs\student-llm\service.log" -Wait

# Ou Git Bash
tail -f logs/student-llm/service.log
```

## 🛠️ Configuration avancée

### Changer le port du Student LLM

Par défaut, le serveur écoute sur port 8000. Pour changer :

1. Ouvrir `scripts\setup_windows_service.ps1`
2. Modifier la ligne `Args` pour `student-llm` :
   ```powershell
   Args = "--port 8080 --model Qwen/Qwen3-8B --device cuda:0"
   ```
3. Réinstaller le service :
   ```powershell
   .\scripts\setup_windows_service.ps1 -Service student-llm -Uninstall
   .\scripts\setup_windows_service.ps1 -Service student-llm -Install
   .\scripts\setup_windows_service.ps1 -Service student-llm -Start
   ```

### Utiliser un autre modèle

Pour utiliser Qwen3-32B au lieu de Qwen3-8B :

```powershell
Args = "--port 8000 --model Qwen/Qwen3-32B --device cuda:0 --quantization awq-4bit"
```

### Variables d'environnement

Le service définit automatiquement :
- `CUDA_VISIBLE_DEVICES=0` (utilise seulement GPU 0)

Pour ajouter d'autres variables :

```powershell
& $NSSMPath set RJEPA-StudentLLM AppEnvironmentExtra "CUDA_VISIBLE_DEVICES=0`nTRANSFORMERS_CACHE=C:\cache"
```

## 🔍 Dépannage

### Le service ne démarre pas

1. **Vérifier les logs** :
   ```powershell
   Get-Content .\logs\student-llm\service.log -Tail 50
   ```

2. **Vérifier que Python venv est correct** :
   ```powershell
   .\.venv\Scripts\python.exe --version
   # Devrait afficher: Python 3.11.9
   ```

3. **Tester manuellement** :
   ```bash
   cd /c/Users/teleadmin/world-txt-model
   source .venv/Scripts/activate
   python rjepa/llm/server.py
   ```

4. **Vérifier CUDA** :
   ```bash
   nvidia-smi
   # Devrait montrer le GPU
   ```

### Le service plante au bout de quelques heures

Problème courant : **Out of Memory (OOM)**

Solutions :
1. **Ajouter un swap GPU** (pas recommandé, lent)
2. **Réduire batch size** dans les configs
3. **Utiliser quantization** (AWQ 4-bit)
4. **Redémarrage automatique** :
   ```powershell
   & $NSSMPath set RJEPA-StudentLLM AppExit Default Restart
   & $NSSMPath set RJEPA-StudentLLM AppRestartDelay 5000
   ```

### Le modèle charge sur CPU au lieu de GPU

Vérifier dans les logs :
```
Device: cuda:0 ✅  (bon)
Device: cpu ❌     (mauvais)
```

Si c'est CPU :
1. **Vérifier CUDA** : `nvidia-smi`
2. **Vérifier PyTorch** :
   ```bash
   .venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
   # Devrait afficher: True
   ```
3. **Réinstaller PyTorch avec CUDA** si False :
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

## 🎛️ Services Windows (GUI)

Vous pouvez aussi gérer les services via l'interface Windows :

1. **Ouvrir Services** : `Win+R` → `services.msc`
2. **Chercher** `RJEPA-StudentLLM`
3. **Clic droit** → Démarrer / Arrêter / Redémarrer

Propriétés utiles :
- **Démarrage automatique** : Le service démarre au boot Windows
- **Redémarrage automatique** : Redémarre si crash
- **Connexion** : Compte utilisateur (par défaut : Système Local)

## 📚 Références

- **NSSM** : https://nssm.cc/
- **PowerShell** : Documentation Microsoft
- **FastAPI** : https://fastapi.tiangolo.com/

## ✅ Checklist de production

Avant de laisser tourner en production :

- [ ] Service installé et démarre au boot
- [ ] Logs vérifiés (pas d'erreurs)
- [ ] GPU utilisé (pas CPU)
- [ ] Endpoint `/health` répond
- [ ] Redémarrage automatique configuré
- [ ] Monitoring (optionnel : Prometheus + Grafana)
- [ ] Backup des checkpoints R-JEPA

## 🚀 Quick Start (TL;DR)

```powershell
# En administrateur
cd C:\Users\teleadmin\world-txt-model

# Installer + démarrer Student LLM
.\scripts\setup_windows_service.ps1 -Service student-llm -Install
.\scripts\setup_windows_service.ps1 -Service student-llm -Start

# Vérifier
curl http://localhost:8000/health

# Voir logs
tail -f logs/student-llm/service.log
```

Voilà! 🎉
