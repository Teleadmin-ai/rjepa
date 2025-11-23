# Legal Notice

## R-JEPA Project

R-JEPA (Reasoning Joint Embedding Predictive Architecture) is an independent implementation inspired by V-JEPA (Meta AI, MIT License).

### Original Contributions

This project contains significant original contributions developed by Teleadmin, including but not limited to:

1. **1D Adaptation for Text Reasoning**
   - Novel adaptation of the JEPA architecture from 2D/3D video to 1D text sequences
   - Step-level masking strategies for Chain-of-Thought reasoning
   - Latent extraction from LLM hidden states (layer -2)

2. **Dual Learning Architecture**
   - Unidirectional guidance system where LLM adapts to R-JEPA (never reverse)
   - Logit guidance mechanism for real-time token generation steering
   - Prevents world model contamination from LLM biases

3. **ValidationGate**
   - Cognitive firewall for automatic CoT validation
   - Mathematical verification (SymPy-based)
   - Code execution sandboxing
   - Only verified reasoning enters memory storage

4. **Associative Memory System**
   - Privacy-preserving memory architecture
   - Latents serve as pointers to local text databases
   - Text data never transmitted to cloud services
   - Cross-session persistent memory without data exposure

5. **Multi-LLM Rejouability**
   - Projection adapters (W_in/W_out) for rapid LLM migration
   - Support for 18+ open-source LLMs (Qwen, Llama, Mistral, DeepSeek, Phi, Yi, etc.)
   - Fast calibration (2-4 hours vs days for full retraining)

6. **Continuous Learning Pipeline**
   - User feedback integration with PII filtering
   - Nightly retraining with A/B testing
   - Automatic model deployment with rollback capability

### Attribution

The foundational JEPA concept is based on research by Yann LeCun and Meta AI:
- V-JEPA: "Video Joint Embedding Predictive Architecture" (Meta AI, 2024)
- Source code: https://github.com/facebookresearch/vjepa (MIT License)

### Copyright

© 2024-2025 Teleadmin / Romain Provençal
All original contributions are released under the MIT License.

### Contact

- Website: https://teleadmin-ai.github.io/rjepa/
- Email: contact@teleadmin.net
- GitHub: https://github.com/Teleadmin-ai/rjepa

### Disclaimer

This software is provided "as is", without warranty of any kind. The authors are not liable for any claims, damages, or other liability arising from the use of this software.

R-JEPA is an independent project and is not affiliated with, endorsed by, or sponsored by Meta Platforms, Inc. or any of its subsidiaries.
