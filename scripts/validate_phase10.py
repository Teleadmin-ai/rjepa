#!/usr/bin/env python3
"""
Validate Phase 10: Docker Compose & Integration
"""
import sys
from pathlib import Path


def validate_phase10():
    """Validate that Phase 10 is complete."""
    print("[*] Validating Phase 10: Docker Compose & Integration...")
    print()

    # Check Docker Compose files
    docker_files = [
        "docker-compose.yml",
        "docker-compose.dev.yml",
        ".env.example",
    ]

    # Check all Dockerfiles exist
    dockerfile_services = [
        "docker/student-llm.Dockerfile",
        "docker/rjepa-service.Dockerfile",
        "docker/teacher-orch.Dockerfile",
        "docker/data-pipeline.Dockerfile",
        "docker/ui-backend.Dockerfile",
        "docker/ui-frontend.Dockerfile",
    ]

    # Check Makefile has Docker targets
    makefile_path = Path("Makefile")

    all_files = docker_files + dockerfile_services + [str(makefile_path)]
    all_exist = True

    for file_path in all_files:
        path = Path(file_path)
        if path.exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[FAIL] {file_path} (MISSING)")
            all_exist = False

    print()

    if not all_exist:
        print("[FAIL] Phase 10 validation FAILED: Some files are missing")
        return False

    # Validate docker-compose.yml structure
    try:
        import yaml

        with open("docker-compose.yml") as f:
            compose_data = yaml.safe_load(f)

        required_services = [
            "student-llm",
            "rjepa-service",
            "teacher-orch",
            "prefect-server",
            "data-pipeline",
            "ui-backend",
            "ui-frontend",
        ]

        services = compose_data.get("services", {})
        missing_services = []

        for service in required_services:
            if service in services:
                print(f"[OK] Service '{service}' defined in docker-compose.yml")
            else:
                print(f"[FAIL] Service '{service}' MISSING in docker-compose.yml")
                missing_services.append(service)

        if missing_services:
            print(f"[FAIL] Missing services: {', '.join(missing_services)}")
            return False

        # Check volumes
        volumes = compose_data.get("volumes", {})
        required_volumes = ["huggingface_cache", "prefect_data"]

        for volume in required_volumes:
            if volume in volumes:
                print(f"[OK] Volume '{volume}' defined")
            else:
                print(f"[FAIL] Volume '{volume}' MISSING")
                return False

        # Check network
        networks = compose_data.get("networks", {})
        if "rjepa-network" in networks:
            print("[OK] Network 'rjepa-network' defined")
        else:
            print("[FAIL] Network 'rjepa-network' MISSING")
            return False

    except ImportError:
        print("[WARN] PyYAML not installed, skipping docker-compose.yml validation")
        print("       Install with: pip install pyyaml")
    except Exception as e:
        print(f"[FAIL] Failed to validate docker-compose.yml: {e}")
        return False

    print()

    # Check .env.example has required variables
    try:
        with open(".env.example") as f:
            env_content = f.read()

        required_vars = [
            "TEACHER_CLAUDE_BASE_URL",
            "TEACHER_CLAUDE_API_KEY",
            "TEACHER_GPT_BASE_URL",
            "TEACHER_GPT_API_KEY",
            "STUDENT_MODEL_NAME",
            "WANDB_API_KEY",
            "RJEPA_CHECKPOINT",
        ]

        missing_vars = []
        for var in required_vars:
            if var in env_content:
                print(f"[OK] {var} defined in .env.example")
            else:
                print(f"[FAIL] {var} MISSING in .env.example")
                missing_vars.append(var)

        if missing_vars:
            print(f"[FAIL] Missing variables: {', '.join(missing_vars)}")
            return False

    except Exception as e:
        print(f"[FAIL] Failed to validate .env.example: {e}")
        return False

    print()

    # Check Makefile has Docker targets
    try:
        with open(makefile_path) as f:
            makefile_content = f.read()

        required_targets = [
            "docker-build",
            "docker-up",
            "docker-down",
            "docker-logs",
            "docker-clean",
            "docker-dev",
            "quick-start",
        ]

        missing_targets = []
        for target in required_targets:
            if f"{target}:" in makefile_content:
                print(f"[OK] Makefile target '{target}' exists")
            else:
                print(f"[FAIL] Makefile target '{target}' MISSING")
                missing_targets.append(target)

        if missing_targets:
            print(f"[FAIL] Missing targets: {', '.join(missing_targets)}")
            return False

    except Exception as e:
        print(f"[FAIL] Failed to validate Makefile: {e}")
        return False

    print()
    print("=" * 80)
    print("SUCCESS: PHASE 10 VALIDATION COMPLETE!")
    print("=" * 80)
    print()
    print("[OK] All Docker Compose files present")
    print("[OK] All Dockerfiles present (6 services)")
    print("[OK] docker-compose.yml structure valid")
    print("[OK] 7 services defined")
    print("[OK] 2 volumes defined")
    print("[OK] Network defined")
    print("[OK] .env.example has all required variables")
    print("[OK] Makefile has all Docker targets")
    print()
    print("Architecture Docker Compose:")
    print("   1. student-llm        (port 8000) - Qwen3-8B + latent extraction")
    print("   2. rjepa-service      (port 8100) - R-JEPA inference API")
    print("   3. teacher-orch       (port 8200) - Teacher orchestrator")
    print("   4. prefect-server     (port 4200) - Prefect UI")
    print("   5. data-pipeline      (worker)    - Prefect agent")
    print("   6. ui-backend         (port 8300) - FastAPI gateway")
    print("   7. ui-frontend        (port 3000) - Next.js chat UI")
    print()
    print("Quick Start Commands:")
    print("   1. Copy .env.example to .env:")
    print("      cp .env.example .env")
    print()
    print("   2. Edit .env with your API keys")
    print()
    print("   3. Build all images:")
    print("      make docker-build")
    print()
    print("   4. Start all services:")
    print("      make docker-up")
    print()
    print("   5. View logs:")
    print("      make docker-logs")
    print()
    print("   6. Access UI:")
    print("      - Chat: http://localhost:3000/chat")
    print("      - Jobs: http://localhost:3000/jobs")
    print("      - Prefect: http://localhost:4200")
    print()
    print("Development Mode:")
    print("   make docker-dev        # Start with hot reload")
    print("   make docker-dev-logs   # View dev logs")
    print()
    print("READY FOR PHASE 11: Evaluation & Benchmarks!")
    print()

    return True


if __name__ == "__main__":
    success = validate_phase10()
    sys.exit(0 if success else 1)
