#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fresh speed test - current session"""
import json
import httpx
import time
from datetime import datetime

# Load 5 problems (quick fresh test)
with open('data/datasets/academic/math/gsm8k/train_problems.json') as f:
    problems = json.load(f)[10:15]  # Different from previous tests

print('=' * 70)
print(f'FRESH TEST: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('Testing 5 NEW problems (indices 10-15)')
print('=' * 70)
print()

client = httpx.Client(timeout=120)
base_url = 'http://localhost:8000'

# Check server
health = client.get(f'{base_url}/health').json()
print(f'Model: {health["model"]}')
print(f'Quantization: {health.get("quantization", "none")}')
print()

results = []
for i, prob in enumerate(problems, 1):
    print(f'[{i}/5] {prob["statement"][:50]}...')

    start = time.time()

    # Generate CoT
    gen_resp = client.post(f'{base_url}/generate', json={
        'prompt': prob['statement'],
        'max_new_tokens': 512,
        'temperature': 0.7,
        'num_samples': 1,
        'force_structure': True
    })
    sample = gen_resp.json()['samples'][0]

    # Extract latents
    lat_resp = client.post(f'{base_url}/extract_latents', json={
        'text': sample['full_text'],
        'force_structure': True
    })
    lat_data = lat_resp.json()

    elapsed = time.time() - start
    results.append({
        'steps': lat_data['num_steps'],
        'time': elapsed
    })

    print(f'  OK: {lat_data["num_steps"]} steps in {elapsed:.1f}s')

print()
print('=' * 70)
print('RESULTS')
print('=' * 70)
avg_time = sum(r['time'] for r in results) / len(results)
avg_steps = sum(r['steps'] for r in results) / len(results)
print(f'Average time: {avg_time:.1f}s per problem')
print(f'Average steps: {avg_steps:.1f} steps')
print()
print(f'ESTIMATION for 21,456 problems:')
print(f'  Time: {(21456 * avg_time) / 3600:.1f} hours')
print(f'  Days (24/7): {(21456 * avg_time) / 86400:.1f} days')
print('=' * 70)
