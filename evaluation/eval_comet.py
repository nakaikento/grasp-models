#!/usr/bin/env python3
"""Evaluate with COMET (neural metric)."""

import argparse
from pathlib import Path
from comet import download_model, load_from_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--translations', required=True, help='File with translations')
    parser.add_argument('--sources', required=True, help='File with source sentences')  
    parser.add_argument('--references', required=True, help='File with references')
    parser.add_argument('--limit', type=int, default=500)
    args = parser.parse_args()
    
    # Load data
    with open(args.sources, encoding='utf-8') as f:
        sources = [l.strip() for l in f][:args.limit]
    with open(args.references, encoding='utf-8') as f:
        references = [l.strip() for l in f][:args.limit]
    with open(args.translations, encoding='utf-8') as f:
        translations = [l.strip() for l in f][:args.limit]
    
    print(f'Loaded {len(sources)} samples')
    
    # Prepare data for COMET
    data = [
        {'src': src, 'mt': mt, 'ref': ref}
        for src, mt, ref in zip(sources, translations, references)
    ]
    
    # Load model (downloads on first run)
    print('Loading COMET model...')
    model_path = download_model('Unbabel/wmt22-comet-da')
    model = load_from_checkpoint(model_path)
    
    # Score
    print('Scoring...')
    output = model.predict(data, batch_size=32, gpus=1)
    
    print(f'\nCOMET Score: {output.system_score:.4f}')

if __name__ == '__main__':
    main()
