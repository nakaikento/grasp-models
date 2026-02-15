#!/usr/bin/env python3
"""Evaluate MarianMT with COMET."""

import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import MarianMTModel
import sentencepiece as spm
import torch
from comet import download_model, load_from_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--data-dir', default='evaluation/data/aihub')
    parser.add_argument('--limit', type=int, default=500)
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Load data
    ko_file = Path(args.data_dir) / 'ko_reference.txt'
    ja_file = Path(args.data_dir) / 'ja_source.txt'
    
    with open(ko_file, encoding='utf-8') as f:
        ko_lines = [l.strip() for l in f][:args.limit]
    with open(ja_file, encoding='utf-8') as f:
        ja_lines = [l.strip() for l in f][:args.limit]
    
    print(f'Loaded {len(ko_lines)} samples')
    
    # Load translation model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model = MarianMTModel.from_pretrained(model_dir).to(device)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_dir / 'spm.model'))
    model.eval()
    
    # Translate
    print('Translating...')
    translations = []
    
    with torch.no_grad():
        for text in tqdm(ko_lines):
            tokens = sp.EncodeAsIds(text)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            output_tokens = outputs[0].tolist()
            output_tokens = [t for t in output_tokens if t > 3]
            decoded = sp.DecodeIds(output_tokens)
            translations.append(decoded)
    
    # Prepare data for COMET
    data = [
        {'src': src, 'mt': mt, 'ref': ref}
        for src, mt, ref in zip(ko_lines, translations, ja_lines)
    ]
    
    # Load COMET model
    print('Loading COMET model...')
    model_path = download_model('Unbabel/wmt22-comet-da')
    comet_model = load_from_checkpoint(model_path)
    
    # Score
    print('Scoring with COMET...')
    output = comet_model.predict(data, batch_size=32, gpus=1)
    
    print(f'\n=== Results ===')
    print(f'COMET Score: {output.system_score:.4f}')
    print(f'Samples: {len(translations)}')

if __name__ == '__main__':
    main()
