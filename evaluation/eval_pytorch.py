#!/usr/bin/env python3
"""Evaluate MarianMT model using PyTorch (GPU)."""

import argparse
from pathlib import Path
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from transformers import MarianMTModel, AutoTokenizer
import sentencepiece as spm
import torch

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
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model = MarianMTModel.from_pretrained(model_dir).to(device)
    
    # Load SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_dir / 'spm.model'))
    
    model.eval()
    
    # Translate
    translations = []
    
    with torch.no_grad():
        for text in tqdm(ko_lines):
            # Tokenize
            tokens = sp.EncodeAsIds(text)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode
            output_tokens = outputs[0].tolist()
            # Remove special tokens (assume 0=pad, 1=unk, 2=bos, 3=eos)
            output_tokens = [t for t in output_tokens if t > 3]
            decoded = sp.DecodeIds(output_tokens)
            translations.append(decoded)
    
    # Evaluate
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    
    bleu_score = bleu.corpus_score(translations, [ja_lines])
    chrf_score = chrf.corpus_score(translations, [ja_lines])
    
    print(f'\nResults:')
    print(f'BLEU: {bleu_score.score:.2f}')
    print(f'chrF++: {chrf_score.score:.2f}')
    print(f'Samples: {len(translations)}')
    
    # Show examples
    print(f'\nExamples:')
    for i in range(min(3, len(translations))):
        print(f'  KO: {ko_lines[i]}')
        print(f'  JA (pred): {translations[i]}')
        print(f'  JA (ref): {ja_lines[i]}')
        print()

if __name__ == '__main__':
    main()
