#!/usr/bin/env python3
"""
Ko-Ja ONNX蒸留モデルをAI Hubコーパスで評価。
"""

import argparse
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

def load_lines(path: Path) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

class KoJaTranslator:
    def __init__(self, model_dir: Path):
        self.encoder = ort.InferenceSession(str(model_dir / 'encoder_model_quantized.onnx'))
        self.decoder_init = ort.InferenceSession(str(model_dir / 'decoder_model_quantized.onnx'))
        self.decoder_past = ort.InferenceSession(str(model_dir / 'decoder_with_past_model_quantized.onnx'))
        
        # トークナイザー
        spm_path = model_dir.parent / 'ko-ja' / 'spm.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(spm_path))
        
        self.bos_id = 2
        self.eos_id = 3
    
    def translate(self, text: str, max_length: int = 128) -> str:
        # エンコード
        ids = self.sp.EncodeAsIds(text)[:max_length]
        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.array([[1]*len(ids)], dtype=np.int64)
        
        # エンコーダ
        encoder_out = self.encoder.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        encoder_hidden = encoder_out[0]
        
        # 最初のデコードステップ
        decoder_input = np.array([[self.bos_id]], dtype=np.int64)
        first_out = self.decoder_init.run(None, {
            'input_ids': decoder_input,
            'encoder_hidden_states': encoder_hidden,
            'encoder_attention_mask': attention_mask
        })
        
        logits = first_out[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        
        if next_token == self.eos_id:
            return ""
        
        # past_key_values抽出
        past_kv = {}
        output_names = [o.name for o in self.decoder_init.get_outputs()]
        for i, name in enumerate(output_names):
            if name.startswith('present.'):
                past_kv[name.replace('present.', 'past_key_values.')] = first_out[i]
        
        generated = [next_token]
        
        # 続きのステップ
        for _ in range(max_length - 1):
            decoder_input = np.array([[next_token]], dtype=np.int64)
            
            inputs = {
                'input_ids': decoder_input,
                'encoder_attention_mask': attention_mask,
            }
            inputs.update(past_kv)
            
            out = self.decoder_past.run(None, inputs)
            
            logits = out[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            
            if next_token == self.eos_id:
                break
            
            generated.append(next_token)
            
            # decoderキャッシュ更新
            output_names_past = [o.name for o in self.decoder_past.get_outputs()]
            for i, name in enumerate(output_names_past):
                if name.startswith('present.') and 'decoder' in name:
                    past_kv[name.replace('present.', 'past_key_values.')] = out[i]
        
        return self.sp.DecodeIds(generated)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/aihub/ko_reference.txt"))
    parser.add_argument("--reference", type=Path, default=Path("data/aihub/ja_source.txt"))
    parser.add_argument("--model-dir", type=Path, default=Path("../models/ko-ja-onnx-int8"))
    parser.add_argument("--limit", type=int, default=1000, help="評価サンプル数")
    parser.add_argument("--output", type=Path, default=Path("results/onnx_koja_eval.txt"))
    args = parser.parse_args()
    
    # 読み込み
    sources = load_lines(args.source)[:args.limit]
    references = load_lines(args.reference)[:args.limit]
    
    print(f"📥 Source (Korean): {len(sources)} lines")
    print(f"📥 Reference (Japanese): {len(references)} lines")
    print(f"🤖 Model: {args.model_dir}")
    
    # 翻訳器
    translator = KoJaTranslator(args.model_dir)
    
    # 翻訳
    print(f"\n🔄 Translating...")
    hypotheses = []
    start = time.time()
    
    for i, text in enumerate(sources):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  Progress: {i+1}/{len(sources)} ({(i+1)/elapsed:.1f} samples/s)")
        
        try:
            hyp = translator.translate(text)
            hypotheses.append(hyp)
        except Exception as e:
            print(f"  ⚠️ Error at {i}: {e}")
            hypotheses.append("")
    
    elapsed = time.time() - start
    print(f"   ⏱️ Total: {elapsed:.1f}s ({len(sources)/elapsed:.2f} samples/s)")
    
    # 評価
    print(f"\n📊 Evaluating...")
    
    from sacrebleu import corpus_chrf, corpus_bleu
    
    chrf = corpus_chrf(hypotheses, [references], word_order=2)
    bleu = corpus_bleu(hypotheses, [references], tokenize='char')
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS: Ko-Ja ONNX Model (Qwen Distilled)")
    print(f"{'='*60}")
    print(f"   Samples: {len(sources)}")
    print(f"   chrF++:  {chrf.score:.2f}")
    print(f"   BLEU:    {bleu.score:.2f}")
    print(f"{'='*60}")
    
    # サンプル表示
    print(f"\n📝 Sample translations:")
    for i in range(min(5, len(sources))):
        print(f"\n[{i}]")
        print(f"  KO:  {sources[i]}")
        print(f"  REF: {references[i]}")
        print(f"  HYP: {hypotheses[i]}")
    
    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for h in hypotheses:
            f.write(h + '\n')
    print(f"\n✅ Translations saved to {args.output}")

if __name__ == "__main__":
    main()
