# Grasp Evaluation Pipeline

Korean→Japanese ASR + Translation evaluation using TED Talks.

## 📋 Overview

This pipeline evaluates:
1. **ASR Accuracy** (Korean speech → Korean text)
2. **Translation Quality** (Korean text → Japanese text)
3. **End-to-End Performance** (Korean audio → Japanese text)

## 🔧 Setup

### Install Dependencies

```bash
cd evaluation
pip install -r requirements.txt
```

### Required Models

Ensure you have the following models in `../models/`:
- Korean ASR model (sherpa-onnx zipformer)
- Translation model: `ko-ja-onnx-int8/`

## 📥 Download TED Talk

Find a Korean TED talk on YouTube with Korean + Japanese subtitles:

```bash
# Example: Download a Korean TED talk
python download_ted.py "https://www.youtube.com/watch?v=EXAMPLE_VIDEO_ID"
```

This will download:
- `data/audio/VIDEO_ID.wav` - Audio file
- `data/subtitles/VIDEO_ID.ko.vtt` - Korean subtitles
- `data/subtitles/VIDEO_ID.ja.vtt` - Japanese subtitles

## 📊 Run Evaluation

### 1. ASR Evaluation (Korean speech → Korean text)

```bash
python evaluate_asr.py \
  data/audio/VIDEO_ID.wav \
  data/subtitles/VIDEO_ID.ko.vtt \
  --model-dir ../models/korean-asr \
  -o data/results/asr_results.json
```

**Metrics:**
- **WER** (Word Error Rate): Lower is better
- **CER** (Character Error Rate): Lower is better

### 2. Translation Evaluation (Korean text → Japanese text)

```bash
python evaluate_translation.py \
  data/subtitles/VIDEO_ID.ko.vtt \
  data/subtitles/VIDEO_ID.ja.vtt \
  --model-dir ../models/ko-ja-onnx-int8 \
  -o data/results/translation_results.json
```

**Metrics:**
- **BLEU**: Higher is better (0-100)
- **chrF**: Higher is better (0-100)

### 3. End-to-End Evaluation (Korean audio → Japanese text)

```bash
python evaluate_e2e.py \
  data/audio/VIDEO_ID.wav \
  data/subtitles/VIDEO_ID.ja.vtt \
  --reference-ko data/subtitles/VIDEO_ID.ko.vtt \
  --asr-model ../models/korean-asr \
  --translation-model ../models/ko-ja-onnx-int8 \
  -o data/results/e2e_results.json
```

**Metrics:**
- ASR quality (WER, CER)
- Translation quality (BLEU, chrF)

## 📈 Interpreting Results

### ASR (WER/CER)
- **< 10%**: Excellent
- **10-20%**: Good
- **20-30%**: Acceptable
- **> 30%**: Needs improvement

### Translation (BLEU)
- **> 40**: Excellent
- **30-40**: Good
- **20-30**: Acceptable
- **< 20**: Needs improvement

### Translation (chrF)
- **> 60**: Excellent
- **50-60**: Good
- **40-50**: Acceptable
- **< 40**: Needs improvement

## 🔍 Finding Korean TED Talks

1. Go to https://www.ted.com/
2. Filter by language: Korean
3. Check if Japanese subtitles are available
4. Copy the YouTube URL (if available)

Or search YouTube directly:
- "TED 한국어" (TED Korean)
- Look for talks with CC (closed captions) in both Korean and Japanese

## 📂 Directory Structure

```
evaluation/
├── download_ted.py           # Download TED talks
├── vtt_to_text.py           # Convert VTT to plain text
├── evaluate_asr.py          # ASR accuracy evaluation
├── evaluate_translation.py  # Translation quality evaluation
├── evaluate_e2e.py          # End-to-end evaluation
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── data/
    ├── audio/              # Downloaded audio files
    ├── subtitles/          # Downloaded subtitles (.vtt)
    └── results/            # Evaluation results (.json)
```

## 🚀 Next Steps

1. **Download multiple TED talks** for robust evaluation
2. **Analyze results** to identify weaknesses
3. **Improve models** based on findings:
   - If ASR WER is high → improve Korean ASR model
   - If BLEU is low → improve translation model
   - If E2E is worse than components → investigate integration issues

## 📝 Notes

- VTT subtitles may have timing information - use `vtt_to_text.py` to extract plain text
- Some TED talks may not have both Korean and Japanese subtitles
- Download multiple talks to get statistically significant results
