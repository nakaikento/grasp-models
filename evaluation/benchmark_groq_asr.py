#!/usr/bin/env python3
"""
Groq Whisper API benchmark for ASR comparison.
Requires GROQ_API_KEY environment variable.
"""

import os
import time
import argparse
from pathlib import Path
from groq import Groq


def transcribe_audio(client: Groq, audio_path: Path, language: str = "ko") -> dict:
    """
    Transcribe audio using Groq Whisper API.
    
    Args:
        client: Groq client
        audio_path: Path to audio file (mp3, wav, etc.)
        language: Language code (ko, ja, en, etc.)
    
    Returns:
        dict with text, duration, and latency
    """
    start_time = time.time()
    
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path.name, audio_file.read()),
            model="whisper-large-v3",
            language=language,
            response_format="verbose_json"
        )
    
    latency = time.time() - start_time
    
    return {
        "text": transcription.text,
        "duration": getattr(transcription, 'duration', None),
        "latency_ms": latency * 1000,
        "language": language
    }


def main():
    parser = argparse.ArgumentParser(description="Groq Whisper ASR Benchmark")
    parser.add_argument("audio", type=Path, help="Audio file path")
    parser.add_argument("--language", "-l", default="ko", help="Language code")
    parser.add_argument("--repeat", "-r", type=int, default=1, help="Repeat count for averaging")
    args = parser.parse_args()
    
    # Check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("   Get your key at: https://console.groq.com/keys")
        return 1
    
    if not args.audio.exists():
        print(f"❌ Error: Audio file not found: {args.audio}")
        return 1
    
    client = Groq(api_key=api_key)
    
    print("="*60)
    print("Groq Whisper ASR Benchmark")
    print("="*60)
    print(f"Audio: {args.audio}")
    print(f"Language: {args.language}")
    print(f"Model: whisper-large-v3")
    print()
    
    latencies = []
    
    for i in range(args.repeat):
        result = transcribe_audio(client, args.audio, args.language)
        latencies.append(result["latency_ms"])
        
        if args.repeat == 1:
            print(f"📝 Transcription:")
            print(f"   {result['text']}")
            print()
            print(f"⏱️ Latency: {result['latency_ms']:.0f}ms")
            if result["duration"]:
                rtf = result["latency_ms"] / 1000 / result["duration"]
                print(f"📊 RTF: {rtf:.3f}x (realtime factor)")
        else:
            print(f"  Run {i+1}: {result['latency_ms']:.0f}ms")
    
    if args.repeat > 1:
        avg = sum(latencies) / len(latencies)
        print()
        print(f"📊 Average latency: {avg:.0f}ms (n={args.repeat})")
        print(f"📝 Last result: {result['text']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
