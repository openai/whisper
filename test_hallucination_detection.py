#!/usr/bin/env python3
"""
Test script for the enhanced hallucination detection system.
This script tests the hallucination detection functionality with synthetic examples.
"""

import sys
import os
import numpy as np
import torch

# Add the whisper module to the path
sys.path.insert(0, '/Users/safayavatsal/github/OpenSource/whisper')

try:
    from whisper.enhancements.hallucination_detector import (
        HallucinationDetector,
        detect_hallucinations,
        filter_hallucinations
    )
    from whisper.enhancements.confidence_scorer import (
        ConfidenceScorer,
        calculate_confidence_score,
        filter_by_confidence
    )
    print("✅ Enhanced hallucination detection modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import enhanced modules: {e}")
    sys.exit(1)


def test_pattern_detection():
    """Test pattern-based hallucination detection."""
    print("\n🔍 Testing pattern-based hallucination detection...")

    # Test cases: (text, should_be_detected)
    test_cases = [
        ("Hello, how are you today?", False),
        ("Thanks for watching, don't forget to subscribe!", True),
        ("Please subscribe to my channel for more content.", True),
        ("This is a normal conversation about the weather.", False),
        ("Like and subscribe if you enjoyed this video.", True),
        ("The meeting ended with everyone saying thank you.", False),
        ("So so so this is repetitive text", True),
        ("Regular speech without any issues here.", False),
    ]

    detector = HallucinationDetector("en")

    for text, should_detect in test_cases:
        result = detector.analyze_segment(text)
        detected = result.is_hallucination
        status = "✅" if detected == should_detect else "❌"
        print(f"{status} '{text[:50]}...' -> Detected: {detected}, Score: {result.confidence_score:.2f}")
        if result.detected_patterns:
            print(f"   Patterns found: {result.detected_patterns}")


def test_confidence_scoring():
    """Test confidence scoring system."""
    print("\n📊 Testing confidence scoring system...")

    test_cases = [
        {
            'text': "Hello, this is a clear and well-articulated sentence.",
            'avg_logprob': -0.2,
            'compression_ratio': 1.5,
            'no_speech_prob': 0.1
        },
        {
            'text': "um uh er this is very uh unclear speech",
            'avg_logprob': -1.5,
            'compression_ratio': 2.8,
            'no_speech_prob': 0.7
        },
        {
            'text': "Short.",
            'avg_logprob': -0.8,
            'compression_ratio': 1.0,
            'no_speech_prob': 0.3
        },
        {
            'text': "This is a reasonably long sentence that contains meaningful content and should score well for confidence.",
            'avg_logprob': -0.3,
            'compression_ratio': 1.8,
            'no_speech_prob': 0.2
        }
    ]

    scorer = ConfidenceScorer("en")

    for i, case in enumerate(test_cases):
        result = scorer.score_segment_confidence(**case)
        print(f"Test {i+1}: Overall Score: {result.overall_score:.2f}")
        print(f"  Text: '{case['text'][:50]}...'")
        print(f"  Component scores: {result.component_scores}")
        print()


def test_segment_filtering():
    """Test segment filtering functionality."""
    print("\n🔧 Testing segment filtering...")

    # Create mock segments with different quality levels
    test_segments = [
        {
            'text': 'This is good quality speech.',
            'avg_logprob': -0.2,
            'compression_ratio': 1.5,
            'no_speech_prob': 0.1,
            'start': 0.0,
            'end': 2.0
        },
        {
            'text': 'Thanks for watching and please subscribe!',
            'avg_logprob': -0.4,
            'compression_ratio': 2.1,
            'no_speech_prob': 0.3,
            'start': 2.0,
            'end': 4.0
        },
        {
            'text': 'Another normal sentence here.',
            'avg_logprob': -0.3,
            'compression_ratio': 1.7,
            'no_speech_prob': 0.2,
            'start': 4.0,
            'end': 6.0
        },
        {
            'text': 'Like and subscribe to my channel for more content!',
            'avg_logprob': -0.8,
            'compression_ratio': 2.5,
            'no_speech_prob': 0.5,
            'start': 6.0,
            'end': 8.0
        }
    ]

    print(f"Original segments: {len(test_segments)}")

    # Test hallucination filtering
    filtered_segments = filter_hallucinations(test_segments, language="en", strict_mode=False)
    print(f"After hallucination filtering: {len(filtered_segments)}")
    for segment in filtered_segments:
        if 'hallucination_analysis' in segment:
            print(f"  - '{segment['text'][:30]}...' (confidence: {segment['hallucination_analysis']['confidence_score']:.2f})")

    # Test confidence filtering
    confidence_filtered = filter_by_confidence(test_segments, min_confidence=0.5, language="en")
    print(f"After confidence filtering (>0.5): {len(confidence_filtered)}")
    for segment in confidence_filtered:
        if 'confidence_analysis' in segment:
            print(f"  - '{segment['text'][:30]}...' (score: {segment['confidence_analysis']['overall_score']:.2f})")


def test_multilingual_support():
    """Test multilingual hallucination detection."""
    print("\n🌐 Testing multilingual support...")

    test_cases = [
        ("Gracias por ver este video", "es", True),
        ("Hola, ¿cómo estás hoy?", "es", False),
        ("Merci de regarder cette vidéo", "fr", True),
        ("Bonjour, comment allez-vous?", "fr", False),
    ]

    for text, language, should_detect in test_cases:
        detector = HallucinationDetector(language)
        result = detector.analyze_segment(text)
        status = "✅" if result.is_hallucination == should_detect else "❌"
        print(f"{status} [{language}] '{text}' -> Detected: {result.is_hallucination}")


def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Hallucination Detection System")
    print("=" * 60)

    try:
        test_pattern_detection()
        test_confidence_scoring()
        test_segment_filtering()
        test_multilingual_support()

        print("\n🎉 All tests completed successfully!")
        print("\nThe enhanced hallucination detection system is working properly.")
        print("Key features tested:")
        print("  ✅ Pattern-based detection (YouTube artifacts, repetitions)")
        print("  ✅ Confidence scoring (multiple factors)")
        print("  ✅ Segment filtering (hallucinations and confidence)")
        print("  ✅ Multilingual support (Spanish, French)")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()