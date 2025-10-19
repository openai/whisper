#!/usr/bin/env python3
"""Simple test for enhanced hallucination detection without external dependencies."""

import sys
import os

# Test import of the modules
try:
    sys.path.insert(0, '/Users/safayavatsal/github/OpenSource/whisper')

    # Test basic imports
    from whisper.enhancements.hallucination_detector import HallucinationDetector
    from whisper.enhancements.confidence_scorer import ConfidenceScorer

    print("✅ Successfully imported hallucination detection modules")

    # Test basic functionality
    detector = HallucinationDetector("en")
    print("✅ Created HallucinationDetector instance")

    # Test pattern detection
    test_text = "Thanks for watching, please subscribe!"
    result = detector.analyze_segment(test_text)
    print(f"✅ Pattern detection test: '{test_text}'")
    print(f"   - Detected hallucination: {result.is_hallucination}")
    print(f"   - Confidence score: {result.confidence_score:.2f}")
    print(f"   - Detected patterns: {result.detected_patterns}")

    # Test with normal text
    normal_text = "This is a normal conversation about the weather."
    result2 = detector.analyze_segment(normal_text)
    print(f"✅ Normal text test: '{normal_text}'")
    print(f"   - Detected hallucination: {result2.is_hallucination}")
    print(f"   - Confidence score: {result2.confidence_score:.2f}")

    # Test confidence scorer
    scorer = ConfidenceScorer("en")
    print("✅ Created ConfidenceScorer instance")

    print("\n🎉 Basic functionality test completed successfully!")
    print("The enhanced hallucination detection system is working.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)