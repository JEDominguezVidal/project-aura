#!/usr/bin/env python3
"""
Test runner script that executes all individual tests in sequence.

Runs all test scripts and reports overall success/failure.
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_script):
    """Run a single test script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {test_script}...")
    print('='*60)

    try:
        result = subprocess.run([sys.executable, test_script],
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run {test_script}: {e}")
        return False


def main():
    """Run all tests and report results."""
    print("🚀 Starting comprehensive test suite...")
    print("This will test all components of the Whisper+MFA pipeline\n")

    # Get test scripts
    tests_dir = Path(__file__).parent
    test_scripts = [
        "test_audio_preprocess.py",
        "test_asr.py",
        "test_alignment.py",
        "test_segmenter.py"
    ]

    results = []
    passed = 0
    failed = 0

    # Run each test
    for test_script in test_scripts:
        test_path = tests_dir / test_script
        if test_path.exists():
            success = run_test(str(test_path))
            results.append((test_script, success))
            if success:
                passed += 1
            else:
                failed += 1
        else:
            print(f"❌ Test script not found: {test_script}")
            results.append((test_script, False))
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print('='*60)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")

    if failed == 0:
        print("\n🎉 All tests passed! The pipeline is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
