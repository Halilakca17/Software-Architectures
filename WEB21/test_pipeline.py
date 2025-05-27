#!/usr/bin/env python3
"""
Test script for the media analysis pipeline to ensure it works properly
before deploying the web application.
"""

import os
import sys
import argparse
from tumkod import MediaAnalysisPipeline

def test_pipeline(video_path=None):
    """
    Test the media analysis pipeline with a video file.
    If no video file is provided, it will ask for one.
    """
    print("=" * 80)
    print(" Media Analysis Pipeline Test Script ")
    print("=" * 80)
    
    if not video_path:
        video_path = input("\nEnter the path to a video file: ").strip()
        
    if not os.path.exists(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        return False
    
    # Create the pipeline
    print("\nInitializing pipeline...")
    pipeline = MediaAnalysisPipeline()
    
    # Run the pipeline
    print("\nRunning full analysis pipeline...")
    result = pipeline.run_full_pipeline(video_path)
    
    if result:
        print("\n✅ Pipeline test completed successfully!")
        print("\nSummary of results:")
        print(f"Video: {result['video_path']}")
        print(f"Audio: {result['audio_path']}")
        print(f"Transcript: {result['transcript_path']}")
        print(f"Summary: {result['summary_path']}")
        print(f"Sentiment Analysis: {result['sentiment_results_path']}")
        print(f"Sentiment Chart: {result['sentiment_chart_path']}")
        
        if result['audio_emotion']:
            print(f"Audio Emotion: {result['audio_emotion']['emotion']} "
                  f"({result['audio_emotion']['confidence']*100:.2f}%)")
            
        return True
    else:
        print("\n❌ Pipeline test failed!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test the media analysis pipeline.')
    parser.add_argument('-v', '--video', type=str, help='Path to the video file for testing')
    
    args = parser.parse_args()
    success = test_pipeline(args.video)
    
    if success:
        print("\nYou can now run the web application with: python app.py")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()