# -*- coding: utf-8 -*-
"""
TensorBoard Event File Parser
Reads a .tfevents file and prints all scalar summaries.
"""

import argparse
from tensorboard.backend.event_processing import event_accumulator

def parse_tfevents(file_path):
    """
    Parses a TensorBoard event file and prints scalar summaries.
    Args:
        file_path (str): The path to the .tfevents file.
    """
    print(f"🔎 Analyzing log file: {file_path}")

    # Create an EventAccumulator
    accumulator = event_accumulator.EventAccumulator(
        file_path,
        size_guidance={ # Configure to load all data
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
        }
    )
    
    # Load the events
    accumulator.Reload()
    
    # Get all scalar tags
    scalar_tags = accumulator.Tags()['scalars']
    
    if not scalar_tags:
        print("❌ No scalar summaries found in the log file.")
        return

    print("\n--- SCALAR METRICS FOUND ---")
    for tag in scalar_tags:
        print(f"\n📈 Metric: {tag}")
        events = accumulator.Scalars(tag)
        for event in events:
            print(f"   Step {event.step}: {event.value:.4f}")
    print("\n--------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a TensorBoard event file.')
    parser.add_argument('event_file', type=str, help='Path to the .tfevents file.')
    args = parser.parse_args()
    
    parse_tfevents(args.event_file)