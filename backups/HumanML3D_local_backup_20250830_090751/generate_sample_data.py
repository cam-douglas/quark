import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import os
import codecs as cs
from os.path import join as pjoin

# Load the spacy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        word_list.append(word.lower())
        pos_list.append(token.pos_)
    return word_list, pos_list

def main():
    """
    This script generates a sample motion file in a format that
    our MotorCortex can consume. It simulates the output of the
    full HumanML3D data processing pipeline.
    """
    # Define a sample walking motion (placeholder)
    # This would normally come from the raw motion files.
    # [Frame, Joint1_x, Joint1_y, ..., JointN_z]
    sample_motion_data = np.random.rand(100, 66) # 100 frames, 22 joints * 3 axes
    
    # Create a dummy text description
    sample_caption = "a person is walking forward"
    word_list, pos_list = process_text(sample_caption)
    tokens = ' '.join([f'{w}/{p}' for w, p in zip(word_list, pos_list)])

    # Define output paths
    output_dir = "brain_architecture/data/HumanML3D/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    motion_file_path = pjoin(output_dir, "sample_motion.npy")
    text_file_path = pjoin(output_dir, "sample_motion.txt")

    # Save the motion data
    np.save(motion_file_path, sample_motion_data)

    # Save the text data in the format expected by the original code
    with cs.open(text_file_path, 'w') as f:
        f.write(f'{sample_caption}#{tokens}#0.0#10.0\n') # caption, tokens, start, end

    print(f"Generated sample motion file: {motion_file_path}")
    print(f"Generated sample text file: {text_file_path}")
    print("You can now use this sample data with the MotorCortex.")

if __name__ == "__main__":
    main()
