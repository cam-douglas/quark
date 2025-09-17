"""
AlphaGenome data module - genome data utilities.
"""

import numpy as np


class GenomeData:
    """Utilities for handling genome sequence data."""

    @staticmethod
    def encode_sequence(sequence: str) -> np.ndarray:
        """One-hot encode DNA sequence.
        
        Args:
            sequence: DNA sequence string (ACGT)
            
        Returns:
            One-hot encoded array of shape (4, len(sequence))
        """
        encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}
        encoded = np.zeros((4, len(sequence)))

        for i, base in enumerate(sequence.upper()):
            if base in encoding and encoding[base] >= 0:
                encoded[encoding[base], i] = 1
            else:
                # For N or unknown bases, use 0.25 for all
                encoded[:, i] = 0.25

        return encoded

    @staticmethod
    def decode_sequence(encoded: np.ndarray) -> str:
        """Decode one-hot encoded sequence back to string.
        
        Args:
            encoded: One-hot encoded array
            
        Returns:
            DNA sequence string
        """
        bases = ['A', 'C', 'G', 'T']
        sequence = []

        for i in range(encoded.shape[1]):
            col = encoded[:, i]
            if np.allclose(col, 0.25):
                sequence.append('N')
            else:
                idx = np.argmax(col)
                sequence.append(bases[idx])

        return ''.join(sequence)

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Get reverse complement of DNA sequence.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Reverse complement sequence
        """
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in sequence.upper()[::-1])


# Create a module-level instance for backward compatibility
genome = GenomeData()
