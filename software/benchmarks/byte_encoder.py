#!/usr/bin/env python3
"""
Byte-level text encoder for processing text as sequences of bytes.
Used for byte-level text classification tasks like IMDb in Long Range Arena.
"""

import numpy as np


class ByteEncoder:
    """
    Encodes text strings as sequences of bytes (0-255).
    
    This is simpler than traditional tokenization and creates longer sequences,
    which is useful for testing long-range sequence models.
    """
    
    def __init__(self, max_length: int = 4096, padding_value: int = 0):
        """
        Initialize the byte encoder.
        
        Args:
            max_length: Maximum sequence length (will truncate or pad to this)
            padding_value: Value to use for padding (default 0)
        """
        self.max_length = max_length
        self.padding_value = padding_value
        self.vocab_size = 256  # Bytes are 0-255
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a text string as a sequence of bytes.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of shape (max_length,) containing byte values
        """
        # Convert text to bytes
        byte_sequence = text.encode('utf-8', errors='ignore')
        
        # Convert to numpy array of integers
        byte_array = np.frombuffer(byte_sequence, dtype=np.uint8).astype(np.int32)
        
        # Truncate if too long
        if len(byte_array) > self.max_length:
            byte_array = byte_array[:self.max_length]
        
        # Pad if too short
        if len(byte_array) < self.max_length:
            padding = np.full(self.max_length - len(byte_array), self.padding_value, dtype=np.int32)
            byte_array = np.concatenate([byte_array, padding])
        
        return byte_array
    
    def encode_batch(self, texts: list) -> np.ndarray:
        """
        Encode a batch of text strings.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (batch_size, max_length)
        """
        return np.stack([self.encode(text) for text in texts])
    
    def decode(self, byte_sequence: np.ndarray) -> str:
        """
        Decode a sequence of bytes back to text.
        
        Args:
            byte_sequence: Array of byte values
            
        Returns:
            Decoded text string
        """
        # Remove padding
        byte_sequence = byte_sequence[byte_sequence != self.padding_value]
        
        # Convert to bytes and decode
        byte_array = byte_sequence.astype(np.uint8).tobytes()
        return byte_array.decode('utf-8', errors='ignore')
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size (256 for bytes)."""
        return self.vocab_size
    
    def get_padding_value(self) -> int:
        """Return the padding value used."""
        return self.padding_value


def create_byte_encoder(max_length: int = 4096) -> ByteEncoder:
    """
    Factory function to create a byte encoder.
    
    Args:
        max_length: Maximum sequence length
        
    Returns:
        ByteEncoder instance
    """
    return ByteEncoder(max_length=max_length)