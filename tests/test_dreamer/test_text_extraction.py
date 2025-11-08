"""
Tests for Pokemon Red text extraction module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from pokemonred_puffer.data.text import (
    TextExtractor,
    decode_char,
    decode_text,
    CHAR_TABLE,
    MAX_TEXT_LENGTH,
)


class TestCharacterDecoding:
    """Test character decoding functionality."""
    
    def test_decode_char_valid(self):
        """Test decoding valid characters."""
        assert decode_char(0x80) == "A"
        assert decode_char(0x81) == "B"
        assert decode_char(0xF4) == "0"
        assert decode_char(0x7F) == " "
    
    def test_decode_char_special(self):
        """Test decoding special characters."""
        assert decode_char(0x00) == "<END>"
        assert decode_char(0x4E) == "\n"
        assert decode_char(0x50) == "@"
    
    def test_decode_char_unknown(self):
        """Test decoding unknown characters."""
        result = decode_char(0x70)
        assert "<" in result and ">" in result
    
    def test_decode_text_simple(self):
        """Test decoding simple text."""
        # "ABC"
        text_ids = np.array([0x80, 0x81, 0x82, 0x00], dtype=np.uint8)
        result = decode_text(text_ids)
        assert result == "ABC"
    
    def test_decode_text_with_terminator(self):
        """Test decoding text with early terminator."""
        text_ids = np.array([0x80, 0x81, 0x00, 0x82], dtype=np.uint8)
        result = decode_text(text_ids)
        assert result == "AB"
    
    def test_decode_text_empty(self):
        """Test decoding empty text."""
        text_ids = np.array([0x00], dtype=np.uint8)
        result = decode_text(text_ids)
        assert result == ""


class TestTextExtractor:
    """Test TextExtractor class."""
    
    def create_mock_pyboy(self):
        """Create a mock PyBoy instance for testing."""
        mock_pyboy = Mock()
        mock_pyboy.memory = MagicMock()
        return mock_pyboy
    
    def test_text_extractor_initialization(self):
        """Test TextExtractor initialization."""
        mock_pyboy = self.create_mock_pyboy()
        extractor = TextExtractor(mock_pyboy)
        
        assert extractor.pyboy is mock_pyboy
        assert extractor.last_text == ""
        assert len(extractor.last_text_ids) == MAX_TEXT_LENGTH
    
    def test_is_text_displaying_positive(self):
        """Test is_text_displaying when text is active."""
        mock_pyboy = self.create_mock_pyboy()
        extractor = TextExtractor(mock_pyboy)
        
        # Mock font loaded (FONT_LOADED address = 0xCFC4)
        def mock_get(addr):
            if addr == 0xCFC4:  # FONT_LOADED
                return 1
            elif addr == 0xD125:  # TEXT_BOX_ID
                return 0
            return 0
        
        mock_pyboy.memory.__getitem__ = mock_get
        
        assert extractor.is_text_displaying() is True
    
    def test_is_text_displaying_negative(self):
        """Test is_text_displaying when no text is active."""
        mock_pyboy = self.create_mock_pyboy()
        extractor = TextExtractor(mock_pyboy)
        
        # Mock no font loaded and no text box
        mock_pyboy.memory.__getitem__ = lambda addr: 0
        
        assert extractor.is_text_displaying() is False
    
    def test_read_text_buffer_basic(self):
        """Test reading text buffer."""
        mock_pyboy = self.create_mock_pyboy()
        extractor = TextExtractor(mock_pyboy)
        
        # Mock text buffer with "HELLO"
        # H=0x87, E=0x84, L=0x8B, L=0x8B, O=0x8E, END=0x00
        buffer_data = [0x87, 0x84, 0x8B, 0x8B, 0x8E, 0x00] + [0x00] * (MAX_TEXT_LENGTH - 6)
        
        def mock_get(arg):
            if isinstance(arg, slice):
                start = arg.start or 0
                stop = arg.stop or len(buffer_data)
                return buffer_data[0:stop-start]
            return 0
        
        mock_pyboy.memory.__getitem__ = mock_get
        
        text_ids, decoded = extractor.read_text_buffer()
        
        assert decoded == "HELLO"
        assert text_ids[0] == 0x87
        assert len(text_ids) == MAX_TEXT_LENGTH  # Padded array
    
    def test_get_current_text_caches_last_text(self):
        """Test that get_current_text caches the last seen text."""
        mock_pyboy = self.create_mock_pyboy()
        extractor = TextExtractor(mock_pyboy)
        
        # First call with text displaying
        buffer_data = [0x80, 0x81, 0x82, 0x00] + [0x00] * (MAX_TEXT_LENGTH - 4)
        
        def mock_getitem_displaying(arg):
            if isinstance(arg, slice):
                start = arg.start or 0
                stop = arg.stop or len(buffer_data)
                return buffer_data[0:stop-start]
            # For is_text_displaying checks
            if arg == 0xCFC4:  # FONT_LOADED
                return 1
            if arg == 0xD125:  # TEXT_BOX_ID
                return 1
            return 0
        
        mock_pyboy.memory.__getitem__ = mock_getitem_displaying
        
        text_ids, decoded, is_displaying = extractor.get_current_text()
        assert decoded == "ABC"
        assert is_displaying is True
        
        # Second call with no text displaying - should return cached text
        def mock_getitem_not_displaying(arg):
            if isinstance(arg, slice):
                return [0x00] * 200
            return 0  # No font loaded, no text box
        
        mock_pyboy.memory.__getitem__ = mock_getitem_not_displaying
        
        text_ids, decoded, is_displaying = extractor.get_current_text()
        assert decoded == "ABC"  # Cached from before
        assert is_displaying is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

