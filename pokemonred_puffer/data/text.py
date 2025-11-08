"""
Pokemon Red text extraction and decoding module.

This module handles extraction of in-game text from Pokemon Red game memory
and decoding the custom character encoding used by the game.

Based on Pokemon Red/Blue disassembly character table:
https://github.com/pret/pokered
"""

from typing import Optional, Tuple
import numpy as np
from pyboy import PyBoy

# Pokemon Red character encoding table
# Maps byte values (0x00-0xFF) to characters
# Special characters: 0x00 = terminator, 0x4E = line break, 0x4F = paragraph, 0x50 = @, etc.
CHAR_TABLE = {
    0x00: "<END>",  # String terminator
    0x4E: "\n",  # New line
    0x4F: "<PAGE>",  # New page (wait for button press)
    0x50: "@",  # Player name
    0x51: "<POKé>",  # POKé
    0x52: "……",  # Ellipsis for menus
    0x53: "<enemy>",  # Enemy trainer name
    0x54: "<RIVAL>",  # Rival name
    0x55: "<PKMN>",  # POKéMON
    0x56: "<_PKMN>",  # POKéMON without accent
    0x57: "<PC>",  # PC
    0x58: "<TM>",  # TM
    0x59: "<TRAINER>",  # TRAINER
    0x5A: "<ROCKET>",  # ROCKET
    0x5B: "<TARGET>",  # Move target's name
    0x5C: "<USER>",  # Move user's name
    0x5D: "<CONT>",  # Continue text
    0x5E: "…",  # Single dot (wait)
    0x5F: "¿",  # Opening quote
    0x60: "¡",  # Opening exclamation
    0x61: "<PARA>",  # Paragraph (equivalent to 0x4F)
    0x62: "<DONE>",  # End text box
    0x63: "<PROMPT>",  # Prompt for button press
    0x64: "<TARGET>",  # Same as 0x5B
    0x65: "<USER>",  # Same as 0x5C
    0x66: "<dex>",  # Pokedex
    # 0x70-0x79: Unused in most contexts
    0x7F: " ",  # Space
    0x80: "A",
    0x81: "B",
    0x82: "C",
    0x83: "D",
    0x84: "E",
    0x85: "F",
    0x86: "G",
    0x87: "H",
    0x88: "I",
    0x89: "J",
    0x8A: "K",
    0x8B: "L",
    0x8C: "M",
    0x8D: "N",
    0x8E: "O",
    0x8F: "P",
    0x90: "Q",
    0x91: "R",
    0x92: "S",
    0x93: "T",
    0x94: "U",
    0x95: "V",
    0x96: "W",
    0x97: "X",
    0x98: "Y",
    0x99: "Z",
    0x9A: "(",
    0x9B: ")",
    0x9C: ":",
    0x9D: ";",
    0x9E: "[",
    0x9F: "]",
    0xA0: "a",
    0xA1: "b",
    0xA2: "c",
    0xA3: "d",
    0xA4: "e",
    0xA5: "f",
    0xA6: "g",
    0xA7: "h",
    0xA8: "i",
    0xA9: "j",
    0xAA: "k",
    0xAB: "l",
    0xAC: "m",
    0xAD: "n",
    0xAE: "o",
    0xAF: "p",
    0xB0: "q",
    0xB1: "r",
    0xB2: "s",
    0xB3: "t",
    0xB4: "u",
    0xB5: "v",
    0xB6: "w",
    0xB7: "x",
    0xB8: "y",
    0xB9: "z",
    0xE0: "'",
    0xE1: "<PK>",  # PK (for Poké Ball)
    0xE2: "<MN>",  # MN (for POKéMON)
    0xE3: "-",
    0xE4: "<r>",  # r (for moves like Fire Punch)
    0xE5: "<l>",  # l (for male symbol)
    0xE6: "?",
    0xE7: "!",
    0xE8: ".",
    0xE9: "&",  # Appears as a special character
    0xEA: "é",  # é for POKé
    0xEB: "→",  # Right arrow
    0xEC: "↓",  # Down arrow
    0xED: "♂",  # Male symbol
    0xEE: "$",  # Money symbol
    0xEF: "×",  # Multiplication symbol
    0xF0: ".",  # Period variant
    0xF1: "/",
    0xF2: ",",
    0xF3: "♀",  # Female symbol
    0xF4: "0",
    0xF5: "1",
    0xF6: "2",
    0xF7: "3",
    0xF8: "4",
    0xF9: "5",
    0xFA: "6",
    0xFB: "7",
    0xFC: "8",
    0xFD: "9",
}

# Fill in any missing values with a placeholder
for i in range(256):
    if i not in CHAR_TABLE:
        CHAR_TABLE[i] = f"<0x{i:02X}>"


# Text memory addresses for Pokemon Red
# Based on https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
TEXT_BOX_ID = 0xD125  # Current text box ID (0 = none)
TEXT_BUFFER_1 = 0xCD3D  # Primary text buffer (up to ~200 bytes)
TEXT_BUFFER_2 = 0xCE9D  # Secondary text buffer
TEXT_DELAY = 0xCFC6  # Text delay/speed
FONT_LOADED = 0xCFC4  # Whether font is loaded for text display
TEXT_FLAGS = 0xCC3C  # Text-related flags
MAX_TEXT_LENGTH = 200  # Maximum text buffer length to read


class TextExtractor:
    """Extracts and decodes text from Pokemon Red game memory."""
    
    def __init__(self, pyboy: PyBoy):
        """
        Initialize text extractor.
        
        Args:
            pyboy: PyBoy emulator instance
        """
        self.pyboy = pyboy
        self.last_text = ""
        self.last_text_ids = np.zeros(MAX_TEXT_LENGTH, dtype=np.uint8)
        
    def read_text_buffer(self, buffer_addr: int = TEXT_BUFFER_1, 
                         max_len: int = MAX_TEXT_LENGTH) -> Tuple[np.ndarray, str]:
        """
        Read and decode text from a memory buffer.
        
        Args:
            buffer_addr: Starting address of text buffer
            max_len: Maximum number of bytes to read
            
        Returns:
            Tuple of (character IDs as numpy array, decoded string)
        """
        # Read raw bytes from memory
        text_bytes = self.pyboy.memory[buffer_addr:buffer_addr + max_len]
        
        # Find terminator (0x00 or 0x50 or other special codes)
        text_ids = []
        decoded_chars = []
        
        for byte_val in text_bytes:
            # Stop at string terminator
            if byte_val == 0x00:
                break
            
            text_ids.append(byte_val)
            
            # Decode character
            if byte_val in CHAR_TABLE:
                decoded_chars.append(CHAR_TABLE[byte_val])
            else:
                decoded_chars.append(f"<{byte_val:02X}>")
                
            # Stop if we've read enough
            if len(text_ids) >= max_len:
                break
        
        # Convert to numpy array with padding
        text_ids_array = np.zeros(max_len, dtype=np.uint8)
        if text_ids:
            text_ids_array[:len(text_ids)] = text_ids
        
        # Join decoded characters
        decoded_str = "".join(decoded_chars)
        
        return text_ids_array, decoded_str
    
    def is_text_displaying(self) -> bool:
        """
        Check if text is currently being displayed.
        
        Returns:
            True if text box is active, False otherwise
        """
        try:
            # Check if font is loaded (indicates text is being rendered)
            font_loaded = self.pyboy.memory[FONT_LOADED]
            
            # Check text box ID (non-zero means text box is active)
            text_box_id = self.pyboy.memory[TEXT_BOX_ID]
            
            return font_loaded != 0 or text_box_id != 0
        except:
            return False
    
    def get_current_text(self) -> Tuple[np.ndarray, str, bool]:
        """
        Get currently displayed text from the game.
        
        Returns:
            Tuple of:
            - Character IDs as numpy array (padded to MAX_TEXT_LENGTH)
            - Decoded text string
            - Boolean indicating if text is currently displaying
        """
        is_displaying = self.is_text_displaying()
        
        if is_displaying:
            # Try to read from primary buffer
            text_ids, decoded = self.read_text_buffer(TEXT_BUFFER_1)
            
            # If primary buffer is empty, try secondary
            if decoded.strip() == "" or len(decoded) < 2:
                text_ids, decoded = self.read_text_buffer(TEXT_BUFFER_2)
            
            # Update last seen text
            if decoded.strip():
                self.last_text = decoded
                self.last_text_ids = text_ids
        else:
            # Return last seen text if no text is currently displaying
            text_ids = self.last_text_ids
            decoded = self.last_text
        
        return text_ids, decoded, is_displaying
    
    def get_text_state(self) -> dict:
        """
        Get comprehensive text state information.
        
        Returns:
            Dictionary with:
            - text_ids: Character IDs array
            - text: Decoded string
            - is_displaying: Boolean
            - text_box_id: Current text box ID
            - font_loaded: Boolean
        """
        text_ids, decoded, is_displaying = self.get_current_text()
        
        return {
            "text_ids": text_ids,
            "text": decoded,
            "is_displaying": is_displaying,
            "text_box_id": int(self.pyboy.memory[TEXT_BOX_ID]),
            "font_loaded": bool(self.pyboy.memory[FONT_LOADED]),
        }


def decode_char(char_id: int) -> str:
    """
    Decode a single Pokemon Red character ID to a string.
    
    Args:
        char_id: Character ID (0-255)
        
    Returns:
        Decoded character string
    """
    return CHAR_TABLE.get(char_id, f"<{char_id:02X}>")


def decode_text(char_ids: np.ndarray) -> str:
    """
    Decode an array of Pokemon Red character IDs to a string.
    
    Args:
        char_ids: Array of character IDs
        
    Returns:
        Decoded text string
    """
    chars = []
    for cid in char_ids:
        if cid == 0:  # Terminator
            break
        chars.append(decode_char(int(cid)))
    
    return "".join(chars)

