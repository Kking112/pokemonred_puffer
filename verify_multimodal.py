
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.policies.multi_modal import MultiModalPolicy
import pufferlib.emulation
import gymnasium.spaces as spaces

def test_multimodal():
    print("Loading config...")
    # Load config manually or create a dummy one
    conf = OmegaConf.load("config.yaml")
    
    # Override for testing
    conf.env.headless = True
    conf.env.save_video = False
    
    # Create Env
    print("Creating Environment...")
    # Mock PyBoy to avoid needing the ROM
    import sys
    from unittest.mock import MagicMock
    
    # Mock PyBoy class
    mock_pyboy_class = MagicMock()
    # Mock the instance
    mock_pyboy_instance = MagicMock()
    mock_pyboy_class.return_value = mock_pyboy_instance
    
    # Mock screen and tilemap
    mock_screen = MagicMock()
    mock_pyboy_instance.screen = mock_screen
    mock_screen.ndarray = np.zeros((144, 160, 3), dtype=np.uint8)
    
    # Mock tilemap_window for text extraction
    # We need to support tilemap_window[1:19, 14:18] -> returning a numpy array of shape (18, 4) or similar?
    # Actually, slicing a MagicMock usually returns another MagicMock.
    # We need to make sure the result can be converted to np.array of correct shape.
    
    # Let's verify what environment.py does:
    # np.array(self.pyboy.tilemap_window[1:19, 14:18], dtype=np.uint8).flatten()
    
    # So tilemap_window[...] should return something convertable to numpy array.
    # We can use a real numpy/list for tilemap_window.
    fake_tilemap = np.zeros((32, 32), dtype=np.uint8)
    mock_pyboy_instance.tilemap_window = fake_tilemap
    
    # Mock memory for bag/party reading
    class MemoryMock:
        def __getitem__(self, key):
            if isinstance(key, slice):
                size = (key.stop or 0) - (key.start or 0)
                if size < 0: size = 0
                return [0] * size
            return 0
            
    mock_pyboy_instance.memory = MemoryMock()

    # Mock symbol_lookup
    mock_pyboy_instance.symbol_lookup.return_value = (0, 0)
    
    # Patch PyBoy in environment module
    import pokemonred_puffer.environment
    pokemonred_puffer.environment.PyBoy = mock_pyboy_class
    
    # Patch abstract method in RedGymEnv
    pokemonred_puffer.environment.RedGymEnv.get_game_state_reward = lambda self: {}
    
    # Set gb_path to dummy
    conf.env.gb_path = "dummy.gb"
    
    # Initiate
    if not os.path.exists(str(conf.env.gb_path)):
         # Create dummy file to pass existing checks if any (though we mocked PyBoy)
         with open("dummy.gb", "wb") as f:
             f.write(b"")

    
    # Instantiate
    env = RedGymEnv(conf.env)
    
    print("Resetting Environment...")
    obs, info = env.reset()
    
    # Check for text in obs
    assert "text" in obs, "Text missing from observation!"
    print(f"Text observation shape: {obs['text'].shape}")
    assert obs['text'].shape == (72,), f"Expected (72,), got {obs['text'].shape}"
    
    # Wrap with PufferEnv for Policy
    # PufferLib policies usually expect a PufferEnv or at least the binding
    # But RedGymEnv is a Gym env.
    # MultiModalPolicy takes 'env' which is used for spaces.
    
    # We can fake the binding or use pufferlib.emulation.GymnasiumPufferEnv
    # But that might require registering.
    # Let's try to just pass the env and see if it works, 
    # as MultiModalPolicy.__init__ uses env.single_action_space etc.
    # If passed a gym env directly, we might need to be careful 
    # about 'env.emulated' access in the policy.
    
    # In MultiModalPolicy:
    # self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
    # So it EXPECTS a PufferEnv.
    
    print("Wrapping with PufferEnv...")
    # Create a simple wrapper mock if needed, or use the real one
    # To use GymnasiumPufferEnv, we need a binding.
    
    # binding = pufferlib.emulation.Binding(
    #    env_cls=RedGymEnv,
    #    env_args=[conf.env],
    #    env_name="PokemonRed",
    # )
    
    # puffer_env = binding.py_env_cls.make() # This creates the env inside?
    # No, binding is just definition.
    
    # Let's make a PufferEnv
    # We need to register it or just use the class
    
    # Easiest way:
    import gymnasium as gym
    # Mocking the PufferEnv structure expected by the policy
    # Wrap with PufferEnv for Policy
    print("Wrapping with PufferEnv...")
    # Use real GymnasiumPufferEnv to ensure dtype is calculated correctly
    # We passed 'env' which is a Gym env.
    
    # PufferLib might require env_creator or just env
    # Try passing env instance directly if supported, or via binding
    
    # If GymnasiumPufferEnv takes env:
    try:
        mock_puffer_env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
    except TypeError:
        # Fallback if signature is different (e.g. requires env_cls)
        # But we want to use OUR mocked env instance
        # We can try to monkeypatch or just use the binding
        binding = pufferlib.emulation.Binding(
            env_cls=lambda: env, # Lambda returns our instance
            env_name="PokemonRed",
        )
        mock_puffer_env = binding.py_env_cls.make()
        
    
    # Instantiate Policy
    # We use real pufferlib methods now (assuming puf ferlib is installed)
    # But we need to make sure the env passed to Policy has 'emulated' attribute with valid spaces.
    
    
    # Instantiate Policy
    # We use real pufferlib methods now (assuming puf ferlib is installed)
    # But we need to make sure the env passed to Policy has 'emulated' attribute with valid spaces.
    
    
    print("Instantiating Policy...")
    policy = MultiModalPolicy(mock_puffer_env, hidden_size=512)
    
    print("Running Forward Pass...")
    
    # Get flat observation from PufferEnv
    # reset() returns (obs, info) where obs is flat
    flat_obs, _ = mock_puffer_env.reset()
    
    # Convert to torch tensor with batch dim
    # PufferEnv usage: obs is typically numpy, we need torch
    obs_tensor = torch.as_tensor(flat_obs).unsqueeze(0)
    
    # Run
    actions, value = policy(obs_tensor) # RecurrentNetwork forward
    
    print("Output shapes:")
    print(f"Actions: {actions.shape}")
    print(f"Value: {value.shape}")
    
    print("Test Passed!")

if __name__ == "__main__":
    test_multimodal()
