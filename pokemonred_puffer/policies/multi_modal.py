import pufferlib.emulation
import pufferlib.models
import pufferlib.pytorch
import torch
from torch import nn

from pokemonred_puffer.data.events import EVENTS_IDXS
from pokemonred_puffer.data.items import Items
from pokemonred_puffer.environment import PIXEL_VALUES


class MultiModalRNN(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


# Because torch.nn.functional.one_hot cannot be traced by torch as of 2.2.0
def one_hot(tensor, num_classes):
    index = torch.arange(0, num_classes, device=tensor.device)
    return (tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])).to(
        torch.int64
    )


class MultiModalPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.emulation.GymnasiumPufferEnv,
        hidden_size: int = 512,
        channels_last: bool = True,
        downsample: int = 1,
    ):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        
        # --- Vision Encoder (CNN) ---
        self.screen_network = nn.Sequential(
            nn.LazyConv2d(32, 8, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, 4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # --- Text Encoder (GRU) ---
        self.text_embedding = nn.Embedding(num_embeddings=256, embedding_dim=64)
        self.text_gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        # Project text embedding to match vision embedding size for easier concatenation/fusion logic if needed
        # Or we can keep it small. The user asked for "Gated Fusion".
        # Let's project it to 512 to match typical hidden sizes if we want to add them, 
        # but the fusion request says: H_fused = z * H_v + (1-z) * H_t.
        # This implies H_v and H_t must have the same dimension.
        # Vision output (H_v) is flattened conv output. Let's check its size. 
        # Screen is 72x80 (downsampled). Conv layers...
        # 72x80 -> (72-8)/2 + 1 = 33x37 -> ... -> It's complex to calc exactly without running.
        # We should probably project both to `hidden_size` (512).
        
        self.vision_proj = nn.LazyLinear(hidden_size)
        self.text_proj = nn.Linear(64, hidden_size)

        # --- Fusion Gate ---
        # z = Sigmoid(Linear(Concat(Hv, Ht)))
        # z = Sigmoid(Linear(Concat(Hv, Ht)))
        self.gate_linear = nn.Linear(hidden_size * 2, 1)

        self.encode_linear = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
        )

        # --- Policy Heads ---
        self.actor = nn.LazyLinear(self.num_actions)
        self.value_fn = nn.LazyLinear(1)

        # --- Other Embeddings (Legacy from MultiConvolutional) ---
        self.two_bit = env.unwrapped.env.two_bit
        self.skip_safari_zone = env.unwrapped.env.skip_safari_zone
        self.use_global_map = env.unwrapped.env.use_global_map

        if self.use_global_map:
            self.global_map_network = nn.Sequential(
                nn.LazyConv2d(32, 8, stride=4),
                nn.ReLU(),
                nn.LazyConv2d(64, 4, stride=2),
                nn.ReLU(),
                nn.LazyConv2d(64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(480),
                nn.ReLU(),
            )

        self.register_buffer(
            "screen_buckets", torch.tensor(PIXEL_VALUES, dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "linear_buckets", torch.tensor([0, 64, 128, 255], dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "unpack_mask",
            torch.tensor([0xC0, 0x30, 0x0C, 0x03], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "unpack_shift", torch.tensor([6, 4, 2, 0], dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "unpack_bytes_mask",
            torch.tensor([0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "unpack_bytes_shift",
            torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.uint8),
            persistent=False,
        )

        self.map_embeddings = nn.Embedding(0xFF, 4, dtype=torch.float32)
        item_count = max(Items._value2member_map_.keys())
        self.item_embeddings = nn.Embedding(
            item_count, int(item_count**0.25 + 1), dtype=torch.float32
        )

        self.party_network = nn.Sequential(nn.LazyLinear(6), nn.ReLU(), nn.Flatten())
        self.species_embeddings = nn.Embedding(0xBE, int(0xBE**0.25) + 1, dtype=torch.float32)
        self.type_embeddings = nn.Embedding(0x1A, int(0x1A**0.25) + 1, dtype=torch.float32)
        self.moves_embeddings = nn.Embedding(0xA4, int(0xA4**0.25) + 1, dtype=torch.float32)

        n_events = env.env.observation_space["events"].shape[0]
        # self.event_embeddings = nn.Embedding(n_events, int(n_events**0.25) + 1, dtype=torch.float32)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        observations = observations.type(torch.uint8)
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        # --- 1. Vision Processing ---
        screen = observations["screen"]
        visited_mask = observations["visited_mask"]
        restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
        
        if self.two_bit:
            screen = torch.index_select(
                self.screen_buckets,
                0,
                ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
            ).reshape(restored_shape)
            visited_mask = torch.index_select(
                self.linear_buckets,
                0,
                ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                .flatten()
                .int(),
            ).reshape(restored_shape)
        
        image_observation = torch.cat((screen, visited_mask), dim=-1)
        if self.channels_last:
            image_observation = image_observation.permute(0, 3, 1, 2)
        
        if self.downsample > 1:
            image_observation = image_observation[:, :, :: self.downsample, :: self.downsample]

        vision_features = self.screen_network(image_observation.float() / 255.0)
        
        # --- 2. Text Processing ---
        # Input shape: (Batch, SequenceLen) -> (Batch, 72)
        text_input = observations["text"].long()
        text_embed = self.text_embedding(text_input) # (Batch, 72, 64)
        
        # GRU
        # output, h_n = gru(input)
        # output: (Batch, SeqLen, Hidden)
        # h_n: (NumLayers, Batch, Hidden)
        _, h_n = self.text_gru(text_embed)
        text_features = h_n[-1] # Take the last layer hidden state: (Batch, 64)

        # --- 3. Projection & Fusion ---
        Hv = self.vision_proj(vision_features) # (Batch, 512)
        Ht = self.text_proj(text_features)     # (Batch, 512)

        # Gate calculation
        # z = Sigmoid(Linear(Concat(Hv, Ht)))
        cat_fusion = torch.cat((Hv, Ht), dim=-1)
        z = torch.sigmoid(self.gate_linear(cat_fusion)) # (Batch, 1)

        # Fused State: Hfused = z * Hv + (1-z) * Ht
        Hfused = z * Hv + (1 - z) * Ht
        
        # --- 4. Auxiliary Features (Party, Events, etc) ---
        # We need to concatenate these to Hfused or handle them separately.
        # In the original MultiConvolutionalPolicy, everything is concatenated then passed to `encode_linear`.
        # But here, we are replacing the main visual path with the fused path.
        # Let's concatenate the aux features to Hfused and pass through a final linear layer if needed?
        # Or, we can treat Hfused as the "Main Context" and append the rest.
        
        # Let's see what MultiConvolutionalPolicy does.
        # It concats (Screen, Direction, Party, Events...) -> encode_linear -> Output.
        
        # We should probably maintain valid gradients for everything.
        # So we will concat (Hfused, AuxFeatures) -> FinalProjection.
        
        # ... (Auxiliary feature processing same as original) ...
        # Party
        species = self.species_embeddings(observations["species"][..., :6].int()).float().squeeze(1)
        status = one_hot(observations["status"][..., :6].int(), 7).float().squeeze(1)
        type1 = self.type_embeddings(observations["type1"][..., :6].int()).squeeze(1)
        type2 = self.type_embeddings(observations["type2"][..., :6].int()).squeeze(1)
        moves = (
            self.moves_embeddings(observations["moves"].int())
            .squeeze(1)
            .float()
            .reshape((-1, 6, 4 * self.moves_embeddings.embedding_dim))
        )
        party_obs = torch.cat(
            (
                species,
                observations["hp"].float().unsqueeze(-1) / 714.0,
                status,
                type1,
                type2,
                observations["level"][..., :6].float().unsqueeze(-1) / 100.0,
                observations["maxHP"].float().unsqueeze(-1) / 714.0,
                observations["attack"].float().unsqueeze(-1) / 714.0,
                observations["defense"].float().unsqueeze(-1) / 714.0,
                observations["speed"].float().unsqueeze(-1) / 714.0,
                observations["special"].float().unsqueeze(-1) / 714.0,
                moves,
            ),
            dim=-1,
        )
        party_latent = self.party_network(party_obs)

        # Events
        events_obs = (
            (
                (
                    (observations["events"].reshape((-1, 1)) & self.unpack_bytes_mask)
                    >> self.unpack_bytes_shift
                )
                .flatten()
                .reshape((observations["events"].shape[0], -1))[:, EVENTS_IDXS]
            )
            .float()
            .squeeze(1)
        )
        
        # Other scalars
        map_id = self.map_embeddings(observations["map_id"].int()).squeeze(1)
        blackout_map_id = self.map_embeddings(observations["blackout_map_id"].int()).squeeze(1)
        items = (
            self.item_embeddings(observations["bag_items"].int())
            * (observations["bag_quantity"].float().unsqueeze(-1) / 100.0)
        ).squeeze(1)

        # Concat everything
        cat_obs = torch.cat(
            (
                Hfused, # Replaces screen_network output
                one_hot(observations["direction"].int(), 4).float().squeeze(1),
                one_hot(observations["battle_type"].int(), 4).float().squeeze(1),
                map_id.squeeze(1),
                blackout_map_id.squeeze(1),
                items.flatten(start_dim=1),
                party_latent,
                events_obs,
                observations["rival_3"].float().reshape(-1, 1),
                observations["game_corner_rocket"].float().reshape(-1, 1),
                observations["saffron_guard"].float().reshape(-1, 1),
                observations["lapras"].float().reshape(-1, 1),
            )
            + (() if self.skip_safari_zone else (observations["safari_steps"].float().reshape(-1, 1) / 502.0,))
            + (
                (self.global_map_network(observations["global_map"].float() / 255.0).squeeze(1),)
                if self.use_global_map
                else ()
            ),
            dim=-1,
        )
        
        # We need a final projection if we want to match the hidden definition of RecurrentNetwork
        # But wait, RecurrentNetwork wraps an LSTM. The input to the LSTM is what we return here.
        # The LSTM hidden size is 'hidden_size'.
        # So we should probably project 'cat_obs' to 'hidden_size'.
        return self.encode_linear(cat_obs), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
