"""
Asymmetric MAE architecture implementation

Extracted from: part2/chapter04/masked_image_modeling.tex
Block: 5
Lines: 67
"""

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, encoder_layers=24, 
                 decoder_layers=8, encoder_dim=1024, decoder_dim=512):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, encoder_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # Learnable mask token for decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Encoder (processes visible patches only)
        self.encoder = TransformerEncoder(
            embed_dim=encoder_dim,
            num_layers=encoder_layers,
            num_heads=16
        )
        
        # Projection from encoder to decoder
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        
        # Decoder (processes all patches)
        self.decoder = TransformerDecoder(
            embed_dim=decoder_dim,
            num_layers=decoder_layers,
            num_heads=16
        )
        
        # Reconstruction head
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * 3)
        
        # Position embeddings
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, encoder_dim)
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim)
        )
    
    def forward_encoder(self, x, mask):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embeddings
        x = x + self.encoder_pos_embed[:, 1:, :]
        
        # Apply mask (remove masked patches)
        x = x[~mask].reshape(x.shape[0], -1, x.shape[-1])
        
        # Add cls token
        cls_token = self.encoder_pos_embed[:, :1, :] 
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Encoder forward pass
        x = self.encoder(x)
        
        return x
    
    def forward_decoder(self, x, ids_restore):
        # Project to decoder dimension
        x = self.encoder_to_decoder(x)
        
        # Add mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        
        # Unshuffle
        x_ = torch.gather(x_, dim=1, 
                         index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Append cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add position embeddings
        x = x + self.decoder_pos_embed
        
        # Decoder forward pass
        x = self.decoder(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        # Prediction head
        x = self.decoder_pred(x)
        
        return x