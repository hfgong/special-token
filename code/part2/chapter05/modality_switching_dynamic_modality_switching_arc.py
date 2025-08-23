"""
Dynamic modality switching architecture

Extracted from: part2/chapter05/modality_switching.tex
Block: 1
Lines: 165
"""

class ModalitySwitchingLayer(nn.Module):
    def __init__(self, embed_dim=768, num_modalities=3):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # Modality importance predictor
        self.modality_importance = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_modalities),
            nn.Sigmoid()
        )
        
        # Modality-specific gates
        self.modality_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # Cross-modality routing
        self.routing_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        
        # Switching control tokens
        self.switching_tokens = nn.Parameter(
            torch.randn(num_modalities, embed_dim)
        )
        
        # Fusion mechanisms
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, modality_inputs, modality_masks=None):
        """
        Args:
            modality_inputs: List of [B, seq_len, embed_dim] tensors for each modality
            modality_masks: List of boolean masks indicating available modalities
        """
        batch_size = modality_inputs[0].shape[0]
        device = modality_inputs[0].device
        
        # Global context for switching decisions
        global_context = torch.stack([
            modal_input.mean(dim=1) for modal_input in modality_inputs
        ], dim=1)  # [B, num_modalities, embed_dim]
        
        # Predict modality importance
        importance_context = global_context.mean(dim=1)  # [B, embed_dim]
        modality_importance = self.modality_importance(importance_context)  # [B, num_modalities]
        
        # Apply availability masks
        if modality_masks is not None:
            for i, mask in enumerate(modality_masks):
                modality_importance[:, i] *= mask.float()
        
        # Normalize importance scores
        modality_importance = F.softmax(modality_importance, dim=-1)
        
        # Apply modality-specific gates
        gated_outputs = []
        for i, (modal_input, gate) in enumerate(zip(modality_inputs, self.modality_gates)):
            # Compute gate values
            gate_values = gate(modal_input)  # [B, seq_len, embed_dim]
            
            # Apply importance weighting
            importance_weight = modality_importance[:, i].unsqueeze(-1).unsqueeze(-1)
            gated_output = modal_input * gate_values * importance_weight
            
            gated_outputs.append(gated_output)
        
        # Cross-modality routing with switching tokens
        switching_tokens = self.switching_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate all gated modality outputs
        all_modal_tokens = torch.cat(gated_outputs, dim=1)  # [B, total_seq_len, embed_dim]
        
        # Route information through switching tokens
        routed_output, routing_attention = self.routing_attention(
            query=switching_tokens,
            key=all_modal_tokens,
            value=all_modal_tokens
        )
        
        # Adaptive fusion
        routed_flat = routed_output.view(batch_size, -1)  # [B, num_modalities * embed_dim]
        fused_output = self.adaptive_fusion(routed_flat)  # [B, embed_dim]
        
        return {
            'fused_output': fused_output,
            'modality_importance': modality_importance,
            'routing_attention': routing_attention,
            'gated_outputs': gated_outputs
        }

class AdaptiveMultimodalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_modalities=3):
        super().__init__()
        
        # Modality encoders
        self.text_encoder = nn.Embedding(vocab_size, embed_dim)
        self.visual_encoder = VisionTransformer(embed_dim=embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        
        # Modality switching layers
        self.switching_layers = nn.ModuleList([
            ModalitySwitchingLayer(embed_dim, num_modalities) for _ in range(4)
        ])
        
        # Task-specific adapters
        self.task_adapters = nn.ModuleDict({
            'classification': nn.Linear(embed_dim, vocab_size),
            'retrieval': nn.Linear(embed_dim, embed_dim),
            'generation': nn.Linear(embed_dim, vocab_size)
        })
        
        # Modality availability detector
        self.availability_detector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_modalities),
            nn.Sigmoid()
        )
    
    def forward(self, text_ids=None, images=None, audio_features=None, 
                task='classification', modality_preferences=None):
        
        # Encode available modalities
        modality_inputs = []
        modality_masks = []
        
        # Text modality
        if text_ids is not None:
            text_tokens = self.text_encoder(text_ids)
            modality_inputs.append(text_tokens)
            modality_masks.append(torch.ones(text_tokens.shape[0], device=text_tokens.device))
        else:
            # Create dummy input
            batch_size = images.shape[0] if images is not None else audio_features.shape[0]
            dummy_text = torch.zeros(batch_size, 1, self.embed_dim, device=self.get_device())
            modality_inputs.append(dummy_text)
            modality_masks.append(torch.zeros(batch_size, device=self.get_device()))
        
        # Visual modality
        if images is not None:
            visual_tokens = self.visual_encoder(images)
            modality_inputs.append(visual_tokens)
            modality_masks.append(torch.ones(visual_tokens.shape[0], device=visual_tokens.device))
        else:
            batch_size = len(modality_inputs[0])
            dummy_visual = torch.zeros(batch_size, 1, self.embed_dim, device=self.get_device())
            modality_inputs.append(dummy_visual)
            modality_masks.append(torch.zeros(batch_size, device=self.get_device()))
        
        # Audio modality
        if audio_features is not None:
            audio_tokens = self.audio_encoder(audio_features)
            modality_inputs.append(audio_tokens)
            modality_masks.append(torch.ones(audio_tokens.shape[0], device=audio_tokens.device))
        else:
            batch_size = len(modality_inputs[0])
            dummy_audio = torch.zeros(batch_size, 1, self.embed_dim, device=self.get_device())
            modality_inputs.append(dummy_audio)
            modality_masks.append(torch.zeros(batch_size, device=self.get_device()))
        
        # Progressive modality switching
        switching_outputs = []
        current_inputs = modality_inputs
        
        for switching_layer in self.switching_layers:
            switch_output = switching_layer(current_inputs, modality_masks)
            switching_outputs.append(switch_output)
            
            # Update inputs for next layer
            fused_repr = switch_output['fused_output'].unsqueeze(1)  # [B, 1, embed_dim]
            current_inputs = [fused_repr] * len(modality_inputs)
        
        # Final representation
        final_representation = switching_outputs[-1]['fused_output']
        
        # Task-specific processing
        if task in self.task_adapters:
            output = self.task_adapters[task](final_representation)
        else:
            output = final_representation
        
        return {
            'output': output,
            'switching_outputs': switching_outputs,
            'modality_importance': switching_outputs[-1]['modality_importance'],
            'final_representation': final_representation
        }
    
    def get_device(self):
        return next(self.parameters()).device