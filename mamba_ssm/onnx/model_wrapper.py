import torch

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, create_block

class ModelWrapper(torch.nn.Module):
    def __init__(self, model_name=None, use_generation=False, config: MambaConfig=None, device='cpu', dtype=torch.float32):
        super(ModelWrapper, self).__init__()
        self.use_generation = use_generation
        if model_name is not None:
            self.model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=dtype)
        else:            
            self.model = MambaLMHeadModel(config=config if config is not None else MambaConfig(), device=device, dtype=dtype)
        self.model.eval()  
        print(f"Number of layers: {self.model.config.n_layer}")
        print(f"Size of d: {self.model.config.d_model}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    
    def forward(self, input_ids, *args):
        max_length = input_ids.shape[1] + 100
        if self.use_generation:
            return self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                cg=True,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                repetition_penalty=1.0,
            )
        else: 
            batch_size, _ = input_ids.shape
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
            return self.model(input_ids, inference_params=inference_params, *args)

class BlockModelWrapper(torch.nn.Module):
    def __init__(self, config: MambaConfig=None, device='cpu', dtype=torch.float32):
        super(BlockModelWrapper, self).__init__() 
        self.d_model = config.d_model
        self.model = create_block(
            d_model=config.d_model,
            ssm_cfg=config.ssm_cfg,
            norm_epsilon=1e-5,
            rms_norm=config.rms_norm,
            residual_in_fp32=config.residual_in_fp32,
            fused_add_norm=config.fused_add_norm,
            layer_idx=0,
            device=device, 
            dtype=dtype
        )
        print(f"Size of d: {self.d_model}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def forward(self, input_ids, hidden_states, residual):
        
        batch_size, _ = input_ids.shape
        max_length = input_ids.shape[1] + 100
        dummy_inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

        output, _ = self.model(hidden_states, residual, dummy_inference_params)
        return output