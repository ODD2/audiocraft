import torch
from audiocraft.models.musicgen import MusicGen


model = MusicGen.get_pretrained('facebook/musicgen-small')

sd = model.lm.state_dict()

device = list(sd.values())[0].device

# modify weights here
# from od.svl import SynoVideoAttrExtractor, OpMode
# syno = SynoVideoAttrExtractor("ViT-L/14", False, op_mode=OpMode.S | OpMode.T, num_frames=30)
# sd = {
#     **{
#         f"condition_provider.conditioners.description._decoder.{k}": v.to(device)
#         for k, v in syno.decoder.named_parameters()
#     },
#     **sd
# }
# small
sd['condition_provider.conditioners.description.output_proj.weight'] = torch.zeros((768, 1024), device=device)
sd['condition_provider.conditioners.description.output_proj.bias'] = torch.zeros((1024), device=device)
# # medium
# sd['condition_provider.conditioners.description.output_proj.weight'] = torch.zeros((1536, 1024), device=device)
# sd['condition_provider.conditioners.description.output_proj.bias'] = torch.zeros((1536), device=device)
# # end weight modification

with open("./small_syno_L14_30F.pt", "wb") as f:
    torch.save(
        dict(
            best_state=dict(
                model=sd
            )
        ),
        f
    )
