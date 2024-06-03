
import os
import torch

# expt_name = "03-31-based-1b-50b-tok-restart/"
expt_name = "04-22-attn-1b-50b/"

# Load the checkpoint file
checkpoint = torch.load(f"/var/cr05_data/sim_data/checkpoints/{expt_name}/last.ckpt")

# Extract the model parameters
params = checkpoint["state_dict"]

# Save the parameters to a .bin file
# save_path = f"/work/sim_data/{expt_name}"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
torch.save(params, f"pytorch_model.bin")



