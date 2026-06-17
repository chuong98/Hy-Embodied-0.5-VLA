import torch
from huggingface_hub import snapshot_download
from hy_vla import HyVLA, HyVLAConfig

ckpt = snapshot_download("tencent/Hy-Embodied-0.5-VLA-RoboTwin")

config = HyVLAConfig.from_pretrained(ckpt)
policy = HyVLA.from_pretrained(ckpt, config=config)
policy.enable_video_encoder_if_needed()
policy = policy.to(device="cuda", dtype=torch.bfloat16).eval()

# (B, K, C, H, W); K=6 history slots
img = torch.zeros(1, 6, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
# Normalized dual-arm EEF: [xyz(3) + rot6d(6) + gripper(1)] * 2
state = torch.zeros((1, config.max_state_dim), device="cuda", dtype=torch.bfloat16)
batch = {
    "observation.images.top_head":   img,
    "observation.images.hand_left":  img,
    "observation.images.hand_right": img,
    "observation.state": state,
    "task": ["pick up the bottle"],
}

with torch.no_grad():
    actions = policy.forward_evaluate(batch)["pred"]
    actions = actions[..., : config.action_feature.shape[0]]
print(actions.shape)