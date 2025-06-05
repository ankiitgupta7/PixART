import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models as tv_models
from tqdm import tqdm
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
import timm
import pandas as pd

# === CONFIG ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_ITER = 50000
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5
SEED = 4242
random.seed(SEED)
ROOT_DIR = "results_all"

MODES = ['blank'] # 'white'

MODELS = {
    "resnet50": lambda: tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT),
    "vgg16": lambda: tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT),
    "vit_base_patch16_224": lambda: timm.create_model('vit_base_patch16_224', pretrained=True)
}

# === LOAD IMAGENET LABELS ===
LABELS_FILE = "imagenet_classes.txt"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

if os.path.isfile(LABELS_FILE):
    with open(LABELS_FILE, 'r') as f:
        imagenet_classes = [line.strip() for line in f.readlines()]
else:
    imagenet_classes = urllib.request.urlopen(LABELS_URL).read().decode("utf-8").splitlines()
    with open(LABELS_FILE, 'w') as f:
        f.write('\n'.join(imagenet_classes))

# === TRANSFORMS ===
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
unnormalize = T.Normalize(
    mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    std=[1/s for s in [0.229, 0.224, 0.225]]
)
to_pil = T.ToPILImage()

# === IMAGE UTILS ===
def create_tensor(mode):
    if mode == 'random':
        img = torch.rand((1, 3, *IMAGE_SIZE))
    elif mode == 'blank':
        img = torch.zeros((1, 3, *IMAGE_SIZE))
    elif mode == 'white':
        img = torch.ones((1, 3, *IMAGE_SIZE))
    else:
        raise ValueError(f"Unknown mode {mode}")
    return normalize(img).to(device)

def save_tensor_image(tensor, path):
    img = unnormalize(tensor.squeeze().cpu().detach())
    img = torch.clamp(img, 0, 1)
    to_pil(img).save(path)

def plot_confidence_curve(log_df, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(log_df['iter'], log_df['target_conf'])
    plt.xlabel("Generation")
    plt.ylabel("Confidence")
    plt.title("Confidence vs. Generation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_combined_curves(log_dict, save_path):
    plt.figure(figsize=(8, 4))
    for mode, df in log_dict.items():
        plt.plot(df['iter'], df['target_conf'], label=mode)
    plt.xlabel("Generation")
    plt.ylabel("Confidence")
    plt.title("Confidence Evolution by Mode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === ATTACK FUNCTION ===
def evolve_image_for_class(model, model_name, target_idx, mode, max_iter=MAX_ITER, early_stop_threshold=0.99):
    model.eval()
    base_img = create_tensor(mode)
    img = base_img.clone()

    best_conf = 0.0
    best_img = img.clone()
    log_rows = []

    pbar = tqdm(range(max_iter), desc=f"{model_name} | {mode} | Class {target_idx}", leave=False)
    for i in pbar:
        candidate = img.clone()
        x = random.randint(0, IMAGE_SIZE[1] - 1)
        y = random.randint(0, IMAGE_SIZE[0] - 1)
        ch = random.randint(0, 2)
        candidate[0, ch, y, x] = torch.rand(1).item()

        with torch.no_grad():
            probs = F.softmax(model(candidate), dim=1)
            target_conf = probs[0, target_idx].item()
            top_probs, top_idxs = probs.topk(3)
            top_probs = top_probs.squeeze(0).tolist()
            top_idxs = top_idxs.squeeze(0).tolist()

        l2 = torch.norm((candidate - base_img)).item()

        log_rows.append({
            "iter": i,
            "target_conf": target_conf,
            "l2": l2,
            "top1_label": imagenet_classes[top_idxs[0]],
            "top1_conf": top_probs[0],
            "top2_label": imagenet_classes[top_idxs[1]],
            "top2_conf": top_probs[1],
            "top3_label": imagenet_classes[top_idxs[2]],
            "top3_conf": top_probs[2],
        })

        pbar.set_description(f"{model_name} | {mode} | Gen {i} | Conf: {target_conf:.4f}")

        if target_conf > best_conf:
            best_conf = target_conf
            best_img = candidate.clone()
            img = candidate

        if target_conf >= early_stop_threshold:
            break

    return {
        "adv_img": best_img,
        "log": pd.DataFrame(log_rows),
        "success_gen": i,
        "final_conf": best_conf
    }

# === MAIN EXPERIMENT ===
summary_rows = []
selected_indices = sorted(random.sample(range(1000), NUM_CLASSES))
print(f"🎯 Selected classes (seed={SEED}):")
for idx in selected_indices:
    print(f"- {idx:03d}: {imagenet_classes[idx]}")

for target_idx in tqdm(selected_indices, desc="All classes"):
    target_class = imagenet_classes[target_idx].replace(" ", "_")

    for model_name, model_loader in tqdm(MODELS.items(), desc=f"Models for {target_idx}", leave=False):
        print(f"\n🚀 {model_name} on {target_class} (idx {target_idx})")
        model = model_loader().to(device)

        logs = {}
        for mode in MODES:
            result = evolve_image_for_class(model, model_name, target_idx, mode=mode)

            # Save directory
            save_dir = os.path.join(ROOT_DIR, mode, model_name, f"{target_idx:03d}_{target_class}")
            os.makedirs(save_dir, exist_ok=True)

            # Save outputs
            save_tensor_image(result["adv_img"], os.path.join(save_dir, "evolved_image.png"))
            result["log"].to_csv(os.path.join(save_dir, "log.csv"), index=False)
            plot_confidence_curve(result["log"], os.path.join(save_dir, "confidence_plot.png"))

            logs[mode] = result["log"]
            summary_rows.append({
                "model": model_name,
                "target_idx": target_idx,
                "target_class": target_class,
                "mode": mode,
                "final_confidence": result['final_conf'],
                "generations": result['success_gen']
            })

        # Combined plot
        combined_plot_path = os.path.join(ROOT_DIR, "combined", model_name)
        os.makedirs(combined_plot_path, exist_ok=True)
        plot_combined_curves(logs, os.path.join(combined_plot_path, f"{target_idx:03d}_{target_class}_compare.png"))

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(ROOT_DIR, "summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n Summary saved to: {summary_path}")
