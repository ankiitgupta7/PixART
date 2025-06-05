import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt, os
from tqdm import tqdm
import argparse

# --- ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g. mnistFashion)')
parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g. SVM)')
parser.add_argument('--target_class', type=int, required=True, help='Class index of image (0-9)')
parser.add_argument('--iters', type=int, default=1000, help='Number of degradation steps')
parser.add_argument('--model_base', type=str, default='model_info', help='Path to model root directory')
parser.add_argument('--outdir', type=str, default='results_degraded', help='Base output dir for degradation')
args = parser.parse_args()

# --- PATHS ---
model_path = os.path.join(args.model_base, args.dataset, args.model_name, 'trained_model.pkl')
img_pkl_path = os.path.join(
    args.model_base, args.dataset, args.model_name,
    'best_images/test/pkl', f'class_{args.target_class}_best_image.pkl'
)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
if not os.path.exists(img_pkl_path):
    raise FileNotFoundError(f"Best image not found at {img_pkl_path}")

# --- LOAD MODEL & IMAGE ---
clf = joblib.load(model_path)
raw = joblib.load(img_pkl_path)

if not isinstance(raw, dict) or 'image' not in raw:
    raise ValueError("Expected the .pkl file to be a dict with an 'image' key")

img = raw['image']
if img.shape != (784,):
    img = img.flatten()


print(f"Loaded best image: class={raw.get('class', '?')}, confidence={raw.get('confidence', 0.0):.4f}, model={raw.get('model', 'unknown')}")


# --- INIT ---
log = []
snap_interval = args.iters // 10

# --- OUTPUT STRUCTURE ---
exp_dir = os.path.join(args.outdir, args.dataset, args.model_name, f"degrade_class_{args.target_class}")
snap_dir = os.path.join(exp_dir, "snapshots")
os.makedirs(snap_dir, exist_ok=True)

# --- DEGRADATION LOOP ---
pbar = tqdm(range(args.iters), desc=f"Class {args.target_class} | Conf: {clf.predict_proba([img])[0][args.target_class]:.4f}")
for i in pbar:
    candidate = img.copy()
    idx = np.random.randint(784)
    candidate[idx] = np.random.rand()

    conf_before = clf.predict_proba([img])[0][args.target_class]
    conf_after = clf.predict_proba([candidate])[0][args.target_class]

    if conf_after < conf_before:
        img = candidate

    log.append({'iter': i, 'changed_pixel': idx, 'conf': conf_after})
    pbar.set_description(f"Class {args.target_class} | Conf: {conf_after:.4f}")

    if (i + 1) % snap_interval == 0 or i == args.iters - 1:
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f"Gen {i+1} | Conf: {conf_after:.4f}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(snap_dir, f"gen_{i+1:04d}.png"))
        plt.close()

# --- SAVE RESULTS ---
np.save(os.path.join(exp_dir, "degraded_image.npy"), img)

plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Degraded | Class {args.target_class}")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, "degraded_image.png"))
plt.close()

df = pd.DataFrame(log)
df.to_csv(os.path.join(exp_dir, "log.csv"), index=False)

plt.figure()
plt.plot(df['iter'], df['conf'])
plt.xlabel("Generation")
plt.ylabel("Confidence")
plt.title(f"Confidence Drop vs. Generation (Class {args.target_class})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, "confidence_drop_plot.png"))
plt.close()

print(f"\n✅ Degradation complete. Output saved to: {exp_dir}")
