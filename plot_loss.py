import os
import re
import matplotlib.pyplot as plt
import argparse

def parse_loss_log(log_path):
    if not os.path.exists(log_path):
        print(f"Log file {log_path} not found.")
        return []
        
    with open(log_path, 'r') as f:
        log_text = f.read()
    
    results = []
    # Pattern to match: (epoch: 1, iters: 100, time: 0.990, data: 42.654) G_GAN: 0.253 ...
    pattern = re.compile(r"\(epoch:\s*(\d+),\s*iters:\s*(\d+),\s*time:\s*([\d.]+),\s*data:\s*([\d.]+)\)\s*(.*)")
    
    for line in log_text.strip().split('\n'):
        match = pattern.search(line)
        if match:
            epoch, iters, time, data, rest = match.groups()
            row = {
                'epoch': int(epoch),
                'iters': int(iters)
            }
            # Find all loss items like "G_GAN: 0.253"
            loss_items = re.findall(r"(\w+):\s*([\d.]+)", rest)
            for key, value in loss_items:
                row[key] = float(value)
            results.append(row)
    return results

def plot_losses(data, save_dir):
    if not data:
        print("No data to plot.")
        return

    # Group keys for better visualization (Discriminator, Generator, NCE/IDT)
    all_keys = [k for k in data[0].keys() if k not in ['epoch', 'iters']]
    
    # Simple Plot: All in one
    plt.figure(figsize=(15, 10))
    x = range(len(data))
    
    for key in all_keys:
        y = [row[key] for row in data]
        plt.plot(x, y, label=key, alpha=0.8)
    
    plt.xlabel('Logged Iterations')
    plt.ylabel('Loss Value')
    plt.title('Training Losses Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'loss_curve_all.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Summary plot saved to {save_path}")

    # Separate Plot: GAN vs NCE vs Others
    # 1. GAN Losses
    gan_keys = [k for k in all_keys if 'GAN' in k or 'D_real' in k or 'D_fake' in k]
    if gan_keys:
        plt.figure(figsize=(12, 6))
        for key in gan_keys:
            plt.plot(x, [row[key] for row in data], label=key)
        plt.title('GAN & Discriminator Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curve_gan.png'))
        plt.close()

    # 2. Comparison/Consistency Losses
    nce_keys = [k for k in all_keys if 'NCE' in k or 'idt' in k]
    if nce_keys:
        plt.figure(figsize=(12, 6))
        for key in nce_keys:
            plt.plot(x, [row[key] for row in data], label=key)
        plt.title('NCE & Identity Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curve_nce.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot losses from CWR/CycleGAN loss_log.txt")
    parser.add_argument('--log_path', type=str, required=True, help="Path to loss_log.txt")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()
    
    log_data = parse_loss_log(args.log_path)
    plot_losses(log_data, args.save_dir)
