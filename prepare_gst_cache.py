import os
import torch
from tqdm import tqdm
from helper.T import EEGToGeneralizedST

def main():
    # Set your input and output directories
    # (Double check that the input directory matches your actual raw 10-sec cache folder name)
    input_dir = "cache_windows_binary_10_sec_eval" 
    output_dir = "cache_gst_binary_10_sec_eval"
    
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the Vectorized GST
    gst_transform = EEGToGeneralizedST(
        fs=250, min_freq=1, max_freq=40, num_bins=40, time_pool_factor=16, p=1.0
    ).to(device)
    
    gst_transform.eval()

    # Get all .pt files
    files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
    print(f"Found {len(files)} files to convert to S-Transforms...")

    with torch.no_grad():
        for f in tqdm(files, desc="Pre-computing GST"):
            in_path = os.path.join(input_dir, f)
            out_path = os.path.join(output_dir, f)

            # Skip if already processed
            if os.path.exists(out_path):
                continue

            # 1. Load the raw data
            data = torch.load(in_path, weights_only=False)

            # Extract tensor
            if isinstance(data, dict):
                x = data["x"].to(dtype=torch.float32) # Keep on CPU initially!
            else:
                x = data.to(dtype=torch.float32)

            # 2. Add fake batch dimension if it's a single window
            if x.dim() == 2:
                x = x.unsqueeze(0)

            # ==========================================
            # THE FIX: Process the file in mini-batches
            # ==========================================
            chunk_size = 16 # Safe size for 12GB GPU
            gst_chunks = []
            
            # Loop through the patient's windows 32 at a time
            for i in range(0, x.size(0), chunk_size):
                # Move just this small chunk to the GPU
                x_chunk = x[i : i + chunk_size].to(device)
                
                # Apply the heavy GST math
                chunk_gst = gst_transform(x_chunk)
                
                # Move the finished 2D image back to CPU RAM immediately
                gst_chunks.append(chunk_gst.cpu())
                
                # Clear GPU cache to be safe
                torch.cuda.empty_cache()

            # Stitch the chunks back together into one file
            x_gst_full = torch.cat(gst_chunks, dim=0)
            # ==========================================

            # 5. Save it back to disk
            if isinstance(data, dict):
                data["x"] = x_gst_full
                torch.save(data, out_path)
            else:
                torch.save(x_gst_full, out_path)

    print(f"\nFinished! All GST files saved to: {output_dir}")

if __name__ == "__main__":
    main()