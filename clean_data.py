
# DATA PRE-PROCESSING (BACKGROUND REMOVAL)

#!pip install rembg

import os
from rembg import remove
from PIL import Image

RAW_REAL_DIR = "/content/drive/MyDrive/cleaned_real"
RAW_AI_DIR   = "/content/drive/MyDrive/cleaned_ai"

CLEAN_REAL_DIR = "/content/drive/MyDrive/CLEAN_DATASET/real"
CLEAN_AI_DIR   = "/content/drive/MyDrive/CLEAN_DATASET/ai"

def clean_backgrounds(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"🧹 Processing {len(files)} images from {input_dir}...")
    for f in files:
        inp_path = os.path.join(input_dir, f)
        # Save as PNG to avoid JPEG compression artifacts on the new white background
        out_path = os.path.join(output_dir, f"{os.path.splitext(f)[0]}.png") 
        
        if os.path.exists(out_path): continue # Skip if already processed
            
        try:
            # Read image, remove background (creates RGBA with transparency)
            img = Image.open(inp_path).convert("RGBA")
            no_bg_img = remove(img)
            
            # Create a solid white background canvas
            white_canvas = Image.new("RGB", no_bg_img.size, (255, 255, 255))
            
            # Paste the receipt onto the white canvas using the alpha channel as a mask
            white_canvas.paste(no_bg_img, mask=no_bg_img.split()[3])
            white_canvas.save(out_path, "PNG")
        except Exception as e:
            print(f"Failed on {f}: {e}")

# Run the background removal
clean_backgrounds(RAW_REAL_DIR, CLEAN_REAL_DIR)
clean_backgrounds(RAW_AI_DIR, CLEAN_AI_DIR)
print("Background removal complete. All images are now flat on white canvases.")