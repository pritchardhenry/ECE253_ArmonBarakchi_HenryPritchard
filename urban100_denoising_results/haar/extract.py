import os
import re
import pdfplumber


# --- INSTRUCTIONS ---
# 1. Install pdfplumber: pip install pdfplumber
# 2. Place this script in the directory with your PDF files.
# 3. Run the script: python your_script_name.py

def get_denoised_metrics(pdf_text):
    """
    Extracts the last triplet of PSNR, SSIM, and LPIPS values from a string of text
    using regular expressions. These are assumed to be the 'denoised' metrics.

    Returns: (denoised_psnr, denoised_ssim, denoised_lpips) or (None, None, None)
    """
    # Regex patterns for the three metrics
    psnr_pattern = r"PSNR:\s*(\d+\.\d+)"
    ssim_pattern = r"SSIM:\s*(\d+\.\d+)"
    lpips_pattern = r"LPIPS:\s*(\d+\.\d+)"  # Added LPIPS pattern

    # Find all occurrences
    all_psnr = re.findall(psnr_pattern, pdf_text)
    all_ssim = re.findall(ssim_pattern, pdf_text)
    all_lpips = re.findall(lpips_pattern, pdf_text)  # Find all LPIPS

    # Return the last one for each metric, which should correspond to the Denoised result
    if all_psnr and all_ssim and all_lpips:
        try:
            denoised_psnr = float(all_psnr[-1])
            denoised_ssim = float(all_ssim[-1])
            denoised_lpips = float(all_lpips[-1])  # Extract the last LPIPS

            return denoised_psnr, denoised_ssim, denoised_lpips
        except ValueError:
            return None, None, None

    return None, None, None


def calculate_average_metrics(folder_path="."):
    """
    Iterates through PDF files, extracts metrics, and calculates the average.
    """
    # 1. Find all PDF files in the current folder
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    # Initialize totals for all three metrics
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0  # New total for LPIPS
    file_count = 0

    print(f"🔎 Processing {len(pdf_files)} PDF files...")

    for filename in pdf_files:
        full_path = os.path.join(folder_path, filename)
        file_text = ""

        # 2. Extract text using pdfplumber
        try:
            with pdfplumber.open(full_path) as pdf:
                for page in pdf.pages:
                    # Concatenate all text from all pages
                    file_text += page.extract_text() or ""

            # 3. Extract metrics from the text
            psnr, ssim, lpips = get_denoised_metrics(file_text)

            if psnr is not None and ssim is not None and lpips is not None:
                total_psnr += psnr
                total_ssim += ssim
                total_lpips += lpips  # Add to LPIPS total
                file_count += 1
                print(f"✅ Extracted from {filename}: PSNR={psnr:.2f}, SSIM={ssim:.3f}, LPIPS={lpips:.3f}")
            else:
                print(f"⚠️ Could not extract all three metrics (PSNR, SSIM, LPIPS) from {filename}. Skipping.")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}. Skipping.")

    print("\n--- Final Summary ---")
    if file_count > 0:
        average_psnr = total_psnr / file_count
        average_ssim = total_ssim / file_count
        average_lpips = total_lpips / file_count  # Calculate average LPIPS

        print(f"Total files processed: {file_count}")
        print(f"🚀 **AVERAGE DENOISED PSNR**: **{average_psnr:.4f}**")
        print(f"🚀 **AVERAGE DENOISED SSIM**: **{average_ssim:.4f}**")
        print(f"🚀 **AVERAGE DENOISED LPIPS**: **{average_lpips:.4f}**")  # Print average LPIPS
    else:
        print("No complete set of metrics (PSNR, SSIM, LPIPS) was successfully extracted from any file.")


if __name__ == "__main__":
    calculate_average_metrics()