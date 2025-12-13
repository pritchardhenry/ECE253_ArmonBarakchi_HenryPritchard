# A Comparative Analysis of 3D Transforms in the BM3D Algorithm

**Armon Barakchi and Henry Pritchard**  
**ECE 253 — Fall 2025**

---

## Repository Usage

This repository provides scripts for evaluating BM3D under different noise and degradation models, using either **custom local images** or the **Urban100 dataset**.

---

## Running Experiments on a Local Image

1. Navigate to the repository’s base directory.

2. The relevant scripts are:
   - `test_denoising_local_input.py`
   - `test_shotnoise_local_input.py`
   - `test_deblurring_local_input.py`

3. To view available command-line options and parameters, run:
   ```bash
   python test_denoising_local_input.py --help
   ```

4. Specify the input image using the `--test_img_path` flag. For example:
   ```bash
   python test_denoising_local_input.py --test_img_path test.png
   ```

---

## Running Experiments on the Urban100 Dataset

1. Navigate to the repository’s base directory.

2. The Urban100 image paths are handled automatically by the scripts.

3. The relevant scripts are:
   - `test_denoising_urban100.py`
   - `test_shotnoise_urban100.py`
   - `test_deblurring_urban100.py`

4. To view available command-line options and parameters, run:
   ```bash
   python test_denoising_urban100.py --help
   ```
