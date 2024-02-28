# 2sts-4dct-article

This repo contain code and data for "**2STS 4DCT (Single Shot per Time Step 4D Computed Tomography) Algorithm**" article by **Grigoriev M.V. & Buzmakov A.V.**

---

### How to use

1. You have to Python 3.x installed in your OS.
2. Copy (or clone) this repo to your local folder.
3. Install required packages:
   ```
    pip install -r requirements.txt
   ```
4. Run JupyterLab:
   ```
   jupyter lab
   ```
5. Open `2sts_4dct_article.ipynb` in JupyterLab

---

Files description:

`_fn.py` — file with functions used

`2sts_4dct_article.ipynb` — Jupyter notebook with implementation of proposed algorithm and code for data calculation

`2sts_4dct_article.py` — corresponding Python file

`bulk_data_8_8_1000.npy` — result data for reconstruction of 1000 dynamic objects with 8 randomly placed channels of 8 pixels radius

`bulk_data_16_4_1000.npy` — result data for reconstruction of 1000 dynamic objects with 16 randomly placed channels of 4 pixels radius

`bulk_data_32_2_1000.npy` — result data for reconstruction of 1000 dynamic objects with 32 randomly placed channels of 2 pixels radius

`requirements.txt` — packages list with versions used

---