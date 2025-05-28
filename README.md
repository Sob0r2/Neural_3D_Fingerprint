
# Neural 3D fingerprint

This project explores the creation of 3D-aware molecular fingerprints using graph neural networks.  
The goal is to generate fixed-size vector representations of molecules that capture both their 2D topology   
and 3D conformation, enabling more meaningful comparisons between molecules.

We use a contrastive learning setup, where the model learns to bring the fingerprints of similar molecules closer together in embedding space. Similarity is defined based on both chemical structure (2D) and molecular geometry (3D). A key component of the project is to identify similar molecule pairs from datasets and use them as positive pairs during training.

This approach ensures that the learned fingerprints are not only robust to conformational changes but also encode subtle structural and chemical similarities, making them useful for tasks like property prediction, similarity search, and molecular clustering.

## How to Get Started

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

Clone the repository to your local machine using the following command:

```bash
git clone https://github.com/Sob0r2/Neural_3D_Fingerprint.git
cd Neural_3D_Fingerprint
```

### 2. Create a Virtual Environment

To create a virtual environment, use the `environment.yaml` file provided in the repository. This will set up the necessary dependencies for the project.

```bash
conda env create -f environment.yaml
conda activate mldd25_project
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

### 3. Set Up the `.env` File

The project uses environment variables stored in a `.env` file. To set it up:

1. Copy the `.env.example` file to `.env`:

   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file in a text editor and fill in the required paths according to the instructions in the file.

   The `.env` file include paths for datasets and model weights.
## Requirements

The full list of dependencies is available in `environment.yaml`.  
Below are the key requirements:

- Python ≥ 3.8  
- [PyTorch](https://pytorch.org/)  
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)  
- [RDKit](https://www.rdkit.org/)  
- NumPy  
- pandas  
- scikit-learn  
- tqdm
# Datasets

This project uses two main datasets:

- **QM9**  
  A dataset of small molecules containing up to 9 heavy atoms, with available conformers.  
  Source: [Curated QM9](https://moldis-group.github.io/curatedQM9/)  
  QM9 is used for training the GNN in a contrastive manner to generate 3D fingerprints, and for evaluating these fingerprints on 3D tasks such as predicting zero-point vibrational energy (ZPVE) or dipole moment (μ).

- **CHEMBL**  
  Data related to serotonin receptors, used to evaluate the 2D performance of fingerprints on tasks like predicting chemical activity (Ki).  
  *(Add download link for CHEMBL dataset here)*

---

## How to prepare the datasets

### CHEMBL  
The data is already available in the folder:  
```
data/CHEMBL
```
If you want to add more receptor data, just copy the corresponding `.csv` files into this folder.

### QM9  
Preparing QM9 requires several preprocessing steps:  

1. Unpack the QM9 data into the folder:  
```
data/
```
2. Run the following preprocessing scripts in order:  
```bash
python app/data_preprocessing/xyz_to_json.py
python app/data_preprocessing/run_preprocessing.py
```
3. Convert each data record into a graph representation by running:  
```bash
python app/create_pos_pairs/run_faiss.py
```
> **Note:** This step may take several hours depending on your hardware.
# Training and Adjusting the Model

To train the model from scratch, run the training notebook:

```python
# Run this notebook to train the model
python app/model/train_model.ipynb
```

The final trained model weights are saved in:

```plaintext
models/FINAL_MODEL.pth
```

Use these weights in the notebooks located in the `hypotheses/` folder for evaluation or inference.

If you retrain the model or tune hyperparameters, remember to update the path to the weights file in the `hypotheses/` notebooks accordingly.

## Results

TO:DO
