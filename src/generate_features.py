import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys


mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_molecule(smiles: str):
    """
    Convert SMILES string to RDKit molecule.
    """
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    return molecule


def molecule_to_features(molecule):
    """
    Convert RDKit molecule to fingerprint features.
    """

    if molecule is None:
        return np.zeros(2048 + 167, dtype=np.float32)

    morgan_fingerprint = mfpgen.GetFingerprintAsNumPy(molecule)
    maccs_fingerprint = np.array(
        MACCSkeys.GenMACCSKeys(molecule), dtype=np.float32)

    features = np.concatenate([morgan_fingerprint, maccs_fingerprint])

    return features.astype(np.float32)


def smiles_to_features(smiles: str):
    """
    Convert SMILES to molecular features.
    """
    molecule = smiles_to_molecule(smiles)
    features = molecule_to_features(molecule)

    return features


def create_feature_matrix(smiles_values):
    """
    Convert a list of SMILES into a feature matrix.
    """

    features = []

    for smiles in smiles_values:
        feature = smiles_to_features(smiles)
        features.append(feature)

    return np.array(features, dtype=np.float32)


if __name__ == "__main__":

    train_data = pd.read_csv("data_train.csv")
    test_data = pd.read_csv("smiles_test.csv")

    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    X_train = create_feature_matrix(train_data["smiles"])
    X_test = create_feature_matrix(test_data["smiles"])

    print("\nTrain features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)

    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)

    print("\nSaved feature matrices:")
    print("X_train.npy")
    print("X_test.npy")
