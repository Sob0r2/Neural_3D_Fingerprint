import json
import os
from typing import Dict, List, Tuple

from ase.units import Ang, Bohr, Hartree, eV
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Constants
PROP_NAMES = [
    "rcA",
    "rcB",
    "rcC",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "energy_U0",
    "energy_U",
    "enthalpy_H",
    "free_G",
    "Cv",
]

CONVERSIONS = [
    1.0,
    1.0,
    1.0,
    1.0,
    Bohr**3 / Ang**3,
    Hartree / eV,
    Hartree / eV,
    Hartree / eV,
    Bohr**2 / Ang**2,
    Hartree / eV,
    Hartree / eV,
    Hartree / eV,
    Hartree / eV,
    Hartree / eV,
    1.0,
]


def from_xyz_to_json(file_path: str) -> Dict:
    """
    Converts a QM9 .xyz file into a dictionary with molecular data.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        Dict: Parsed molecular data including atoms, SMILES string, and properties.
    """
    mol_dict = {}

    with open(file_path, "r") as f:
        data = f.readlines()

    # Extract SMILES string
    mol_dict["smiles"] = data[-2].split()[-1]

    # Parse and convert properties
    mol_properties = data[1].split()[1:16]
    for i, value in enumerate(mol_properties):
        mol_dict[PROP_NAMES[i]] = float(value.replace("^", "")) * CONVERSIONS[i]

    # Extract atom coordinates
    num_atoms = int(data[0])
    atoms = []
    for line in data[2 : 2 + num_atoms]:
        atom_data = line.split()
        atom = (
            atom_data[0],
            float(atom_data[1].replace("^", "")),
            float(atom_data[2].replace("^", "")),
            float(atom_data[3].replace("^", "")),
        )
        atoms.append(atom)

    mol_dict["atoms"] = atoms
    return mol_dict


def main():
    """
    Main script for converting QM9 .xyz files into JSON format.
    """
    input_folder = os.path.join(
        os.getenv("DATA_PATH"), "133660_curatedQM9_outof_133885"
    )
    output_folder = os.path.join(os.getenv("DATA_PATH"), "qm9_data_json")

    if not input_folder:
        raise ValueError("RAW_DATA_PATH not set in .env file.")

    os.makedirs(output_folder, exist_ok=True)

    file_list = sorted(os.listdir(input_folder))[137:]
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(input_folder, file_name)
        mol_dict = from_xyz_to_json(file_path)

        json_file_name = f"qm9_{str(i + 1).zfill(6)}.json"
        json_path = os.path.join(output_folder, json_file_name)

        with open(json_path, "w") as f:
            json.dump(mol_dict, f)

        if i % 1000 == 0:
            print(f"Processed {i} molecules...")


if __name__ == "__main__":
    main()
