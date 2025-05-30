import json

from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import (
    CalcAsphericity,
    CalcAUTOCORR3D,
    CalcEccentricity,
    CalcGETAWAY,
    CalcInertialShapeFactor,
    CalcMORSE,
    CalcPBF,
    CalcRDF,
    CalcSpherocityIndex,
    CalcWHIM,
)


def calculate_descriptors_v2(smiles, mol, homo, lumo, gt = True):
    try:
        autocorr3d = CalcAUTOCORR3D(mol)
        rdf = CalcRDF(mol)
        whim = CalcWHIM(mol)
        morse = CalcMORSE(mol)
        if gt:
            getaway = CalcGETAWAY(mol)
        else:
            getaway = None
    except:
        autocorr3d = rdf = whim = morse = getaway = []

    return {
        "SMILES": smiles,
        
        # --- Basic molecular properties ---
        "MolWt": Descriptors.MolWt(mol),  # Molecular weight
        "TPSA": Descriptors.TPSA(mol),  # Topological Polar Surface Area (correlates with drug absorption)
        "HDonors": Descriptors.NumHDonors(mol),  # Number of hydrogen bond donors
        "HAcceptors": Descriptors.NumHAcceptors(mol),  # Number of hydrogen bond acceptors
        "Polarizability": Descriptors.MolMR(mol),  # Molar refractivity (approximate indicator of polarizability)
        "RingCount": rdMolDescriptors.CalcNumRings(mol),  # Total number of rings in the molecule

        # --- 3D shape descriptors ---
        "PBF": CalcPBF(mol),  # Plane of Best Fit (measures flatness of molecule)
        "Asphericity": CalcAsphericity(mol),  # Deviation from spherical shape
        "Eccentricity": CalcEccentricity(mol),  # Measure of elongation of the shape
        "SpherocityIndex": CalcSpherocityIndex(mol),  # Another measure of sphericity (how spherical the molecule is)
        "InertialShapeFactor": CalcInertialShapeFactor(mol),  # Shape-related descriptor from inertia tensor

        # --- 3D geometric descriptor families (only first component shown here) ---
        "WHIM_1": whim[0] if whim else 0,  # WHIM: Weighted Holistic Invariant Molecular descriptor
        "MORSE_1": morse[0] if morse else 0,  # 3D-MoRSE: 3D Molecule Representation of Structures based on Electron diffraction
        "GETAWAY_1": getaway[0] if getaway else 0,  # GETAWAY: Geometry, Topology, and Atom-Weights Assembly
        "RDF_1": rdf[0] if rdf else 0,  # RDF: Radial Distribution Function descriptor
        "AUTOCORR3D_1": autocorr3d[0] if autocorr3d else 0,  # 3D Autocorrelation descriptor

        # --- QM-level properties (from external input) ---
        "HOMO": homo,  # Highest Occupied Molecular Orbital energy
        "LUMO": lumo,  # Lowest Unoccupied Molecular Orbital energy
    }


def calculate_descriptors(smiles, mol, homo, lumo):

    desc = {
        "SMILES": smiles,
        "MolWt": Descriptors.MolWt(mol),  # Masa molowa
        "LogP": Descriptors.MolLogP(mol),  # LogP (lipofilowość)
        "TPSA": Descriptors.TPSA(mol),  # Topologiczna powierzchnia polarowa
        "HDonors": Descriptors.NumHDonors(mol),  # Liczba donorów wiązań wodorowych
        "HAcceptors": Descriptors.NumHAcceptors(
            mol
        ),  # Liczba akceptorów wiązań wodorowych
        "RotatableBonds": Descriptors.NumRotatableBonds(
            mol
        ),  # Liczba rotowalnych wiązań
        "RingCount": rdMolDescriptors.CalcNumRings(mol),  # Liczba pierścieni
        "Kappa1": Descriptors.Kappa1(mol),  # Indeks kształtu Kappa 1
        "Kappa2": Descriptors.Kappa2(mol),  # Indeks kształtu Kappa 2
        "Kappa3": Descriptors.Kappa3(mol),  # Indeks kształtu Kappa 3
        "Polarizability": Descriptors.MolMR(mol),  # Polarowalność molowa
        "MaxEStateIndex": Descriptors.MaxEStateIndex(mol),  # Maksymalny indeks EState
        "MinEStateIndex": Descriptors.MinEStateIndex(mol),  # Minimalny indeks EState
        "CPSA1": (
            rdMolDescriptors.CalcAUTOCORR3D(mol)[0]
            if len(rdMolDescriptors.CalcAUTOCORR3D(mol)) > 0
            else None
        ),
        "RDF": (
            rdMolDescriptors.CalcRDF(mol)[0]
            if len(rdMolDescriptors.CalcRDF(mol)) > 0
            else None
        ),  # Deskryptor 3D RDF
        "HOMO": homo,  # Przybliżony wskaźnik QED jako zastępnik HOMO
        "LUMO": lumo,  # Przybliżony wskaźnik dla LUMO
        "Asphericity": rdMolDescriptors.CalcAsphericity(mol),  # Anizotropia kształtu
        "Eccentricity": rdMolDescriptors.CalcEccentricity(
            mol
        ),  # Ekscentryczność cząsteczki
    }

    return desc
