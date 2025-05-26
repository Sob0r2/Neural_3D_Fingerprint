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
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HDonors": Descriptors.NumHDonors(mol),
        "HAcceptors": Descriptors.NumHAcceptors(mol),
        "Polarizability": Descriptors.MolMR(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "PBF": CalcPBF(mol),
        "Asphericity": CalcAsphericity(mol),
        "Eccentricity": CalcEccentricity(mol),
        "SpherocityIndex": CalcSpherocityIndex(mol),
        "InertialShapeFactor": CalcInertialShapeFactor(mol),
        "WHIM_1": whim[0] if whim else 0,
        "MORSE_1": morse[0] if morse else 0,
        "GETAWAY_1": getaway[0] if getaway else 0,
        "RDF_1": rdf[0] if rdf else 0,
        "AUTOCORR3D_1": autocorr3d[0] if autocorr3d else 0,
        "HOMO": homo,
        "LUMO": lumo,
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
