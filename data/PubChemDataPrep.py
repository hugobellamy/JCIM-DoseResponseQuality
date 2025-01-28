import pandas as pd
import random
from PubChem_data import All_data_info
import DataTools as dt
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os


files = os.listdir("PubChem/raw")
files = [files[i] for i in range(len(files)) if "datatable" in files[i]]
AIDS_to_convert = [int(files[i].split("_")[1]) for i in range(len(files))]
RS = 32425
NOISE = 176
np.random.seed(RS)


def fingerprints_from_smiles(smiles, radius=2, nbits=1024):
    all_fingerprints = []
    all_indexes = []

    # Create the MorganGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    for molecule in smiles.index:
        mol_smile = smiles[molecule]
        mol = Chem.MolFromSmiles(mol_smile)
        # Generate fingerprint using the MorganGenerator
        fp = np.array(generator.GetFingerprint(mol))
        all_fingerprints.append(fp)
        all_indexes.append(molecule)

    col_name = [f"Bit_{i}" for i in range(nbits)]
    col_bits = [list(i) for i in all_fingerprints]
    fingerprints = pd.DataFrame(col_bits, columns=col_name, index=all_indexes)
    fingerprints = fingerprints.values.tolist()
    return fingerprints


def get_dose_response(val, keys, multiplyer=1):
    dose_response = []
    for key in keys:
        response = val[key]
        if np.isfinite(float(response)):
            dose_response.append([keys[key], multiplyer * float(response)])
    return dose_response


def get_numeric(val):
    z = val.split("u")
    return float(z[0])


def main():
    for AID in AIDS_to_convert:
        print("AID:", AID)
        datatable = pd.read_csv(
            f"PubChem/raw/AID_{AID}_datatable.csv", low_memory=False
        )
        data_info = All_data_info[f"AID_{AID}"]
        datatable.drop(data_info["rows"], inplace=True)
        for SID in data_info["SID_drop"]:
            datatable = datatable[datatable["PUBCHEM_SID"] != SID]
        datatable = datatable.iloc[
            : data_info["use"]
        ]  # some of our datatables have many single point dose responses at the end of the file
        X = fingerprints_from_smiles(datatable[data_info["smiles"]])
        PubChem_target_values = list(datatable[data_info["ec50s"]])
        input_numbers = [i for i in range(len(X))]
        new_order = np.random.RandomState(RS).permutation(input_numbers)
        PubChem_target_values = [PubChem_target_values[i] for i in new_order]
        with open(
            f"PUBCHEM/converted_data/AID_{AID}_PubChem_target_values.pkl", "wb"
        ) as f:
            pickle.dump(PubChem_target_values, f)
        datatable.drop(columns=data_info["cols"], inplace=True)
        if data_info["ec50s"] in list(
            datatable.columns
        ):  # we sometime have not added the ec50 col to the drop col
            datatable.drop(columns=data_info["ec50s"], inplace=True)
        if type(data_info["keys"]) is int:
            all_cols = datatable.columns
            keys = {
                all_cols[i + 1]: float(all_cols[i + 1].split(" ")[data_info["keys"]])
                for i in range(len(all_cols) - 1)
            }
        elif type(data_info["keys"]) is tuple:
            all_cols = datatable.columns
            character = data_info["keys"][0]
            split_value = data_info["keys"][1]
            keys = {
                all_cols[i + 1]: float(
                    get_numeric(all_cols[i + 1].split(character)[split_value])
                )
                for i in range(len(all_cols) - 1)
            }
        else:
            keys = data_info["keys"]
        dose_response = [
            get_dose_response(datatable.iloc[i], keys, data_info["multiplyer"])
            for i in range(len(datatable))
        ]
        data = dt.make_X_y(
            X, dose_response, dose_factor=10, random_seed=RS, base_noise=NOISE
        )
        with open(f"PUBCHEM/converted_data/{AID}.pkl", "wb") as f:
            pickle.dump(data, f)

        data_reduced = dt.make_X_y(
            X,
            dose_response,
            dose_factor=10,
            random_seed=RS,
            reduce=True,
            base_noise=NOISE,
        )
        with open(
            f"PUBCHEM/converted_data/{AID}000.pkl", "wb"
        ) as f:  # triple 0s will denote reduced data
            pickle.dump(data_reduced, f)


if __name__ == "__main__":
    main()
