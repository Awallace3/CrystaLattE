import crystalatte
from crystalatte.plugins import force_fields
import os
# ZoomLineSearch fixed by newest version not on PyPi:
# `pip install git+https://github.com/google/jaxopt`
# import warnings
#
# warnings.filterwarnings('ignore', '.*jaxopt.ZoomLineSearch.*')


file_dir = os.path.dirname(os.path.realpath(__file__)) + "/" + "imidazole/"


def main():
    # e = force_fields.openmm_inputs_polarization_energy(
    #     pdb_file=f"{file_dir}imidazole.pdb",
    #     xml_file=f"{file_dir}imidazole.xml",
    #     residue_file=f"{file_dir}imidazole_residue.xml",
    # )
    # print(e)
    # return
    _, _, output_data = crystalatte.main(
        cif_input=file_dir + "./imidazole.cif",
        cif_output=file_dir + "./imidazole.xyz",
        bfs_thresh=1.2,
        uniq_filter="ChSEV",
        nmers_up_to=2,
        r_cut_com=1000,
        r_cut_monomer=0,
        r_cut_dimer=30.0,
        r_cut_trimer=25.0,
        r_cut_tetramer=None,
        r_cut_pentamer=None,
        cle_run_type=["psithon"],
        method="my_method",
        bsse_type=None,
        job_memory=None,
        verbose=2,
        # custom_function=force_fields.example_energy_function,
        custom_function=force_fields.polarization_energy_function,
        pdb_file=f"{file_dir}imidazole.pdb",
        xml_file=f"{file_dir}imidazole.xml",
        residue_file=f"{file_dir}imidazole_residue.xml",
        # atom_types=monomer_atom_types_in_order_of_xyz,
        atom_types_map=f"{file_dir}imidazole_map.csv",
    )
    return
    _, _, output_data = crystalatte.main(
        cif_input=file_dir + "./imidazole.cif",
        cif_output=file_dir + "./imidazole.xyz",
        bfs_thresh=1.2,
        uniq_filter="ChSEV",
        nmers_up_to=2,
        r_cut_com=1000,
        r_cut_monomer=0,
        r_cut_dimer=30.0,
        r_cut_trimer=25.0,
        r_cut_tetramer=None,
        r_cut_pentamer=None,
        cle_run_type=["custom"],
        method="my_method",
        bsse_type=None,
        job_memory=None,
        verbose=2,
        # custom_function=force_fields.example_energy_function,
        custom_function=force_fields.polarization_energy_function,
        pdb_file=f"{file_dir}imidazole.pdb",
        xml_file=f"{file_dir}imidazole.xml",
        residue_file=f"{file_dir}imidazole_residue.xml",
        # atom_types=monomer_atom_types_in_order_of_xyz,
        atom_types_map=f"{file_dir}imidazole_map.csv",
    )
    try:
        import pandas as pd

        df = pd.DataFrame(output_data)
        print(df)
        df.to_csv("./ammonia_results.csv", index=False)
    except ImportError:
        print("Pandas not installed, printing dictionary")
        print(output_data)
    return


if __name__ == "__main__":
    main()
