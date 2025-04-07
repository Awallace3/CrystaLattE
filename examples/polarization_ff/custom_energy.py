import crystalatte

# from crystalatte.plugins import force_fields
import os
# ZoomLineSearch fixed by newest version not on PyPi:
# `pip install git+https://github.com/google/jaxopt`
# import warnings
#
# warnings.filterwarnings('ignore', '.*jaxopt.ZoomLineSearch.*')


file_dir = os.path.dirname(os.path.realpath(__file__)) + "/" + "imidazole/"


def example_energy_function(
    qcel_mol,
    cif_output: str,
    nmers: dict,
    keynmer: str,
    nmer: dict,
    rminseps: str,
    rcomseps: str,
    cle_run_type: list,
    method="method_name_if_applicable",
    bsse_type=None,
    job_memory=None,
    verbose=0,
    **kwargs,
):
    """
    Every crystalatte energy function plugin must accept the above arguments.

    Takes the `nmers` dictionary; `keynmer`, the key of a given N-mer of
    the N-mers dictionary;

    Results are stored in the `nmer` dictionary under the key `nambe` standing
    for non-additive many-body energy.

    kwargs passed to crystalatte.main() are passed to the energy function
    allowing the user to specify any additional arguments.
    """
    example_arg = kwargs.get("example_extra_arg", 0.0)
    # print(f"Example extra argument: {example_arg}")
    print(keynmer, qcel_mol)
    print(qcel_mol.atomic_numbers)
    print(qcel_mol.geometry)
    n_body_energy = -0.0105
    if len(nmer["monomers"]) > 2:
        n_minus_1_body_energy = -0.0005
        nmer["nambe"] = n_body_energy - n_minus_1_body_energy

    else:
        nmer["nambe"] = n_body_energy
    return


def main():
    # e = force_fields.openmm_inputs_polarization_energy(
    #     pdb_file=f"{file_dir}imidazole.pdb",
    #     xml_file=f"{file_dir}imidazole.xml",
    #     residue_file=f"{file_dir}imidazole_residue.xml",
    # )
    # print(e)
    # return
    # cif_a           = 0
    # cif_b           = 0
    # cif_c           = 0
    # nmers_up_to     = 2
    # r_cut_com       = 1000
    # r_cut_monomer   = 0
    # r_cut_dimer     = 15
    _, _, output_data = crystalatte.main(
        cif_input=file_dir + "./imidazole.cif",
        cif_output=file_dir + "./imidazole.xyz",
        bfs_thresh=1.2,
        cif_a=0,
        cif_b=0,
        cif_c=0,
        uniq_filter="ChSEV",
        nmers_up_to=2,
        r_cut_com=1000,
        r_cut_monomer=0,
        r_cut_dimer=30,
        r_cut_trimer=30,
        r_cut_tetramer=None,
        r_cut_pentamer=None,
        cle_run_type=["custom"],
        method="my_method",
        bsse_type=None,
        job_memory=None,
        verbose=2,
        # custom_function=force_fields.example_energy_function,
        # custom_function=force_fields.polarization_energy_function,
        custom_function=example_energy_function,
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
