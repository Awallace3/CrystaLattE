import crystalatte
from crystalatte.plugins import force_fields
import os
# ZoomLineSearch fixed by newest version not on PyPi:
# `pip install git+https://github.com/google/jaxopt`

file_dir = os.path.dirname(os.path.realpath(__file__))


def test_drude_energy():
    _, _, output_data = crystalatte.main(
        cif_input=file_dir + "/imidazole.cif",
        cif_output=file_dir + "/imidazole.xyz",
        bfs_thresh=1.2,
        cif_a=0,
        cif_b=0,
        cif_c=0,
        uniq_filter="ChSEV",
        nmers_up_to=3,
        r_cut_com=1000,
        r_cut_monomer=0,
        r_cut_dimer=5.0,
        r_cut_trimer=5.0,
        r_cut_tetramer=None,
        r_cut_pentamer=None,
        cle_run_type=["custom"],
        method="my_method",
        bsse_type=None,
        job_memory=None,
        verbose=2,
        custom_function=force_fields.polarization_energy_function,
        polarization_energy_type="jax_ind",
        platform_name="CPU",
        pdb_file=f"{file_dir}/imidazole.pdb",
        xml_file=f"{file_dir}/imidazole.xml",
        residue_file=f"{file_dir}/imidazole_residue.xml",
        atom_types_map=f"{file_dir}/imidazole_map.csv",
    )
    print(output_data)
    return


if __name__ == "__main__":
    test_drude_energy()
