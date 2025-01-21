use showcase_rust_riccsd::*;

fn main() {
    // Following directory is defined by the user.
    // This program is only for demonstration purpose, so we do not provide CLI arguments.
    // Data are provided by jupyter notebooks in folder `python_scripts`.
    let mol_dir_root = "/home/a/Documents-Group-Xu/2025-01-17-rust_ccsd/showcase_rust_riccsd/python_scripts/h2o_pp5-cc-pvdz/";
    let mut mol_info = read_mol_info(mol_dir_root.to_string());
    read_mol_cderi(&mut mol_info, mol_dir_root.to_string());

    let cc_config = CCSDConfig {
        max_cycle: 100,
        conv_tol_e: 1.0e-7,
        conv_tol_t1: 1.0e-5,
        conv_tol_t2: 1.0e-5,
    };

    let ccsd_info = naive_riccsd_iteration(&mol_info, &cc_config);
    println!("Final CCSD Corr Energy {:18.10}", ccsd_info.e_corr);
}
