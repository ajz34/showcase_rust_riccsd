use showcase_rust_riccsd::*;

fn main() {
    // Following directory is defined by the user.
    // This program is only for demonstration purpose, so we do not provide CLI arguments.
    // Data are provided by jupyter notebooks in folder `python_scripts`.
    println!("This program reads *.npy from first argument, otherwise assumes `./`");
    println!("You may provide mo_coeff.npy, mo_occ.npy, mo_energy.npy, cderi.npy (in c-contiguous)");
    println!("cderi is in AO basis, with s2ij symmetry");
    let args = std::env::args().collect::<Vec<String>>();
    let mol_dir_root = if args.len() == 1 { "./" } else { args[1].as_str() };
    println!("Directory root of *.npy files: {}", mol_dir_root);
    println!("======");

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
