use showcase_rust_riccsd::*;

#[test]
fn test_h2o() {
    let mol_dir_root = "/home/a/Documents-Group-Xu/2025-01-17-rust_ccsd/python/h2o-cc-pvdz/";
    let mut mol_info = read_mol_info(mol_dir_root.to_string());
    read_mol_cderi(&mut mol_info, mol_dir_root.to_string());
    // let ccsd_info_ref = read_amplitude(mol_dir_root.to_string(), &mol_info);
}
