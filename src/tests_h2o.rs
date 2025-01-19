use crate::*;
use rstsr_core::prelude::*;

#[test]
fn test_h2o() {
    let mol_dir_root = "/home/a/Documents-Group-Xu/2025-01-17-rust_ccsd/python/h2o-cc-pvdz/";
    let mut mol_info = read_mol_info(mol_dir_root.to_string());
    read_mol_cderi(&mut mol_info, mol_dir_root.to_string());

    let B = get_cderi_mo(&mol_info).unwrap();
    println!("{:10.3}", mol_info.cderi.as_ref().unwrap().sin().sum_all());
    println!("B summation       {:10.3}", (&B).sin().sum_all());

    let mut ccsd_intermediates = CCSDIntermediates::new_empty();
    ccsd_intermediates.cderi = Some(B);

    let ccsd_info_ref = read_amplitude(mol_dir_root.to_string(), &mol_info);
    let t1 = &ccsd_info_ref.t1;
    let t2 = &ccsd_info_ref.t2;

    get_riccsd_intermediates_1(&mol_info, &mut ccsd_intermediates, t1, t2);
    get_riccsd_intermediates_2(&mol_info, &mut ccsd_intermediates, t1, t2);
}
