use crate::*;
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::DeviceRayonAPI;

#[test]
fn test_h2o_parts() {
    let mol_dir_root = "/home/a/Documents-Group-Xu/2025-01-17-rust_ccsd/python/h2o-cc-pvdz/";
    let mut mol_info = read_mol_info(mol_dir_root.to_string());
    read_mol_cderi(&mut mol_info, mol_dir_root.to_string());
    let ccsd_info_ref = read_amplitude(mol_dir_root.to_string(), &mol_info);

    let B = get_cderi_mo(&mol_info);
    let mut intermediates = CCSDIntermediates::new_empty();
    intermediates.cderi = Some(B);

    println!("{:10.3}", intermediates.cderi.as_ref().unwrap().device().get_num_threads());

    let t1 = &ccsd_info_ref.t1;
    let t2 = &ccsd_info_ref.t2;

    get_riccsd_intermediates_1(&mol_info, &mut intermediates, t1, t2);
    get_riccsd_intermediates_2(&mol_info, &mut intermediates, t1);
    let e_corr = get_riccsd_energy(&mol_info, &intermediates);
    println!("{:18.10}", e_corr);

    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;
    let device = DeviceOpenBLAS::default();
    let mut rhs1: Tsr = rt::zeros(([nocc, nvir], &device));
    get_riccsd_rhs1(&mol_info, rhs1.view_mut(), &intermediates, t1, t2);
    // println!("{:16.10}", rhs1);

    let mut rhs2 = rt::zeros(([nocc, nocc, nvir, nvir], &device));
    get_riccsd_rhs2_lt2_contract(&mol_info, rhs2.view_mut(), &intermediates, t2);
    let rhs2_ref = get_vec_from_npy::<f64>(mol_dir_root.to_string() + "RHS2-lt2_contract.npy");
    let rhs2_ref = rt::asarray((rhs2_ref, [nocc, nocc, nvir, nvir], &device));
    println!("{:16.10}", (rhs2 - rhs2_ref).abs().sum_all());

    let mut rhs2 = rt::zeros(([nocc, nocc, nvir, nvir], &device));
    get_riccsd_rhs2_direct_dot(&mol_info, rhs2.view_mut(), &intermediates);
    let rhs2_ref = get_vec_from_npy::<f64>(mol_dir_root.to_string() + "RHS2-direct_dot.npy");
    let rhs2_ref = rt::asarray((rhs2_ref, [nocc, nocc, nvir, nvir], &device));
    println!("{:16.10}", (rhs2 - rhs2_ref).abs().sum_all());

    let mut rhs2 = rt::zeros(([nocc, nocc, nvir, nvir], &device));
    get_riccsd_rhs2_o3v3(&mol_info, rhs2.view_mut(), &intermediates, t2);
    let rhs2_ref = get_vec_from_npy::<f64>(mol_dir_root.to_string() + "RHS2-o3v3.npy");
    let rhs2_ref = rt::asarray((rhs2_ref, [nocc, nocc, nvir, nvir], &device));
    println!("{:16.10}", (rhs2 - rhs2_ref).abs().sum_all());

    let mut rhs2 = rt::zeros(([nocc, nocc, nvir, nvir], &device));
    get_riccsd_rhs2_o4v2(&mol_info, rhs2.view_mut(), &intermediates, t1, t2);
    let rhs2_ref = get_vec_from_npy::<f64>(mol_dir_root.to_string() + "RHS2-o4v2.npy");
    let rhs2_ref = rt::asarray((rhs2_ref, [nocc, nocc, nvir, nvir], &device));
    println!("{:16.10}", (rhs2 - rhs2_ref).abs().sum_all());

    let mut rhs2 = rt::zeros(([nocc, nocc, nvir, nvir], &device));
    get_riccsd_rhs2_o2v4(&mol_info, rhs2.view_mut(), &intermediates, t1, t2);
    let rhs2_ref = get_vec_from_npy::<f64>(mol_dir_root.to_string() + "RHS2-o2v4.npy");
    let rhs2_ref = rt::asarray((rhs2_ref, [nocc, nocc, nvir, nvir], &device));
    println!("{:16.10}", (&rhs2 - &rhs2_ref).abs().sum_all());

    // let rhs2 = rhs2.view() + rhs2.view().transpose((1, 0, 3, 2));
    // println!("{:16.10}", rhs2.sin().sum_all());
}

#[test]
fn test_h2o_one_iter() {
    let mol_dir_root = "/home/a/Documents-Group-Xu/2025-01-17-rust_ccsd/python/h2o-cc-pvdz/";
    let mut mol_info = read_mol_info(mol_dir_root.to_string());
    read_mol_cderi(&mut mol_info, mol_dir_root.to_string());
    let ccsd_info_ref = read_amplitude(mol_dir_root.to_string(), &mol_info);

    let B = get_cderi_mo(&mol_info);
    let mut intermediates = CCSDIntermediates::new_empty();
    intermediates.cderi = Some(B);

    let t1 = &ccsd_info_ref.t1;
    let t2 = &ccsd_info_ref.t2;

    get_riccsd_intermediates_1(&mol_info, &mut intermediates, &t1, &t2);
    let ccsd_info = update_riccsd_amplitude(&mol_info, &mut intermediates, &ccsd_info_ref);
    println!("{:18.10}", (&ccsd_info.t1 - &ccsd_info_ref.t1).abs().sum_all());
    println!("{:18.10}", (&ccsd_info.t2 - &ccsd_info_ref.t2).abs().sum_all());
    println!("{:18.10}", ccsd_info.e_corr);
}

#[test]
fn test_h2o_full_iter() {
    let mol_dir_root = "/home/a/Documents-Group-Xu/2025-01-17-rust_ccsd/python/h2o-cc-pvdz/";
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
