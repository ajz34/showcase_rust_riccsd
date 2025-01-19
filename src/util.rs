use crate::*;
use rstsr_core::prelude::*;

pub fn get_vec_from_npy<T>(path: String) -> Vec<T>
where
    T: npyz::Deserialize,
{
    let bytes = std::fs::read(path).unwrap();
    let vec = npyz::NpyFile::new(&bytes[..])
        .unwrap()
        .into_vec::<T>()
        .unwrap();
    return vec;
}

pub fn read_mol_info(path: String) -> MolInfo {
    let device = DeviceOpenBLAS::default();

    // mo_occ
    let mo_occ_vec = get_vec_from_npy::<f64>(path.to_string() + "mo_occ.npy");
    let mo_occ = rt::asarray((mo_occ_vec, &device));
    let nmo = mo_occ.size();
    let nocc = mo_occ.mapv(|x| if x > 0.0 { 1 } else { 0 }).sum_all();
    let nvir = nmo - nocc;

    // mo_energy
    let mo_energy_vec = get_vec_from_npy::<f64>(path.to_string() + "mo_energy.npy");
    let mo_energy = rt::asarray((mo_energy_vec, &device));

    // mo_coeff
    let mo_coeff_vec = get_vec_from_npy::<f64>(path.to_string() + "mo_coeff.npy");
    let mo_coeff = rt::asarray((mo_coeff_vec, &device)).into_shape((-1, nmo));
    let nao = mo_coeff.shape()[0];

    return MolInfo {
        mo_energy,
        mo_coeff,
        mo_occ,
        nao,
        nmo,
        nocc,
        nvir,
        naux: 0,
        cderi: None,
    };
}

pub fn read_mol_cderi(mol_info: &mut MolInfo, path: String) {
    let device = DeviceOpenBLAS::default();
    let nao = mol_info.nao;
    let nao_tp = (nao * (nao + 1)) / 2;

    let cderi_vec = get_vec_from_npy::<f64>(path.to_string() + "cderi.npy");
    let len_cderi = cderi_vec.len();
    if len_cderi % nao_tp != 0 {
        panic!("seems cderi dimension is not correct.");
    }

    let naux = len_cderi / nao_tp;
    let cderi = rt::asarray((cderi_vec, &device)).into_shape((naux, nao_tp));
    mol_info.naux = naux;
    mol_info.cderi = Some(cderi);
}

pub fn read_amplitude(path: String, mol_info: &MolInfo) -> CCSDInfo {
    let device = DeviceOpenBLAS::default();

    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let t1_vec = get_vec_from_npy::<f64>(path.to_string() + "t1.npy");
    let t1 = rt::asarray((t1_vec, [nocc, nvir], &device));

    let t2_vec = get_vec_from_npy::<f64>(path.to_string() + "t2.npy");
    let t2 = rt::asarray((t2_vec, [nocc, nocc, nvir, nvir], &device));

    let e_corr = 0.0;
    return CCSDInfo { e_corr, t1, t2 };
}

/*

pub fn read_mol_cint_data(path: String) -> CINTR2CDATA {
    // read raw data from npy file
    let atm = get_vec_from_npy::<i32>(path.to_string() + "mol_atm.npy").unwrap();
    let bas = get_vec_from_npy::<i32>(path.to_string() + "mol_bas.npy").unwrap();
    let env = get_vec_from_npy::<f64>(path.to_string() + "mol_env.npy").unwrap();

    // fold atm to (natm, 6)
    let c_atm: Vec<Vec<i32>> = atm.chunks(6).map(|x| x.to_vec()).collect();
    let natm = c_atm.len() as i32;

    // fold bas to (nbas, 8)
    let c_bas: Vec<Vec<i32>> = bas.chunks(8).map(|x| x.to_vec()).collect();
    let nbas = c_bas.len() as i32;

    // convert raw data to CINTR2CDATA

    let mut cint_data = CINTR2CDATA::new();
    cint_data.initial_r2c(&c_atm, natm, &c_bas, nbas, &env);
    return cint_data;
}

    */

#[test]
fn test() {
    let mol_dir_root = "/home/a/Documents-Group-Xu/2025-01-17-rust_ccsd/python/h2o-cc-pvdz/";
    let mut mol_info = read_mol_info(mol_dir_root.to_string());
    println!("{:10.3}", mol_info.mo_energy);
    println!("{:10.3}", mol_info.mo_coeff);
    println!("{:10.3}", mol_info.mo_occ);
    println!("{:}", mol_info.nao);
    println!("{:}", mol_info.nmo);
    println!("{:}", mol_info.nocc);
    println!("{:}", mol_info.nvir);

    read_mol_cderi(&mut mol_info, mol_dir_root.to_string());
    println!("{:}", mol_info.naux);
    println!("{:10.3}", mol_info.cderi.as_ref().unwrap());

    let ccsd_info = read_amplitude(mol_dir_root.to_string(), &mol_info);
    println!("{:10.3}", ccsd_info.e_corr);
    println!("{:10.3}", ccsd_info.t1);
    println!("{:10.3}", ccsd_info.t2);
}
