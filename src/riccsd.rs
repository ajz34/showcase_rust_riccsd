use crate::*;
use rayon::prelude::*;
use rstsr_core::prelude::*;
use std::error::Error;

pub fn get_cderi_mo(mol_info: &MolInfo) -> Result<Tsr, Box<dyn Error>> {
    let naux = mol_info.naux;
    let nao = mol_info.nao;
    let nmo = mol_info.nmo;
    let mo_coeff = &mol_info.mo_coeff;
    let cderi_ao = mol_info.cderi.as_ref().unwrap();

    let cderi_puv = cderi_ao.unpack_tril(TensorSymm::Sy);
    let cderi_spu = mo_coeff.t() % cderi_puv.reshape((-1, nao)).t();
    let cderi_rsp = mo_coeff.t() % cderi_spu.reshape((-1, nao)).t();
    let B = cderi_rsp.into_shape([nmo, nmo, naux]);
    Ok(B)
}

pub fn get_riccsd_intermediates_1(
    mol_info: &MolInfo,
    intermediates: &mut CCSDIntermediates,
    t1: &Tsr,
    t2: &Tsr,
) {
    let naux = mol_info.naux;
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let B = intermediates.cderi.as_ref().unwrap();
    let device = B.device().clone();

    // M1j = np.einsum("jbP, jb -> P", B[so, sv], t1)
    let m1_j = t1.reshape(-1) % B.i((so, sv)).reshape((-1, naux));

    // M1oo = np.einsum("jaP, ia -> ijP", B[so, sv], t1)
    let m1_oo: Tsr = rt::zeros(([nocc, nocc, naux], &device));
    (0..nocc).into_par_iter().for_each(|j| {
        let mut m1_oo = unsafe { m1_oo.force_mut() };
        *&mut m1_oo.i_mut((.., j)) += t1 % &B.i((j, sv));
    });

    // M2a = np.einsum("jbP, ijab -> iaP", B[so, sv], (2 * t2 - t2.swapaxes(-1, -2)))
    let m2a_ov: Tsr = rt::zeros(([nocc, nvir, naux], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        let mut m2a_ov = unsafe { m2a_ov.force_mut() };
        let scr_jba: Tsr = -t2.i(i) + 2 * t2.i(i).swapaxes(-1, -2);
        *&mut m2a_ov.i_mut(i) +=
            scr_jba.reshape((-1, nvir)).t() % B.i((so, sv)).reshape((-1, naux));
    });

    // println!("m1_j   {:}", (&m1_j).sin().sum_all());
    // println!("m1_oo  {:}", (&m1_oo).sin().sum_all());
    // println!("m2a_ov {:}", (&m2a_ov).sin().sum_all());

    intermediates.m1_j = Some(m1_j);
    intermediates.m1_oo = Some(m1_oo);
    intermediates.m2a_ov = Some(m2a_ov);
}

pub fn get_riccsd_intermediates_2(
    mol_info: &MolInfo,
    intermediates: &mut CCSDIntermediates,
    t1: &Tsr,
    _t2: &Tsr,
) {
    let naux = mol_info.naux;
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let B = intermediates.cderi.as_ref().unwrap();
    let device = B.device().clone();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();

    // M1aov = np.einsum("ijP, ja -> iaP", B[so, so], t1)
    let m1a_ov = rt::zeros(([nocc, nvir, naux], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        let mut m1a_ov = unsafe { m1a_ov.force_mut() };
        *&mut m1a_ov.i_mut(i) += t1.t() % &B.i((i, so));
    });

    // M1bov  = np.einsum("abP, ib -> iaP", B[sv, sv], t1)
    let m1b_ov = t1 % B.i((sv, sv)).reshape((nvir, -1));
    let m1b_ov = m1b_ov.into_shape((nocc, nvir, naux));

    // M1vv = np.einsum("ibP, ia -> abP", B[so, sv], t1)
    let m1_vv = t1.t() % B.i((so, sv)).reshape((nocc, -1));
    let m1_vv = m1_vv.into_shape((nvir, nvir, naux));

    // M2b = np.einsum("ikP, ka -> iaP", M1oo, t1)
    let m2b_ov = rt::zeros(([nocc, nvir, naux], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        let mut m2b_ov = unsafe { m2b_ov.force_mut() };
        *&mut m2b_ov.i_mut(i) += t1.t() % m1_oo.i(i);
    });

    // println!("m1a_ov {:}", (&m1a_ov).sin().sum_all());
    // println!("m1b_ov {:}", (&m1b_ov).sin().sum_all());
    // println!("m1_vv  {:}", (&m1_vv).sin().sum_all());
    // println!("m2b_ov {:}", (&m2b_ov).sin().sum_all());

    intermediates.m1a_ov = Some(m1a_ov);
    intermediates.m1b_ov = Some(m1b_ov);
    intermediates.m1_vv = Some(m1_vv);
    intermediates.m2b_ov = Some(m2b_ov);
}
