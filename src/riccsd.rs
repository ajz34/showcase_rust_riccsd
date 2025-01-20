use crate::*;
use rayon::prelude::*;
use rstsr_core::prelude::*;

pub fn get_cderi_mo(mol_info: &MolInfo) -> Tsr {
    let naux = mol_info.naux;
    let nao = mol_info.nao;
    let nmo = mol_info.nmo;
    let mo_coeff = &mol_info.mo_coeff;
    let cderi_ao = mol_info.cderi.as_ref().unwrap();

    let cderi_puv = cderi_ao.unpack_tril(TensorSymm::Sy);
    let cderi_spu = mo_coeff.t() % cderi_puv.reshape((-1, nao)).t();
    let cderi_rsp = mo_coeff.t() % cderi_spu.reshape((-1, nao)).t();
    let cderi_rsp = cderi_rsp.into_shape([nmo, nmo, naux]);
    return cderi_rsp;
}

pub fn get_riccsd_intermediates_1(mol_info: &MolInfo, intermediates: &mut CCSDIntermediates, t1: &Tsr, t2: &Tsr) {
    let naux = mol_info.naux;
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let b_ov = intermediates.cderi.as_ref().unwrap().i((so, sv));
    let device = b_ov.device().clone();

    // M1j = np.einsum("jbP, jb -> P", B[so, sv], t1)
    let m1_j = t1.reshape(-1) % b_ov.reshape((-1, naux));

    // M1oo = np.einsum("jaP, ia -> ijP", B[so, sv], t1)
    let m1_oo: Tsr = rt::zeros(([nocc, nocc, naux], &device));
    (0..nocc).into_par_iter().for_each(|j| {
        let mut m1_oo = unsafe { m1_oo.force_mut() };
        *&mut m1_oo.i_mut((.., j)) += t1 % &b_ov.i(j);
    });

    // M2a = np.einsum("jbP, ijab -> iaP", B[so, sv], (2 * t2 - t2.swapaxes(-1, -2)))
    let m2a_ov: Tsr = rt::zeros(([nocc, nvir, naux], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        let mut m2a_ov = unsafe { m2a_ov.force_mut() };
        let scr_jba: Tsr = -t2.i(i) + 2 * t2.i(i).swapaxes(-1, -2);
        *&mut m2a_ov.i_mut(i) += scr_jba.reshape((-1, nvir)).t() % b_ov.reshape((-1, naux));
    });

    intermediates.m1_j = Some(m1_j);
    intermediates.m1_oo = Some(m1_oo);
    intermediates.m2a_ov = Some(m2a_ov);
}

pub fn get_riccsd_intermediates_2(mol_info: &MolInfo, intermediates: &mut CCSDIntermediates, t1: &Tsr) {
    let naux = mol_info.naux;
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let b_oo = intermediates.cderi.as_ref().unwrap().i((so, so));
    let b_ov = intermediates.cderi.as_ref().unwrap().i((so, sv));
    let b_vv = intermediates.cderi.as_ref().unwrap().i((sv, sv));
    let device = b_oo.device().clone();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();

    // M1aov = np.einsum("ijP, ja -> iaP", B[so, so], t1)
    let m1a_ov = rt::zeros(([nocc, nvir, naux], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        let mut m1a_ov = unsafe { m1a_ov.force_mut() };
        *&mut m1a_ov.i_mut(i) += t1.t() % &b_oo.i(i);
    });

    // M1bov  = np.einsum("abP, ib -> iaP", B[sv, sv], t1)
    let m1b_ov = t1 % b_vv.reshape((nvir, -1));
    let m1b_ov = m1b_ov.into_shape((nocc, nvir, naux));

    // M1vv = np.einsum("ibP, ia -> abP", B[so, sv], t1)
    let m1_vv = t1.t() % b_ov.reshape((nocc, -1));
    let m1_vv = m1_vv.into_shape((nvir, nvir, naux));

    // M2b = np.einsum("ikP, ka -> iaP", M1oo, t1)
    let m2b_ov = rt::zeros(([nocc, nvir, naux], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        let mut m2b_ov = unsafe { m2b_ov.force_mut() };
        *&mut m2b_ov.i_mut(i) += t1.t() % m1_oo.i(i);
    });

    intermediates.m1a_ov = Some(m1a_ov);
    intermediates.m1b_ov = Some(m1b_ov);
    intermediates.m1_vv = Some(m1_vv);
    intermediates.m2b_ov = Some(m2b_ov);
}

pub fn get_riccsd_energy(mol_info: &MolInfo, intermediates: &CCSDIntermediates) -> f64 {
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let b_ov = intermediates.cderi.as_ref().unwrap().i((so, sv));
    let m1_j = intermediates.m1_j.as_ref().unwrap();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();
    let m2a_ov = intermediates.m2a_ov.as_ref().unwrap();

    let e_t1_j = 2.0 * (m1_j.reshape(-1) % m1_j.reshape(-1));
    let e_t1_k = -(m1_oo.reshape(-1) % m1_oo.swapaxes(0, 1).reshape(-1));
    let e_t2 = b_ov.reshape(-1) % m2a_ov.reshape(-1);
    let e_corr: Tsr = e_t1_j + e_t1_k + e_t2;
    let e_corr = e_corr.to_scalar();
    return e_corr;
}

pub fn get_riccsd_rhs1(mol_info: &MolInfo, mut rhs1: TsrMut, intermediates: &CCSDIntermediates, t1: &Tsr, t2: &Tsr) {
    let naux = mol_info.naux;
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let b_oo = intermediates.cderi.as_ref().unwrap().i((so, so));
    let b_ov = intermediates.cderi.as_ref().unwrap().i((so, sv));
    let b_vv = intermediates.cderi.as_ref().unwrap().i((sv, sv));
    let m1_j = intermediates.m1_j.as_ref().unwrap();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();
    let m1a_ov = intermediates.m1a_ov.as_ref().unwrap();
    let m1b_ov = intermediates.m1b_ov.as_ref().unwrap();
    let m2a_ov = intermediates.m2a_ov.as_ref().unwrap();
    let m2b_ov = intermediates.m2b_ov.as_ref().unwrap();

    let device = b_oo.device().clone();

    // === TERM 1 === //
    // RHS1 += - 1 * np.einsum("lcP, lkP, ikac -> ia", B[so, sv], M1oo, (2 * t2 - t2.swapaxes(-1, -2)))

    // "lcP, lkP -> kc", B[so, sv], M1oo
    let mut scr_kc: Tsr = rt::zeros(([nocc, nvir], &device));
    for l in 0..nocc {
        scr_kc += m1_oo.i(l) % b_ov.i(l).t();
    }

    // "kc, ikac -> ia", scr_kc, (2 * t2 - t2.swapaxes(-1, -2)))
    (0..nocc).into_par_iter().for_each(|i| {
        let mut rhs1 = unsafe { rhs1.force_mut() };
        let mut t2_i: Tsr = 2.0 * t2.i(i) - t2.i(i).swapaxes(-1, -2);
        t2_i *= scr_kc.i((.., None, ..));
        *&mut rhs1.i_mut(i) -= t2_i.sum([0, 2]);
    });

    // // === TERM 2 === //
    // RHS1 += - 1 * np.einsum("kcP, icP, ka -> ia", B[so, sv], (M2a - M1aov), t1)

    rhs1 -= (m2a_ov - m1a_ov).reshape((nocc, -1)) % b_ov.reshape((nocc, -1)).t() % t1;

    // === TERM 3 === //
    // RHS1 +=   1 * np.einsum("icP, acP -> ia", M2a, B[sv, sv])

    rhs1 += m2a_ov.reshape((nocc, -1)) % b_vv.reshape((nvir, -1)).t();

    // === TERM 4 === //
    // RHS1 += - 1 * np.einsum("ikP, kaP -> ia", (B[so, so] + M1oo), (M2a + M1bov))

    for k in 0..nocc {
        rhs1 -= (b_oo.i(k) + m1_oo.i((.., k))) % (m2a_ov.i(k) + m1b_ov.i(k)).t();
    }

    // === TERM 5 === //
    // RHS1 +=   2 * np.einsum("iaP, P -> ia", (B[so, sv] + M1bov - M1aov + M2a - 0.5 * M2b), M1j)

    let scr_iaP: Tsr = b_ov + m1b_ov - m1a_ov + m2a_ov - 0.5 * m2b_ov;
    rhs1 += 2.0 * (scr_iaP.reshape((-1, naux)) % m1_j).into_shape((nocc, nvir));
    // AJZ NOTE: it seems that cow tensor could not be multiplied by value, and that can be inconvenient.
    //           So using into_shape here.
}

pub fn get_riccsd_rhs2_lt2_contract(mol_info: &MolInfo, mut rhs2: TsrMut, intermediates: &CCSDIntermediates, t2: &Tsr) {
    let naux = mol_info.naux;
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let b_oo = intermediates.cderi.as_ref().unwrap().i((so, so));
    let b_ov = intermediates.cderi.as_ref().unwrap().i((so, sv));
    let b_vv = intermediates.cderi.as_ref().unwrap().i((sv, sv));
    let m1_j = intermediates.m1_j.as_ref().unwrap();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();
    let m1_vv = intermediates.m1_vv.as_ref().unwrap();
    let m1b_ov = intermediates.m1b_ov.as_ref().unwrap();
    let m2a_ov = intermediates.m2a_ov.as_ref().unwrap();

    // === TERM 1 === //
    // Loo = (
    //     + np.einsum("kcP, icP -> ik", B[so, sv], M2a)
    //     - np.einsum("liP, lkP -> ik", B[so, so], M1oo)
    //     + np.einsum("ikP, P -> ik", 2 * B[so, so] + M1oo, M1j))
    // RHS2 -= np.einsum("ik, kjab -> ijab", Loo, t2)

    let mut l_oo = m2a_ov.reshape((nocc, -1)) % b_ov.reshape((nocc, -1)).t();
    let scr: Tsr = 2 * &b_oo + m1_oo;
    l_oo += (scr.reshape((-1, naux)) % m1_j).into_shape((nocc, nocc));
    for l in 0..nocc {
        l_oo -= b_oo.i(l) % m1_oo.i(l).t();
    }

    rhs2 -= (l_oo % t2.reshape((nocc, -1))).into_shape((nocc, nocc, nvir, nvir));

    // Lvv = (
    //     - 1 * np.einsum("kcP, kaP -> ac", B[so, sv], M2a + M1bov)
    //     + 1 * np.einsum("acP, P -> ac", 2 * B[sv, sv] - M1vv, M1j))
    // RHS2 += np.einsum("ac, ijcb -> ijab", Lvv, t2)

    let scr: Tsr = 2 * b_vv - m1_vv;
    let mut l_vv = (scr.reshape((-1, naux)) % m1_j).into_shape((nvir, nvir));
    for k in 0..nocc {
        l_vv -= (m2a_ov + m1b_ov).i(k) % b_ov.i(k).t();
    }

    rhs2 += (t2.reshape((-1, nvir)) % l_vv.t()).into_shape((nocc, nocc, nvir, nvir));
}

pub fn get_riccsd_rhs2_direct_dot(mol_info: &MolInfo, rhs2: TsrMut, intermediates: &CCSDIntermediates) {
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let B = intermediates.cderi.as_ref().unwrap();
    let m1a_ov = intermediates.m1a_ov.as_ref().unwrap();
    let m1b_ov = intermediates.m1b_ov.as_ref().unwrap();
    let m2a_ov = intermediates.m2a_ov.as_ref().unwrap();
    let m2b_ov = intermediates.m2b_ov.as_ref().unwrap();

    // RHS2 += np.einsum("iaP, jbP -> ijab", B[so, sv] + M2a, 0.5 * B[so, sv] + 0.5 * M2a - M1aov + M1bov - M2b)
    // RHS2 -= np.einsum("iaP, jbP -> ijab", M1aov, M1bov)

    let scr_iaP: Tsr = B.i((so, sv)) + m2a_ov;
    let scr_jbP: Tsr = 0.5 * B.i((so, sv)) + 0.5 * m2a_ov - m1a_ov + m1b_ov - m2b_ov;

    (0..nocc).into_par_iter().for_each(|i| {
        (0..nocc).into_par_iter().for_each(|j| {
            let mut rhs2 = unsafe { rhs2.force_mut() };
            *&mut rhs2.i_mut((i, j)) += scr_iaP.i(i) % scr_jbP.i(j).t();
            *&mut rhs2.i_mut((i, j)) -= m1a_ov.i(i) % m1b_ov.i(j).t();
        });
    });
}

pub fn get_riccsd_rhs2_o3v3(mol_info: &MolInfo, rhs2: TsrMut, intermediates: &CCSDIntermediates, t2: &Tsr) {
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let b_ov = intermediates.cderi.as_ref().unwrap().i((so, sv));
    let b_oo = intermediates.cderi.as_ref().unwrap().i((so, so));
    let b_vv = intermediates.cderi.as_ref().unwrap().i((sv, sv));
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();
    let m1_vv = intermediates.m1_vv.as_ref().unwrap();

    let device = b_ov.device().clone();

    // scr_kcld = np.einsum("kdP, lcP -> kcld", B[so, sv], B[so, sv])

    let scr_kcld: Tsr = rt::zeros(([nocc, nvir, nocc, nvir], &device));
    (0..nocc).into_par_iter().for_each(|k| {
        (0..k + 1).into_par_iter().for_each(|l| {
            let mut scr_kcld = unsafe { scr_kcld.force_mut() };
            let scr = b_ov.i(l) % b_ov.i(k).t();
            *&mut scr_kcld.i_mut((k, .., l)) += &scr;
            if k != l {
                *&mut scr_kcld.i_mut((l, .., k)) += scr.t();
            }
        });
    });

    // === O3V3 Term2 === //
    // scr_ikca = np.einsum("kcld, ilad -> ikca", scr_kcld, (2 * t2 - t2.swapaxes(-1, -2)))
    // RHS2 += -0.5 * np.einsum("ikca, jkbc -> ijab", scr_ikca, t2)

    (0..nocc).into_iter().for_each(|i| {
        let t2_lad: Tsr = 2 * t2.i(i).transpose((1, 0, 2)) - t2.i(i).transpose((2, 0, 1));
        let scr_kca = scr_kcld.reshape((nocc * nvir, -1)) % t2_lad.reshape((nvir, -1)).t();
        let scr_kca = scr_kca.into_shape((nocc, nvir, nvir));
        (0..nocc).into_par_iter().for_each(|j| {
            let mat_ab = (0..nocc)
                .into_par_iter()
                .map(|k| scr_kca.i(k).t() % t2.i((j, k)).t())
                .reduce(|| rt::zeros(([nvir, nvir], &device)), |a: Tsr, b: Tsr| a + b);
            // let mut mat_ab: Tsr = rt::zeros(([nvir, nvir], &device));
            // for k in 0..nocc {
            //     mat_ab += scr_kca.i(k).t() % t2.i((j, k)).t();
            // }
            let mut rhs2 = unsafe { rhs2.force_mut() };
            *&mut rhs2.i_mut((i, j)) -= 0.5 * mat_ab;
        });
    });

    // === O3V3 Term1 === //
    // reuse scr_kcld
    // scr_ikca = 0.5 * np.einsum("kcld, ilda -> ikca", scr_kcld, t2)
    // scr_ikca -= np.einsum("ikP, acP -> ikca", (B[so, so] + M1oo), (B[sv, sv] - M1vv))
    // RHS2 += np.einsum("ikcb, jkca -> ijab", scr_ikca, t2)
    // RHS2 += np.einsum("ikca, jkbc -> ijab", scr_ikca, t2)

    let scr_ikP = b_oo + m1_oo;
    let scr_acP = b_vv - m1_vv;

    (0..nocc).into_iter().for_each(|i| {
        let scr_kca: Tsr = 0.5 * (scr_kcld.reshape((nocc * nvir, -1)) % t2.i(i).reshape((-1, nvir)));
        let scr_kca = scr_kca.into_shape((nocc, nvir, nvir));
        (0..nvir).into_par_iter().for_each(|c| {
            let mut scr_kca = unsafe { scr_kca.force_mut() };
            *&mut scr_kca.i_mut((.., c)) -= scr_ikP.i(i) % scr_acP.i((.., c)).t();
        });
        (0..nocc).into_par_iter().for_each(|j| {
            let mat_ab = (0..nocc)
                .into_par_iter()
                .map(|k| t2.i((j, k)).t() % scr_kca.i(k) + scr_kca.i(k).t() % t2.i((j, k)).t())
                .reduce(|| rt::zeros(([nvir, nvir], &device)), |a: Tsr, b: Tsr| a + b);
            // let mut mat_ab: Tsr = rt::zeros(([nvir, nvir], &device));
            // for k in 0..nocc {
            //     mat_ab += t2.i((j, k)).t() % scr_kca.i(k) + scr_kca.i(k).t() % t2.i((j, k)).t();
            // }
            let mut rhs2 = unsafe { rhs2.force_mut() };
            *&mut rhs2.i_mut((i, j)) += mat_ab;
        });
    });
}

pub fn get_riccsd_rhs2_o4v2(mol_info: &MolInfo, mut rhs2: TsrMut, intermediates: &CCSDIntermediates, t1: &Tsr, t2: &Tsr) {
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let b_ov = intermediates.cderi.as_ref().unwrap().i((so, sv));
    let b_oo = intermediates.cderi.as_ref().unwrap().i((so, so));
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();

    let device = b_ov.device().clone();

    // === O4V2 (HHL) === //
    // Woooo = 0
    // Woooo += np.einsum("ikP, jlP         -> ijkl", B[so, so] + M1oo, B[so, so] + M1oo)
    // Woooo += np.einsum("kcP, ldP, ijcd   -> ijkl", B[so, sv], B[so, sv], t2)
    // RHS2 += 0.5 * np.einsum("ijkl, klab   -> ijab", Woooo, t2,   )
    // RHS2 += 0.5 * np.einsum("ijkl, ka, lb -> ijab", Woooo, t1, t1)

    let scr_bm1_oo = b_oo + m1_oo;

    let mut scr_ijkl: Tsr = rt::zeros(([nocc, nocc, nocc, nocc], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        (0..i + 1).into_par_iter().for_each(|j| {
            let mut scr_ijkl = unsafe { scr_ijkl.force_mut() };
            let scr = scr_bm1_oo.i(i) % scr_bm1_oo.i(j).t();
            *&mut scr_ijkl.i_mut((i, j)) += &scr;
            if i != j {
                *&mut scr_ijkl.i_mut((j, i)) += scr.t();
            }
        });
    });

    let scr_klcd: Tsr = rt::zeros(([nocc, nocc, nvir, nvir], &device));
    (0..nocc).into_par_iter().for_each(|k| {
        (0..k + 1).into_par_iter().for_each(|l| {
            let mut scr_klcd = unsafe { scr_klcd.force_mut() };
            let scr = b_ov.i(k) % b_ov.i(l).t();
            *&mut scr_klcd.i_mut((k, l)) += &scr;
            if k != l {
                *&mut scr_klcd.i_mut((l, k)) += scr.t();
            }
        });
    });

    scr_ijkl += (t2.reshape((nocc * nocc, -1)) % scr_klcd.reshape((nocc * nocc, -1)).t()).reshape(scr_ijkl.shape());
    let tau2 = t2 + t1.i((.., None, .., None)) * t1.i((None, .., None, ..));
    rhs2 += 0.5 * (scr_ijkl.reshape((nocc * nocc, -1)) % tau2.reshape((-1, nvir * nvir))).into_shape(t2.shape());
}

pub fn get_riccsd_rhs2_o2v4(mol_info: &MolInfo, mut rhs2: TsrMut, intermediates: &CCSDIntermediates, t1: &Tsr, t2: &Tsr) {
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;

    let sv = slice!(nocc, nocc + nvir);

    let b_vv = intermediates.cderi.as_ref().unwrap().i((sv, sv));
    let m1_vv = intermediates.m1_vv.as_ref().unwrap();

    let device = b_vv.device().clone();

    // ====== O2V4 (PPL) ====== //
    // Wvvvv = 0
    // Wvvvv += np.einsum("acP, bdP -> abcd", B[sv, sv] - M1vv, B[sv, sv] - M1vv)
    // Wvvvv -= np.einsum("acP, bdP -> abcd", M1vv, M1vv)
    // RHS2 += 0.5 * np.einsum("abcd, ijcd   -> ijab", Wvvvv, t2,   )
    // RHS2 += 0.5 * np.einsum("abcd, ic, jd -> ijab", Wvvvv, t1, t1)

    let rhs_ppl: Tsr<Ix4> = rt::zeros(([nocc, nocc, nvir, nvir], &device)).into_dim::<Ix4>();
    let scr_acP = b_vv - m1_vv;
    let tau2 = t2 + t1.i((.., None, .., None)) * t1.i((None, .., None, ..));

    let nbatch_a = 8;
    let nbatch_b = 32;
    let mut batched_slices = vec![];
    for a_start in (0..nvir).step_by(nbatch_a) {
        let a_end = (a_start + nbatch_a).min(nvir);
        for b_start in (0..a_end).step_by(nbatch_b) {
            let b_end = (b_start + nbatch_b).min(nvir);
            batched_slices.push(([a_start, a_end], [b_start, b_end]));
        }
    }

    batched_slices.into_par_iter().for_each(|([a_start, a_end], [b_start, b_end])| {
        let nbatch_a = a_end - a_start;
        let nbatch_b = b_end - b_start;
        let sa = slice!(a_start, a_end);
        let sb = slice!(b_start, b_end);

        let mut scr_abcd: Tsr = rt::zeros(([nbatch_a, nbatch_b, nvir, nvir], &device));
        // delibrately use serial loop here, but should be possible to be paralleled
        for a in 0..nbatch_a {
            for b in 0..nbatch_b {
                *&mut scr_abcd.i_mut((a, b)) += scr_acP.i(a + a_start) % scr_acP.i(b + b_start).t();
                *&mut scr_abcd.i_mut((a, b)) -= m1_vv.i(a + a_start) % m1_vv.i(b + b_start).t();
            }
        }
        let scr_ijab: Tsr = 0.5 * (tau2.reshape((nocc * nocc, -1)) % scr_abcd.reshape((nbatch_a * nbatch_b, -1)).t());
        let scr_ijab = scr_ijab.into_shape((nocc, nocc, nbatch_a, nbatch_b)).into_dim::<Ix4>();

        let mut rhs_ppl = unsafe { rhs_ppl.force_mut() };
        *&mut rhs_ppl.i_mut((.., .., sa, sb)) += scr_ijab;
    });

    (0..nocc).into_par_iter().for_each(|i| {
        (0..nocc).into_par_iter().for_each(|j| {
            let mut rhs_ppl = unsafe { rhs_ppl.force_mut() };
            for a in 0..nvir {
                for b in 0..a {
                    unsafe {
                        let rhs_ppl_jiab = rhs_ppl.index_uncheck([j, i, a, b]).clone();
                        *rhs_ppl.index_mut_uncheck([i, j, b, a]) = rhs_ppl_jiab;
                    }
                }
            }
        });
    });

    rhs2 += rhs_ppl;
}

pub fn update_riccsd_amplitude(mol_info: &MolInfo, intermediates: &mut CCSDIntermediates, cc_info: &CCSDInfo) -> CCSDInfo {
    let timer_outer = std::time::Instant::now();

    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;
    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let mo_energy = &mol_info.mo_energy;
    let t1 = &cc_info.t1;
    let t2 = &cc_info.t2;
    let mut rhs1 = rt::zeros_like(t1);
    let mut rhs2 = rt::zeros_like(t2);

    let timer = std::time::Instant::now();
    get_riccsd_intermediates_2(mol_info, intermediates, t1);
    println!("Time elapsed (intermediates_2): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs1(mol_info, rhs1.view_mut(), intermediates, t1, t2);
    println!("Time elapsed (rhs1): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_lt2_contract(&mol_info, rhs2.view_mut(), &intermediates, t2);
    println!("Time elapsed (rhs2 lt2_contract): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_direct_dot(&mol_info, rhs2.view_mut(), &intermediates);
    println!("Time elapsed (rhs2 direct_dot): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_o3v3(&mol_info, rhs2.view_mut(), &intermediates, t2);
    println!("Time elapsed (rhs2 o3v3): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_o4v2(&mol_info, rhs2.view_mut(), &intermediates, t1, t2);
    println!("Time elapsed (rhs2 o4v2): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_o2v4(&mol_info, rhs2.view_mut(), &intermediates, t1, t2);
    println!("Time elapsed (rhs2 o2v4): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    let d_ov = mo_energy.i((so, None)) - mo_energy.i((None, sv));
    let t1_new = rhs1 / &d_ov;
    // let t2_new = (&rhs2 + rhs2.transpose((1, 0, 3, 2))) / (d_ov.i((.., None, .., None)) + d_ov.i((None, .., None, ..)));
    let t2_new = rhs2;
    (0..nocc).into_par_iter().for_each(|i| {
        (0..i + 1).into_par_iter().for_each(|j| {
            let mut t2_new = unsafe { t2_new.force_mut() };
            let t2_ab = t2_new.i((i, j)) + t2_new.i((j, i)).t();
            let d2_ab = d_ov.i((i, .., None)) + d_ov.i((j, None, ..));
            let t2_ab = t2_ab / d2_ab;
            t2_new.i_mut((i, j)).assign(&t2_ab);
            if i != j {
                t2_new.i_mut((j, i)).assign(&t2_ab.t());
            }
        });
    });
    println!("Time elapsed (update_t): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_intermediates_1(mol_info, intermediates, &t1_new, &t2_new);
    println!("Time elapsed (intermediates_1): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    let e_corr = get_riccsd_energy(mol_info, &intermediates);
    println!("Time elapsed (energy): {:?}", timer.elapsed());

    let result = CCSDInfo {
        t1: t1_new,
        t2: t2_new,
        e_corr,
    };

    println!("Time elapsed: {:?}", timer_outer.elapsed());

    return result;
}

pub fn naive_riccsd_iteration(mol_info: &MolInfo, cc_config: &CCSDConfig) -> CCSDInfo {
    let nocc = mol_info.nocc;
    let nvir = mol_info.nvir;
    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let device = mol_info.cderi.as_ref().unwrap().device().clone();

    // cderi ao2mo
    let timer = std::time::Instant::now();
    let mut intermediates = CCSDIntermediates::new_empty();
    intermediates.cderi = Some(get_cderi_mo(&mol_info));
    println!("Time elapsed (cderi ao2mo): {:?}", timer.elapsed());

    // initial guess
    let timer = std::time::Instant::now();
    let b_ov = intermediates.cderi.as_ref().unwrap().i((so, sv));
    let mo_energy = &mol_info.mo_energy;
    let d_ov = mo_energy.i((so, None)) - mo_energy.i((None, sv));

    let t1 = rt::zeros(([nocc, nvir], &device));
    let t2 = rt::zeros(([nocc, nocc, nvir, nvir], &device));
    let e_corr_oo = rt::zeros(([nocc, nocc], &device));

    (0..nocc).into_par_iter().for_each(|i| {
        (0..i + 1).into_par_iter().for_each(|j| {
            let d2_ab = d_ov.i((i, .., None)) + d_ov.i((j, None, ..));
            let t2_ab = (b_ov.i(i) % b_ov.i(j).t()) / &d2_ab;
            let mut t2 = unsafe { t2.force_mut() };
            t2.i_mut((i, j)).assign(&t2_ab);
            if i != j {
                t2.i_mut((j, i)).assign(&t2_ab.t());
            }
            let e_bi1 = (&t2_ab * &t2_ab * &d2_ab).sum_all();
            let e_bi2 = (&t2_ab * &t2_ab.swapaxes(-1, -2) * &d2_ab).sum_all();
            let e_corr_ij = 2.0 * e_bi1 - e_bi2;
            let mut e_corr_oo = unsafe { e_corr_oo.force_mut() };
            *e_corr_oo.index_mut([i, j]) = e_corr_ij;
            *e_corr_oo.index_mut([j, i]) = e_corr_ij;
        });
    });

    let e_corr = e_corr_oo.sum_all();
    println!("Initial energy (MP2): {:?}", e_corr);
    println!("Time elapsed (initial guess): {:?}", timer.elapsed());

    let mut ccsd_info = CCSDInfo { t1, t2, e_corr };

    // intermediates_1 should be initialized first before iteration
    let timer = std::time::Instant::now();
    get_riccsd_intermediates_1(mol_info, &mut intermediates, &ccsd_info.t1, &ccsd_info.t2);
    println!("Time elapsed (initialization of intermediates_1): {:?}", timer.elapsed());

    for niter in 0..cc_config.max_cycle {
        println!("Iteration: {:?}", niter);
        let ccsd_info_new = update_riccsd_amplitude(mol_info, &mut intermediates, &ccsd_info);
        println!("    Energy: {:?}", ccsd_info.e_corr);
        let diff_eng = ccsd_info_new.e_corr - ccsd_info.e_corr;
        let diff_t1 = (&ccsd_info_new.t1 - &ccsd_info.t1).abs().sum_all();
        let diff_t2 = (&ccsd_info_new.t2 - &ccsd_info.t2).abs().sum_all();
        println!("    Energy diff: {:?}", diff_eng);
        println!("    T1 diff: {:?}", diff_t1);
        println!("    T2 diff: {:?}", diff_t2);
        if diff_eng.abs() < cc_config.conv_tol_e && diff_t1 < cc_config.conv_tol_t1 && diff_t2 < cc_config.conv_tol_t2 {
            println!("CCSD converged in {niter} iterations.");
            return ccsd_info_new;
        }
        ccsd_info = ccsd_info_new;
    }

    panic!("[fatal] CCSD did not converge.");
}
