#![allow(dead_code)]

use crate::*;

pub struct MolInfo {
    pub(crate) mo_energy: Tsr,
    pub(crate) mo_coeff: Tsr,
    pub(crate) mo_occ: Tsr,
    pub(crate) nao: usize,
    pub(crate) nmo: usize,
    pub(crate) nocc: usize,
    pub(crate) nvir: usize,
    pub(crate) naux: usize,
    pub(crate) cderi: Option<Tsr>,
}

pub struct CCSDConfig {
    pub max_cycle: usize,
    pub conv_tol_e: f64,
    pub conv_tol_t1: f64,
    pub conv_tol_t2: f64,
}

pub struct CCSDInfo {
    pub e_corr: f64,
    pub t1: Tsr,
    pub t2: Tsr,
}

pub struct CCSDIntermediates {
    pub(crate) cderi: Option<Tsr>, // in molecular orbital basis
    pub(crate) m1_j: Option<Tsr>,
    pub(crate) m1_oo: Option<Tsr>,
    pub(crate) m1_vv: Option<Tsr>,
    pub(crate) m1a_ov: Option<Tsr>,
    pub(crate) m1b_ov: Option<Tsr>,
    pub(crate) m2a_ov: Option<Tsr>,
    pub(crate) m2b_ov: Option<Tsr>,
}

impl CCSDIntermediates {
    pub fn new_empty() -> Self {
        CCSDIntermediates {
            cderi: None,
            m1_j: None,
            m1_oo: None,
            m1_vv: None,
            m1a_ov: None,
            m1b_ov: None,
            m2a_ov: None,
            m2b_ov: None,
        }
    }
}
