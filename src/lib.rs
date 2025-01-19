#![allow(non_snake_case)]

use rstsr_core::prelude::{IxD, Tensor, TensorMut, TensorView};
use rstsr_openblas::DeviceOpenBLAS;

pub type Tsr<D = IxD> = Tensor<f64, DeviceOpenBLAS, D>;
pub type TsrView<'a, D = IxD> = TensorView<'a, f64, DeviceOpenBLAS, D>;
pub type TsrMut<'a, D = IxD> = TensorMut<'a, f64, DeviceOpenBLAS, D>;

pub mod cc_structs;
pub mod riccsd;
pub mod util;

pub use cc_structs::*;
pub use riccsd::*;
pub use util::*;

#[cfg(test)]
mod tests_h2o;
