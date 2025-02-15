# Showcase of RI-CCSD with rust

This program evaluates restricted RI-CCSD energy. By pure Rust code.

> To non-chemists: RI-CCSD can be seen as a group of dense 2-4 dimension tensor numerical computations. Most of tasks in RI-CCSD can be converted to matrix multiplication, so it is mostly compute-bounded.
>
> RI-CCSD is 4-dimension problem in it's nature; coding with libraries that focus on 2-dimension matrices is usually not convenient, if not impossible.
>
> Refer incomplete article list ([Q-Chem](https://dx.doi.org/10.1063/1.4820484), [Psi4 fnocc](https://dx.doi.org/10.1021/ct400250u), [FHI-Aims](https://dx.doi.org/10.1021/acs.jctc.8b01294), [Gamess US](https://dx.doi.org/10.1021/acs.jctc.1c00389), to name a few) to interested readers.

## Contents

~ 600 lines of code [riccsd.rs](src/riccsd.rs), with comments that how numpy implements with `np.einsum`.

This showcase uses [RSTSR](https://github.com/ajz34/rstsr) as tensor library with [commit 403e815](https://github.com/ajz34/rstsr/commit/403e815f9014f60346716b4fc754b78e0006db9f). This crate is under development, and currently has not been published to crates.io.

This showcase hope to show that, with some sugar from tensor (n-dimensional array) library, rust can handle problems that most FLOPs come from matmul of large dense matrices, with
- acceptable lines of code (maybe 1.5-4 times of numpy if `np.einsum` is not allowed)
- acceptable efficiency
- rayon parallel with acceptable unsafe mutables

In other words, rust is a proper candidate for a few scientific computing tasks, to balance program efficiency / code readibility / development efficiency / memory reliablity.

**This conclusion is not trivial**, and existing tensor libraries in rust language seem not fully prepared for this task; so a showcase project is here to demonstrate the possibility.

## Efficiency demonstration

Computation device information:
- personal computer
- AMD Ryzen 7945HX, 16 physical cores
- only one NUMA node
- 2 x 32 GB memory (5600 MT/s)

System information:
- (H2O)<sub>10</sub> cluster, PP5 structure from [10.1021/jp104865w](https://dx.doi.org/10.1021/jp104865w).
- basis: cc-pVDZ
- auxiliary basis: cc-pVDZ-ri (both for SCF and CCSD, which is not recommended for real-world evaluation, but this project is only efficiency benchmark)
- $n_\mathrm{occ} = 40$ (frozen core), $n_\mathrm{vir} = 190$, $n_\mathrm{aux} = 820$.

| | this showcase | Psi4 | PySCF |
|--|--|--|--|
| corr eng (a.u.) | -2.1735512 | -2.1735494 | -2.1735499 |
| time each iter (sec) | ~ 18.5 | ~ 19.0 | ~ 29.5 |
| version | - | 1.9.1 (conda-forge) | 2.7.0 (pypi) |
| math library | OpenBLAS (compiled) | Intel OneAPI (conda-forge) | OpenBLAS (pypi) |
| math library threading | pthread | TBB | serial |
| algorithms | DF | DF (fnocc) | Conv |

*DF* refers to density fitting integral and algorithms, *Conv* refers to conventional integral.

Some important notes:
- Psi4 uses Intel OneAPI (MKL), which is not very efficienct on AMD CPUs, so this is actually not fair comparasion. Estimated 20% efficiency boost if using OpenBLAS.
- Psi4 have multiple CC engines. FNOCC is more efficient, while OCC have more functionalities.
- By comparing to conventional integral algorithms, density fitting (RI-CCSD) actually increases FLOPs, but only decreases memory footprints for large species (if I get both RI-CCSD and Conv-CCSD algorithms correctly).

## Details of Efficiency

- Time of each iter: ~ 18.5 sec
- $O(n_\mathrm{occ}^3 n_\mathrm{vir}^3)$ term: ~ 6.0 sec
    - FLOPs estimation: slightly larger than $2 \times 4 \times n_\mathrm{occ}^3 n_\mathrm{vir}^3 = 3.27 \ \mathrm{T}$
    - ~ 540 GFLOP/sec, 48% CPU maximum L1 bandwidth
- $O(n_\mathrm{occ}^2 n_\mathrm{vir}^4)$ term (pp-Ladder): ~ 8.8 sec
    - FLOPs estimation: slightly larger than $2 \times (n_\mathrm{vir}^4 n_\mathrm{aux} + 0.5 \times n_\mathrm{occ}^2 n_\mathrm{vir}^4) = 3.98 \ \mathrm{T}$
    - ~ 450 GFLOP/sec, 40% CPU maximum L1 bandwidth

We expect 50% efficiency usage is achievable, but that requires more fine-tuned code.

This project has not optimized for lowering memory footprints. This code accepts dupilcating some $O(n_\mathrm{occ}^2 n_\mathrm{vir}^2)$ and $O(n_\mathrm{vir}^2 n_\mathrm{aux})$ tensors. This project also does not use advanced iteration drivers (DIIS), so more iterations than usual is expected.

## To reproduce

This project is only for efficiency and code style demonstration. Usability is not the first concern.

Binary file is available for this showcase. Refer to [release page](https://github.com/ajz34/showcase_rust_riccsd/releases/tag/v0.1).

To use this binary, some preparation is required:
```bash
export RAYON_NUM_THREADS=16     # number of parallel
export RUST_MIN_STACK=16777216  # could be larger if stack overflow
export LD_LIBRARY_PATH=<your pthread openblas directory>:$LD_LIBRARY_PATH

./showcase_rust_riccsd_glibc_2.17 <directory of your npy files>
```

This project requires the user (more details in [env file](env.sh) or [vscode setting](.vscode/settings.json))
- Provide `libopenblas.so` (pthread scheme) in `$LD_LIBRARY_PATH`. Due to how rust's FFI works, OpenMP compiled OpenBLAS does not work;
- Provide `*.npy` files, in pyscf convention (see [python_scripts](python_scripts) for details):
    - mo_coeff.npy (in c-contiguous, shape (nao, nmo))
    - mo_energy.npy
    - mo_coeff.npy
    - cderi.npy (in lower-triangular packed AO basis)
