#include "matCPU.hh"
#include <iostream>
#include <metis.h>
#include <petscmat.h>
#include <petscpctypes.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <slepceps.h>
#include <slepcst.h>

PetscErrorCode MyMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *ctx) {
  PetscPrintf(PETSC_COMM_WORLD, "Iteration %d: True Residual Norm %g\n", n,
              rnorm);
  return 0;
}

PetscInt main(PetscInt argc, char *argv[]) {

  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));

  Mat A;

  PetscInt nrows, nnz;
  std::vector<PetscScalar> values(1);
  std::vector<PetscInt> col_indices(1);
  std::vector<PetscInt> row_ptr(1);

  // readMat(&nrows, &nnz, row_ptr, col_indices, values);
  // generateMat(&nrows, &nnz, row_ptr, col_indices, values);
  PetscInt meshsize = 10;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &meshsize, NULL));
  generateMatMFEM(&nrows, &nnz, row_ptr, col_indices, values, meshsize);
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "MFEM Matrix generated...\n"));

  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nrows, nrows,
                                      row_ptr.data(), col_indices.data(),
                                      values.data(), &A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatShift(A, 1e-5));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPCG));
  Vec rhs, sol;
  PetscCall(MatCreateVecs(A, &rhs, NULL));
  PetscCall(VecDuplicate(rhs, &sol));
  PetscCall(MatGetDiagonal(A, rhs));
  PetscCall(KSPSetFromOptions(ksp));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCGAMG));

  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSolve(ksp, rhs, sol));

  // 验证ksp有没有收敛
  PetscInt its;
  PetscReal rnorm;
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPGetResidualNorm(ksp, &rnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "KSP converged in %d iterations with "
                        "residual norm %g\n",
                        its, rnorm));

  PetscCall(VecView(sol, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(SlepcFinalize());
}