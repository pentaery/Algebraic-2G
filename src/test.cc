#include "matCPU.hh"
#include <iostream>
#include <metis.h>
#include <petscdevicetypes.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscksp.h>
#include <petsclogtypes.h>
#include <petscmat.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <slepceps.h>
#include <slepcst.h>

// PetscErrorCode MyMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *ctx) {
//   PetscPrintf(PETSC_COMM_WORLD, "Iteration %d: True Residual Norm %g\n", n,
//               rnorm);
//   return 0;
// }

PetscInt main(PetscInt argc, char *argv[]) {

  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));

  PetscLogEvent matGen, LS, MAS, EigenSolver;
  PetscCall(PetscLogEventRegister("MatGen", 0, &matGen));
  PetscCall(PetscLogEventRegister("LS", 1, &LS));
  PetscCall(PetscLogEventRegister("MatAssembly", 2, &MAS));
  PetscCall(PetscLogEventRegister("EPSolver", 3, &EigenSolver));

  PetscCall(PetscLogEventBegin(matGen, 0, 0, 0, 0));

  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscInt rank, nprocs;
  PetscCall(MPI_Comm_rank(comm, &rank));
  PetscCall(MPI_Comm_size(comm, &nprocs));

  PetscInt *cluster_sizes = NULL;
  PetscCall(PetscMalloc1(nprocs, &cluster_sizes));

  PetscScalar *arr_B;
  const PetscInt *row_B;
  const PetscInt *col_B;

  Mat B;
  if (rank == 0) {

    for (PetscInt i = 0; i < nprocs; ++i) {
      cluster_sizes[i] = 0;
    }

    Mat A;

    PetscInt nrows, nnz;
    std::vector<PetscScalar> values(1);
    std::vector<PetscInt> col_indices(1);
    std::vector<PetscInt> row_ptr(1);

    // readMat(&nrows, &nnz, row_ptr, col_indices, values);
    // generateMat(&nrows, &nnz, row_ptr, col_indices, values);
    PetscInt meshsize = 50;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &meshsize, NULL));
    generateMatMFEM(&nrows, &nnz, row_ptr, col_indices, values, meshsize);
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "MFEM Matrix generated...\n"));

    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nrows, nrows,
                                        row_ptr.data(), col_indices.data(),
                                        values.data(), &A));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Starting partitioning...\n"));
    PetscInt ncon = 1;
    PetscInt objval;
    PetscInt options[METIS_NOPTIONS];
    std::vector<PetscInt> part(nrows);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    options[METIS_OPTION_NCUTS] = 1;
    METIS_PartGraphKway(&nrows, &ncon, row_ptr.data(), col_indices.data(), NULL,
                        NULL, NULL, &nprocs, NULL, NULL, NULL, &objval,
                        part.data());
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
                          "Partitioning completed with objective value %d\n",
                          objval));

    IS is;
    std::vector<PetscInt> idx(nrows, 0);
    for (PetscInt i = 0; i < nrows; ++i) {
      idx[i] = cluster_sizes[part[i]];
      cluster_sizes[part[i]]++;
    }
    for (PetscInt i = 0; i < nrows; ++i) {
      for (PetscInt j = 0; j < part[i]; ++j) {
        idx[i] += cluster_sizes[j];
      }
    }

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nrows, idx.data(),
                              PETSC_COPY_VALUES, &is));
    IS isinvert;
    PetscCall(ISInvertPermutation(is, nrows, &isinvert));

    PetscCall(MatPermute(A, isinvert, isinvert, &B));

    PetscCall(ISDestroy(&is));
    PetscCall(ISDestroy(&isinvert));
    PetscCall(MatDestroy(&A));

    PetscCall(MatSeqAIJGetCSRAndMemType(B, &row_B, &col_B, &arr_B, NULL));

    // std::ofstream out("clustering.txt");
    // if (!out) {
    //   std::cerr << "Error: Could not open clustering.txt for writing!"
    //             << std::endl;
    //   return 1;
    // }
    // for (PetscInt i = 0; i < nrows; i++) {
    //   out << part[i] << std::endl;
    // }
    // out.close();
  }

  PetscCall(MPI_Bcast(cluster_sizes, nprocs, MPI_INT, 0, comm));

  PetscInt *row_offset = NULL;
  PetscCall(PetscMalloc1(nprocs, &row_offset));
  row_offset[0] = 0;
  for (PetscInt i = 1; i < nprocs; ++i) {
    row_offset[i] = row_offset[i - 1] + cluster_sizes[i - 1];
  }

  PetscInt *sendcounts_rowptr, *displs_rowptr, *sendcounts_nnz, *displs_nnz;
  PetscCall(PetscMalloc1(nprocs, &sendcounts_rowptr));
  PetscCall(PetscMalloc1(nprocs, &displs_rowptr));
  PetscCall(PetscMalloc1(nprocs, &sendcounts_nnz));
  PetscCall(PetscMalloc1(nprocs, &displs_nnz));

  // 计算每个进程的偏移量
  if (rank == 0) {
    for (PetscInt i = 0; i < nprocs; ++i) {
      sendcounts_rowptr[i] = cluster_sizes[i] + 1;
      displs_rowptr[i] = row_offset[i];
      sendcounts_nnz[i] =
          row_B[row_offset[i] + cluster_sizes[i]] - row_B[row_offset[i]];
      displs_nnz[i] = row_B[row_offset[i]];
    }
  }

  PetscInt local_nnz;
  PetscCall(MPI_Scatter(sendcounts_nnz, 1, MPIU_INT, &local_nnz, 1, MPIU_INT, 0,
                        PETSC_COMM_WORLD));

  PetscInt local_rows = cluster_sizes[rank];

  PetscFree(cluster_sizes);

  PetscInt *local_row_ptr, *local_col_index;
  PetscScalar *local_values;
  PetscCall(PetscMalloc1(local_rows + 1, &local_row_ptr));
  PetscCall(PetscMalloc1(local_nnz, &local_col_index));
  PetscCall(PetscMalloc1(local_nnz, &local_values));

  PetscCall(MPI_Scatterv(row_B, sendcounts_rowptr, displs_rowptr, MPIU_INT,
                         local_row_ptr, local_rows + 1, MPIU_INT, 0,
                         PETSC_COMM_WORLD));
  PetscCall(MPI_Scatterv(col_B, sendcounts_nnz, displs_nnz, MPIU_INT,
                         local_col_index, local_nnz, MPIU_INT, 0,
                         PETSC_COMM_WORLD));
  PetscCall(MPI_Scatterv(arr_B, sendcounts_nnz, displs_nnz, MPIU_SCALAR,
                         local_values, local_nnz, MPIU_SCALAR, 0,
                         PETSC_COMM_WORLD));

  for (PetscInt i = local_rows; i >= 0; --i) {
    local_row_ptr[i] -= local_row_ptr[0];
  }

  PetscCall(PetscFree(sendcounts_rowptr));
  PetscCall(PetscFree(displs_rowptr));
  PetscCall(PetscFree(sendcounts_nnz));
  PetscCall(PetscFree(displs_nnz));
  PetscCall(PetscFree(row_offset));

  PetscCall(MatDestroy(&B));

  Mat C;

  PetscCall(MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, local_rows, local_rows,
                                      PETSC_DECIDE, PETSC_DECIDE, local_row_ptr,
                                      local_col_index, local_values, &C));

  PetscCall(MatScale(C, 100000));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Finished creating MPI matrix! \n"));

  // PetscCall(MatView(C, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscLogEventEnd(matGen, 0, 0, 0, 0));

  Vec b;
  PetscCall(MatCreateVecs(C, &b, NULL));
  PetscCall(MatGetDiagonal(C, b));

  PetscCall(PetscLogEventBegin(LS, 0, 0, 0, 0));
  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, C, C));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));

  PetscBool using2G = PETSC_TRUE, usingHEP = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-using2G", &using2G, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-usingHEP", &usingHEP, NULL));
  PetscInt eigennum = 4;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-eigennum", &eigennum, NULL));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));

  if (using2G) {
    PetscCall(PetscLogEventBegin(MAS, 0, 0, 0, 0));
    Mat Adiagonal, Ai;
    PetscCall(MatGetDiagonalBlock(C, &Adiagonal));
    PetscCall(MatDuplicate(Adiagonal, MAT_COPY_VALUES, &Ai));

    Vec rowsum;
    PetscCall(MatCreateVecs(Ai, &rowsum, NULL));
    PetscCall(VecSetFromOptions(rowsum));
    PetscCall(MatGetRowSum(Ai, rowsum));
    // PetscCall(VecView(rowsum, PETSC_VIEWER_STDOUT_SELF));
    Vec diagonal, vdiagonal;
    PetscCall(MatCreateVecs(Ai, &vdiagonal, NULL));
    PetscCall(VecDuplicate(vdiagonal, &diagonal));
    PetscCall(MatGetDiagonal(Ai, diagonal));

    PetscCall(VecAXPY(diagonal, -1.0, rowsum));
    PetscCall(MatDiagonalSet(Ai, diagonal, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Ai, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Ai, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(Ai, MAT_SYMMETRIC, PETSC_TRUE));

    // PetscCall(MatScale(Ai, 100000));

    Mat Si;
    if (usingHEP) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solving HEP EigenProblem\n"));
      PetscCall(VecReciprocal(diagonal));
      PetscCall(VecSqrtAbs(diagonal));
      PetscCall(MatDiagonalScale(Ai, diagonal, diagonal));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solving GHEP EigenProblem\n"));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, local_rows, local_rows, 1,
                                NULL, &Si));
      PetscCall(MatDiagonalSet(Si, diagonal, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(Si, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Si, MAT_FINAL_ASSEMBLY));
      PetscCall(MatSetOption(Si, MAT_SYMMETRIC, PETSC_TRUE));
      // PetscCall(MatScale(Si, 100000));
    }

    // if (rank == 0) {
    //   // PetscCall(MatView(Ai, PETSC_VIEWER_STDOUT_SELF));
    //   // PetscCall(MatView(Si, PETSC_VIEWER_STDOUT_SELF));
    //   PetscViewer viewer;
    //   PetscViewerASCIIOpen(PETSC_COMM_SELF, "vector.txt", &viewer);

    //   // 将向量输出到文件
    //   VecView(diagonal, viewer);
    // }

    Mat R;
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, local_rows, eigennum,
                           PETSC_DETERMINE, PETSC_DETERMINE, eigennum, NULL, 0,
                           NULL, &R));
    PetscCall(PetscLogEventBegin(EigenSolver, 0, 0, 0, 0));
    EPS eps;
    PetscInt nconv;
    PetscScalar eig_val, *arr_eig_vec;
    Vec eig_vec;
    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    if (usingHEP) {
      PetscCall(EPSSetOperators(eps, Ai, NULL));
      PetscCall(EPSSetProblemType(eps, EPS_HEP));
    } else {
      PetscCall(EPSSetOperators(eps, Ai, Si));
      PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    }
    PetscCall(EPSSetType(eps, EPSKRYLOVSCHUR));
    PetscCall(EPSSetTolerances(eps, 1e-6, 1000));
    PetscCall(EPSSetTarget(eps, 1e-12));
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL));
    PetscCall(EPSSetDimensions(eps, eigennum, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(MatCreateVecs(Ai, &eig_vec, NULL));
    ST st;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(EPSSetOptionsPrefix(eps, "eps"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));

    PetscCall(PetscLogEventEnd(EigenSolver, 0, 0, 0, 0));

    PetscCheck(nconv >= eigennum, PETSC_COMM_SELF, PETSC_ERR_USER,
               "Not enough converged eigenvalues found!");

    PetscInt row_start, row_end;
    PetscCall(MatGetOwnershipRange(C, &row_start, &row_end));
    PetscInt *idxm;
    PetscCall(PetscMalloc1(local_rows, &idxm));
    for (PetscInt i = 0; i < local_rows; ++i) {
      idxm[i] = row_start + i;
      // idxm[i] = i;
    }

    for (PetscInt j = 0; j < eigennum; ++j) {

      PetscInt idxn = eigennum * rank + j;
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));

      // if (usingHEP) {
      //   PetscInt its;
      //   PetscCall(EPSGetIterationNumber(eps, &its));
      //   PetscScalar err;
      //   PetscCall(EPSComputeError(eps, j, EPS_ERROR_ABSOLUTE, &err));
      //   PetscCall(PetscPrintf(
      //       PETSC_COMM_SELF,
      //       "Rank %d, Eiegnvalue %d, Absolute error of the first eigenvalue:
      //       "
      //       "%.18f, iter %d\n",
      //       rank, j, err, its));
      // } else {
      //   PetscInt its;
      //   PetscCall(EPSGetIterationNumber(eps, &its));
      //   PetscScalar err;
      //   PetscCall(EPSComputeError(eps, j, EPS_ERROR_ABSOLUTE, &err));
      //   PetscCall(PetscPrintf(
      //       PETSC_COMM_SELF,
      //       "Rank %d, Eiegnvalue %d, Absolute error of the first eigenvalue:
      //       "
      //       "%.18f, iter %d\n",
      //       rank, j, err, its));
      // }

      if (usingHEP) {
        PetscCall(VecPointwiseMult(eig_vec, eig_vec, diagonal));
      }

      PetscCall(PetscPrintf(PETSC_COMM_SELF,
                            "Rank %d, number %d, eigval %.18f\n", rank, j,
                            eig_val));
      PetscCall(VecGetArray(eig_vec, &arr_eig_vec));
      PetscCall(MatSetValues(R, local_rows, idxm, 1, &idxn, arr_eig_vec,
                             INSERT_VALUES));
    }

    PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));

    // PetscCall(MatView(R, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(PetscLogEventEnd(MAS, 0, 0, 0, 0));

    PetscCall(MatDestroy(&Ai));
    if (!usingHEP) {
      PetscCall(MatDestroy(&Si));
    }
    PetscCall(VecDestroy(&eig_vec));

    KSP kspCoarse, kspSmoother;
    PC pcCoarse, pcSmoother;
    // 设置二层multigrid
    PetscCall(PCSetType(pc, PCMG));
    PetscCall(PCMGSetLevels(pc, 2, NULL));
    // 设为V-cycle
    PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
    PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
    PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
    // 设置coarse solver
    PetscCall(PCMGGetCoarseSolve(pc, &kspCoarse));
    PetscCall(KSPSetType(kspCoarse, KSPPREONLY));
    PetscCall(KSPGetPC(kspCoarse, &pcCoarse));
    PetscCall(PCSetType(pcCoarse, PCLU));
    PetscCall(PCFactorSetMatSolverType(pcCoarse, MATSOLVERMKL_CPARDISO));
    PetscCall(KSPSetErrorIfNotConverged(kspCoarse, PETSC_TRUE));
    // PetscCall(KSPMonitorSet(kspCoarse, MyMonitor, NULL, NULL));
    PetscCall(KSPSetFromOptions(kspCoarse));
    // 设置一阶smoother
    PetscCall(PCMGGetSmoother(pc, 1, &kspSmoother));
    PetscCall(KSPSetFromOptions(kspSmoother));
    PetscCall(KSPSetTolerances(kspSmoother, PETSC_DEFAULT, PETSC_DEFAULT,
                               PETSC_DEFAULT, 1));
    // PetscCall(KSPMonitorSet(kspSmoother, MyMonitor, NULL, NULL));
    PetscCall(KSPGetPC(kspSmoother, &pcSmoother));
    PetscCall(PCSetType(pcSmoother, PCBJACOBI));
    PetscCall(KSPSetErrorIfNotConverged(kspSmoother, PETSC_TRUE));
    // 设置Prolongation
    PetscCall(PCMGSetInterpolation(pc, 1, R));
    PetscCall(
        PCShellSetName(pc, "2levels-MG-via-GMsFEM-with-velocity-elimination"));

  } else {
    PetscCall(PCSetType(pc, PCGAMG));
  }

  // PetscCall(MatDiagonalSet(C, b, INSERT_VALUES));
  PetscCall(MatShift(C, 1e-5));

  Vec x;
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(x, 0.0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Starting to solve...\n"));
  // PetscCall(KSPMonitorSet(ksp, MyMonitor, NULL, NULL));

  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(PetscLogEventEnd(LS, 0, 0, 0, 0));
  KSPConvergedReason reason;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscInt num;
  PetscCall(KSPGetIterationNumber(ksp, &num));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of iterations: %d\n", num));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged reason: %d\n", reason));

  PetscCall(MatDestroy(&C));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(SlepcFinalize());
}