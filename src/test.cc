#include "matCPU.hh"

#include <iomanip>
#include <iostream>
#include <metis.h>
#include <petscdevicetypes.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <slepceps.h>

#include <chrono>
#include <slepcst.h>

#define cStar 1.0
#define scalartype double

PetscErrorCode MyMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *ctx) {
  PetscPrintf(PETSC_COMM_WORLD, "Iteration %d: True Residual Norm %g\n", n,
              rnorm);
  return 0;
}

int main(int argc, char *argv[]) {

  auto start = std::chrono::high_resolution_clock::now();
  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));

  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscInt rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  PetscInt *cluster_sizes = NULL;
  PetscCall(PetscMalloc1(nprocs, &cluster_sizes));

  PetscScalar *arr_B;
  const PetscInt *row_B;
  const PetscInt *col_B;

  if (rank == 0) {

    for (int i = 0; i < nprocs; ++i) {
      cluster_sizes[i] = 0;
    }

    Mat A, B;

    int nrows, nnz;
    std::vector<PetscScalar> values(1);
    std::vector<PetscInt> col_indices(1);
    std::vector<PetscInt> row_ptr(1);

    readMat(&nrows, &nnz, row_ptr, col_indices, values);
    // generateMat(&nrows, &nnz, row_ptr, col_indices, values);

    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nrows, nrows,
                                        row_ptr.data(), col_indices.data(),
                                        values.data(), &A));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    // PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF));

    int ncon = 1;
    int objval;
    int options[METIS_NOPTIONS];
    int nparts = 5;
    std::vector<int> part(nrows);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    options[METIS_OPTION_NCUTS] = 1;
    METIS_PartGraphKway(&nrows, &ncon, row_ptr.data(), col_indices.data(), NULL,
                        NULL, NULL, &nprocs, NULL, NULL, NULL, &objval,
                        part.data());
    std::cout << "Objective for the partition is " << objval << std::endl;

    auto end = std::chrono::high_resolution_clock::now();

    IS is;
    std::vector<PetscInt> idx(nrows, 0);

    std::cout << std::endl;
    for (int i = 0; i < nrows; ++i) {
      idx[i] = cluster_sizes[part[i]];
      cluster_sizes[part[i]]++;
    }
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < part[i]; ++j) {
        idx[i] += cluster_sizes[j];
      }
    }

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nrows, idx.data(),
                              PETSC_COPY_VALUES, &is));
    IS isinvert;
    PetscCall(ISInvertPermutation(is, nrows, &isinvert));

    PetscCall(MatPermute(A, isinvert, isinvert, &B));
    // PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF));

    PetscMemType mtype;
    PetscCall(MatSeqAIJGetCSRAndMemType(B, &row_B, &col_B, &arr_B, &mtype));

    // for (int i = 0; i < nnz; ++i) {
    //   std::cout << col_B[i] << " ";
    // }
    // std::cout << std::endl << std::endl << std::endl;

    // 打印每个cluster控制的行索引
    // std::vector<std::vector<int>> cluster_indices(nparts);
    // for (int i = 0; i < nrows; ++i) {
    //   cluster_indices[part[i]].push_back(i);
    // }
    // for (int i = 0; i < nparts; ++i) {
    //   std::cout << "Cluster " << i << ": ";
    //   for (int j = 0; j < cluster_indices[i].size(); ++j) {
    //     std::cout << cluster_indices[i][j] << " ";
    //   }
    //   std::cout << std::endl;
    // }

    // 计算时间差
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken: " << duration.count() / 1e6 << " seconds"
              << std::endl;

    std::ofstream out("clustering.txt");
    if (!out) {
      std::cerr << "Error: Could not open clustering.txt for writing!"
                << std::endl;
      return 1;
    }
    for (int i = 0; i < nrows; i++) {
      out << part[i] << std::endl;
    }
    out.close();
  }

  PetscCall(MPI_Bcast(cluster_sizes, nprocs, MPI_INT, 0, comm));

  PetscInt *row_offset = NULL;
  PetscCall(PetscMalloc1(nprocs, &row_offset));
  row_offset[0] = 0;
  for (int i = 1; i < nprocs; ++i) {
    row_offset[i] = row_offset[i - 1] + cluster_sizes[i - 1];
  }
  // PetscInt total_rows = row_offset[nprocs - 1] + cluster_sizes[nprocs - 1];

  PetscInt *sendcounts_rowptr, *displs_rowptr, *sendcounts_nnz, *displs_nnz;
  PetscCall(PetscMalloc1(nprocs, &sendcounts_rowptr));
  PetscCall(PetscMalloc1(nprocs, &displs_rowptr));
  PetscCall(PetscMalloc1(nprocs, &sendcounts_nnz));
  PetscCall(PetscMalloc1(nprocs, &displs_nnz));

  // 计算每个进程的偏移量
  if (rank == 0) {
    for (int i = 0; i < nprocs; ++i) {
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

  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "rank %d, local_rows %d\n", rank,
  //               local_rows));

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

  for (int i = local_rows; i >= 0; --i) {
    local_row_ptr[i] -= local_row_ptr[0];
  }

  Mat C;

  PetscCall(MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, local_rows, local_rows,
                                      PETSC_DECIDE, PETSC_DECIDE, local_row_ptr,
                                      local_col_index, local_values, &C));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Finished creating matrix! \n"));

  // PetscCall(MatView(C, PETSC_VIEWER_STDOUT_WORLD));

  Vec b;
  PetscCall(MatCreateVecs(C, &b, NULL));
  PetscCall(MatGetDiagonal(C, b));

  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, C, C));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));

  PetscBool using2G = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-using2G", &using2G, NULL));
  int eigennum = 4;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-eigennum", &eigennum, NULL));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  if (using2G) {
    Mat Ai;
    PetscCall(MatGetDiagonalBlock(C, &Ai));
    Vec rowsum;
    PetscCall(MatCreateVecs(Ai, &rowsum, NULL));
    PetscCall(VecSetFromOptions(rowsum));
    PetscCall(MatGetRowSum(Ai, rowsum));
    // PetscCall(VecView(rowsum, PETSC_VIEWER_STDOUT_SELF));
    Vec diagonal;
    PetscCall(MatCreateVecs(Ai, &diagonal, NULL));
    PetscCall(MatGetDiagonal(Ai, diagonal));

    PetscCall(VecAXPY(diagonal, -1.0, rowsum));
    PetscCall(MatDiagonalSet(Ai, diagonal, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Ai, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Ai, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(Ai, MAT_SYMMETRIC, PETSC_TRUE));

    Vec ones;
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, local_rows, &ones));
    PetscCall(VecSet(ones, 1.0));
    Mat Si;
    PetscCall(
        MatCreateSeqAIJ(PETSC_COMM_SELF, local_rows, local_rows, 1, NULL, &Si));
    PetscCall(MatZeroEntries(Si));
    PetscCall(MatDiagonalSet(Si, diagonal, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Si, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Si, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(Si, MAT_SYMMETRIC, PETSC_TRUE));

    // PetscCall(MatView(C, PETSC_VIEWER_STDOUT_WORLD));
    // if (rank == 2) {
    //   PetscCall(MatView(Ai, PETSC_VIEWER_STDOUT_SELF));
    //   PetscCall(MatView(Si, PETSC_VIEWER_STDOUT_SELF));
    // }

    Mat R;
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, local_rows, eigennum,
                           PETSC_DEFAULT, PETSC_DEFAULT, eigennum, NULL, 0,
                           NULL, &R));

    EPS eps;
    PetscInt nconv;
    PetscScalar eig_val, *arr_eig_vec;
    Vec eig_vec;
    PetscCall(MatCreateVecs(C, &eig_vec, NULL));
    PetscCall(VecSetFromOptions(eig_vec));
    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    PetscCall(EPSSetOperators(eps, Ai, Si));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    PetscCall(EPSSetDimensions(eps, eigennum, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(MatCreateVecs(Ai, &eig_vec, NULL));
    ST st;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(EPSSetTarget(eps, 1e-12));
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL));
    PetscCall(EPSSetOptionsPrefix(eps, "epsl1_"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));

    PetscCheck(nconv >= eigennum, PETSC_COMM_SELF, PETSC_ERR_USER,
               "Not enough converged eigenvalues found!");

    PetscInt row_start, row_end;
    PetscCall(MatGetOwnershipRange(C, &row_start, &row_end));
    PetscInt *idxm;
    PetscCall(PetscMalloc1(local_rows, &idxm));
    for (int i = 0; i < local_rows; ++i) {
      idxm[i] = row_start + i;
    }

    for (int j = 0; j < eigennum; ++j) {
      PetscInt idxn = eigennum * rank + j;
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      PetscCall(PetscPrintf(PETSC_COMM_SELF,
                            "Rank %d, number %d, eigval %.18f\n", rank, j,
                            eig_val));
      PetscCall(VecGetArray(eig_vec, &arr_eig_vec));
      PetscCall(MatSetValues(R, local_rows, idxm, 1, &idxn, arr_eig_vec,
                             INSERT_VALUES));
      // if (j == 0 && rank == 0) {
      //   Vec rhs;
      //   PetscCall(MatCreateVecs(Ai, NULL, &rhs));
      //   PetscCall(MatMult(Ai, eig_vec, rhs));
      //   PetscCall(VecView(rhs, PETSC_VIEWER_STDOUT_SELF));
      //   PetscCall(VecDestroy(&rhs));
      // }
    }

    PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));

    // PetscCall(MatView(R, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(MatDestroy(&Ai));
    PetscCall(MatDestroy(&Si));
    PetscCall(VecDestroy(&eig_vec));

    // PetscCall(PetscFree(arr_eig_vec));
    // PetscCall(PetscFree(idxm));

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
    // PetscCall(KSPSetType(kspCoarse, KSPGMRES));
    PetscCall(KSPSetType(kspCoarse, KSPPREONLY));
    PetscCall(KSPGetPC(kspCoarse, &pcCoarse));
    PetscCall(PCSetType(pcCoarse, PCLU));
    PetscCall(PCFactorSetMatSolverType(pcCoarse, MATSOLVERMKL_CPARDISO));
    PetscCall(KSPSetErrorIfNotConverged(kspCoarse, PETSC_TRUE));
    PetscCall(KSPMonitorSet(kspCoarse, MyMonitor, NULL, NULL));
    PetscCall(KSPSetFromOptions(kspCoarse));
    // 设置一阶smoother
    PetscCall(PCMGGetSmoother(pc, 1, &kspSmoother));
    PetscCall(KSPSetFromOptions(kspSmoother));
    // PetscCall(KSPSetType(kspSmoother, KSPCHEBYSHEV));
    PetscCall(KSPSetTolerances(kspSmoother, PETSC_DEFAULT, PETSC_DEFAULT,
                               PETSC_DEFAULT, 1));
    PetscCall(KSPMonitorSet(kspSmoother, MyMonitor, NULL, NULL));
    PetscCall(KSPGetPC(kspSmoother, &pcSmoother));
    PetscCall(PCSetType(pcSmoother, PCBJACOBI));
    // PetscCall(KSPSetErrorIfNotConverged(kspSmoother, PETSC_TRUE));
    // 设置Prolongation
    PetscCall(PCMGSetInterpolation(pc, 1, R));
    PetscCall(
        PCShellSetName(pc, "2levels-MG-via-GMsFEM-with-velocity-elimination"));

  } else {
    PetscCall(PCSetType(pc, PCGAMG));
  }

  PetscCall(MatDiagonalSet(C, b, INSERT_VALUES));
  // PetscCall(MatShift(C, 1e-10));
  // PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  // PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  Vec x;
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(x, 0.0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Starting to solve...\n"));
  PetscCall(KSPMonitorSet(ksp, MyMonitor, NULL, NULL));
  PetscCall(KSPSolve(ksp, b, x));
  KSPConvergedReason reason;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  int num;
  PetscCall(KSPGetIterationNumber(ksp, &num));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of iterations: %d\n", num));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged reason: %d\n", reason));

  PetscCall(SlepcFinalize());
}