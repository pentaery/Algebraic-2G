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

#define cStar 1.0
#define scalartype double

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
    // std::vector<PetscInt> cluster_sizes(nprocs, 0);
    for (int i = 0; i < nrows; ++i) {
      idx[i] = cluster_sizes[part[i]];
      cluster_sizes[part[i]]++;
    }

    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < part[i]; ++j) {
        idx[i] += cluster_sizes[j];
      }
    }

    // 打印每个cluster的大小
    //  for (int i = 0; i < nprocs; ++i) {
    //    std::cout << "Cluster " << i << ": " << cluster_sizes[i] << std::endl;
    //  }

    // for (int i = 0; i < nrows; ++i) {
    //   std::cout << "Row " << i << " is in cluster " << part[i]
    //             << ", index in cluster: " << idx[i] << std::endl;
    // }

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nrows, idx.data(),
                              PETSC_COPY_VALUES, &is));
    // PetscCall(ISView(is, PETSC_VIEWER_STDOUT_SELF));
    PetscCall(MatPermute(A, is, is, &B));

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

  Mat A;

  PetscCall(MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, local_rows, local_rows,
                                      PETSC_DECIDE, PETSC_DECIDE, local_row_ptr,
                                      local_col_index, local_values, &A));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  Vec b;
  PetscCall(MatCreateVecs(A, &b, NULL));
  PetscCall(MatGetDiagonal(A, b));

  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
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
    PetscCall(MatGetDiagonalBlock(A, &Ai));
    // PetscCall(MatAssemblyBegin(Ai, MAT_FINAL_ASSEMBLY));
    // PetscCall(MatAssemblyEnd(Ai, MAT_FINAL_ASSEMBLY));
    Vec rowsum;
    PetscCall(MatCreateVecs(Ai, &rowsum, NULL));
    PetscCall(VecSetFromOptions(rowsum));
    PetscCall(MatGetRowSum(Ai, rowsum));
    Vec diagonal;
    PetscCall(MatCreateVecs(Ai, &diagonal, NULL));
    PetscCall(VecSetFromOptions(diagonal));
    PetscCall(MatGetDiagonal(Ai, diagonal));
    PetscCall(VecAXPY(diagonal, -1.0, rowsum));
    PetscCall(MatDiagonalSet(Ai, diagonal, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Ai, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Ai, MAT_FINAL_ASSEMBLY));
    // PetscCall(MatSetOptions(Ai, MAT_SYMMETRIC, PETSC_TRUE));

    Mat Si;
    PetscCall(MatCreateConstantDiagonal(PETSC_COMM_SELF, local_rows, local_rows, PETSC_DETERMINE, PETSC_DETERMINE, 1.0, &Si));
    // PetscCall(MatCreateDiagonal(diagonal, &Si));
    PetscCall(MatAssemblyBegin(Si, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Si, MAT_FINAL_ASSEMBLY));
    // PetscCall(MatSetOption(Si, MAT_SYMMETRIC, PETSC_TRUE));

    Mat R;
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, local_rows, eigennum,
                           PETSC_DEFAULT, PETSC_DEFAULT, eigennum, NULL, 0,
                           NULL, &R));

    EPS eps;
    PetscInt nconv;
    PetscScalar eig_val, *arr_eig_vec;
    Vec eig_vec;
    PetscCall(MatCreateVecs(A, &eig_vec, NULL));
    PetscCall(VecSetFromOptions(eig_vec));
    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    PetscCall(EPSSetOperators(eps, Ai, Si));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    PetscCall(EPSSetDimensions(eps, eigennum, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(MatCreateVecs(Ai, &eig_vec, NULL));
    ST st;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSHIFT));
    PetscCall(EPSSetTarget(eps, -1e-12));
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL));
    PetscCall(EPSSetOptionsPrefix(eps, "epsl1_"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));

    PetscCheck(nconv >= eigennum, PETSC_COMM_SELF, PETSC_ERR_USER,
               "Not enough converged eigenvalues found!");

    PetscInt row_start, row_end;
    PetscCall(MatGetOwnershipRange(A, &row_start, &row_end));
    PetscInt *idxm;
    PetscCall(PetscMalloc1(local_rows, &idxm));
    for (int i = 0; i < local_rows; ++i) {
      idxm[i] = row_start + i;
    }

    for (int j = 0; j < eigennum; ++j) {
      PetscInt idxn = eigennum * rank + j;
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Rank %d, number %d, eigval %.18f\n",
                            rank, j, eig_val));
      PetscCall(VecGetArray(eig_vec, &arr_eig_vec));
      PetscCall(MatSetValues(R, local_rows, idxm, 1, &idxn, arr_eig_vec,
                             INSERT_VALUES));
    }

    PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));

    PetscCall(MatDestroy(&Ai));
    PetscCall(MatDestroy(&Si));
    PetscCall(VecDestroy(&eig_vec));
    // PetscCall(EPSDestroy(&eps));

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
    PetscCall(PCFactorSetMatSolverType(pcCoarse, MATSOLVERSUPERLU_DIST));
    PetscCall(KSPSetErrorIfNotConverged(kspCoarse, PETSC_TRUE));
    // 设置一阶smoother
    PetscCall(PCMGGetSmoother(pc, 1, &kspSmoother));
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

  Vec x;
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPSolve(ksp, b, x));
  KSPConvergedReason reason;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  int num;
  PetscCall(KSPGetIterationNumber(ksp, &num));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of iterations: %d\n", num));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged reason: %d\n", reason));

  // if (rank == 7) {
  //   for (int i = 0; i < local_nnz; ++i) {
  //     std::cout << local_col_index[i] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // // int cluster_sizes[nparts];
  // // for (int i = 0; i < nparts; i++) {
  // //   cluster_sizes[i] = 0;
  // // }
  // // for (int i = 0; i < nrows; i++) {
  // //   cluster_sizes[part[i]]++;
  // // }
  // // for (int i = 0; i < nparts; i++) {
  // //   printf("Cluster %d size: %d\n", i, cluster_sizes[i]);
  // // }

  // std::vector<int> globalTolocal(nrows);
  // std::vector<int> count(nparts, 0);
  // for (int i = 0; i < nrows; ++i) {
  //   globalTolocal[i] = count[part[i]];
  //   count[part[i]]++;
  // }

  // std::vector<std::vector<int>> localToGlobal(nparts);
  // for (int i = 0; i < nparts; ++i) {
  //   localToGlobal[i].resize(count[i]);
  // }
  // for (int i = 0; i < nrows; ++i) {
  //   localToGlobal[part[i]][globalTolocal[i]] = i;
  // }

  // // for (int i = 0; i < nparts; ++i) {
  // //   std::cout << "cluster " << i << ": " << count[i] << std::endl;
  // // }

  // std::vector<Mat> Ai(nparts);
  // std::vector<Mat> Si(nparts);

  // std::vector<std::vector<int>> Ai_row_index(nparts);
  // std::vector<std::vector<int>> Ai_col_index(nparts);
  // std::vector<std::vector<scalartype>> Ai_values(nparts);

  // std::vector<std::vector<int>> Si_row_index(nparts);
  // std::vector<std::vector<int>> Si_col_index(nparts);
  // std::vector<std::vector<scalartype>> Si_values(nparts);

  // for (int i = 0; i < nrows; ++i) {
  //   for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
  //     if (part[i] == part[col_indices[j]]) {
  //       if (col_indices[j] != i) {
  //         Ai_row_index[part[i]].push_back(globalTolocal[i]);
  //         Ai_col_index[part[i]].push_back(globalTolocal[i]);
  //         Ai_values[part[i]].push_back(values[j]);
  //         Ai_row_index[part[i]].push_back(globalTolocal[i]);
  //         Ai_col_index[part[i]].push_back(globalTolocal[col_indices[j]]);
  //         Ai_values[part[i]].push_back(-values[j]);

  //         Si_col_index[part[i]].push_back(globalTolocal[i]);
  //         Si_row_index[part[i]].push_back(globalTolocal[i]);
  //         Si_values[part[i]].push_back(values[j] / cStar / cStar / 2);
  //       } else {
  //         Ai_row_index[part[i]].push_back(globalTolocal[i]);
  //         Ai_col_index[part[i]].push_back(globalTolocal[i]);
  //         Ai_values[part[i]].push_back(values[j]);

  //         Si_col_index[part[i]].push_back(globalTolocal[i]);
  //         Si_row_index[part[i]].push_back(globalTolocal[i]);
  //         Si_values[part[i]].push_back(values[j] / cStar / cStar);
  //       }
  //     }
  //   }
  // }

  // int eigennum = 4;
  // PetscCall(PetscOptionsGetInt(NULL, NULL, "-eigennum", &eigennum, NULL));

  // Mat R;
  // std::vector<int> nonzeros(nparts * eigennum, 0);
  // for (int i = 0; i < nparts; ++i) {
  //   for (int j = 0; j < eigennum; ++j) {
  //     nonzeros[i * eigennum + j] = count[part[i]];
  //   }
  // }
  // PetscCall(MatCreateSeqAIJCUSPARSE(PETSC_COMM_SELF, nparts * eigennum,
  // nrows,
  //                                   0, nonzeros.data(), &R));

  // MatSetOption(R, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  // for (int i = 0; i < nparts; ++i) {
  //   PetscCall(MatCreateSeqAIJFromTriple(
  //       PETSC_COMM_SELF, count[i], count[i], Ai_col_index[i].data(),
  //       Ai_row_index[i].data(), Ai_values[i].data(), &Ai[i],
  //       Ai_values[i].size(), PETSC_FALSE));
  //   PetscCall(MatAssemblyBegin(Ai[i], MAT_FINAL_ASSEMBLY));
  //   PetscCall(MatAssemblyEnd(Ai[i], MAT_FINAL_ASSEMBLY));
  //   PetscCall(MatSetOption(Ai[i], MAT_SYMMETRIC, PETSC_TRUE));

  //   PetscCall(MatConvert(Ai[i], MATSEQAIJCUSPARSE, MAT_INPLACE_MATRIX,
  //   &Ai[i]));

  //   PetscCall(MatCreateSeqAIJFromTriple(
  //       PETSC_COMM_SELF, count[i], count[i], Si_col_index[i].data(),
  //       Si_row_index[i].data(), Si_values[i].data(), &Si[i],
  //       Si_values[i].size(), PETSC_FALSE));
  //   PetscCall(MatAssemblyBegin(Si[i], MAT_FINAL_ASSEMBLY));
  //   PetscCall(MatAssemblyEnd(Si[i], MAT_FINAL_ASSEMBLY));
  //   PetscCall(MatSetOption(Si[i], MAT_SYMMETRIC, PETSC_TRUE));
  //   PetscCall(MatConvert(Si[i], MATSEQAIJCUSPARSE, MAT_INPLACE_MATRIX,
  //   &Si[i]));

  //   // PetscCall(MatView(Si[i], PETSC_VIEWER_STDOUT_WORLD));

  //   EPS eps;
  //   PetscInt nconv;
  //   PetscScalar eig_val, *arr_eig_vec;
  //   Vec eig_vec;
  //   PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, count[i], &eig_vec));
  //   PetscCall(VecSetFromOptions(eig_vec));

  //   PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
  //   PetscCall(EPSSetOperators(eps, Ai[i], Si[i]));
  //   PetscCall(EPSSetProblemType(eps, EPS_GHEP));
  //   PetscCall(EPSSetDimensions(eps, eigennum, PETSC_DEFAULT, PETSC_DEFAULT));
  //   PetscCall(MatCreateVecs(Ai[i], &eig_vec, NULL));
  //   ST st;
  //   PetscCall(EPSGetST(eps, &st));
  //   PetscCall(STSetType(st, STSHIFT));
  //   PetscCall(EPSSetTarget(eps, -1e-12));
  //   PetscCall(EPSSetOptionsPrefix(eps, "epsl1_"));
  //   PetscCall(EPSSetFromOptions(eps));
  //   PetscCall(EPSSolve(eps));
  //   PetscCall(EPSGetConverged(eps, &nconv));

  //   PetscCheck(nconv >= eigennum, PETSC_COMM_SELF, PETSC_ERR_USER,
  //              "Not enough converged eigenvalues found!");

  //   int *col = new int;
  //   for (int j = 0; j < eigennum; ++j) {
  //     *col = i * eigennum + j;
  //     PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
  //     // std::cout << "Eigenvalue: " << eig_val << " ";
  //     PetscCall(VecGetArray(eig_vec, &arr_eig_vec));
  //     PetscCall(MatSetValues(R, 1, col, count[i], localToGlobal[i].data(),
  //                            arr_eig_vec, INSERT_VALUES));
  //   }
  //   // std::cout << std::endl;

  //   PetscCall(MatDestroy(&Ai[i]));
  //   PetscCall(MatDestroy(&Si[i]));
  // }

  // std::cout << "Finish computing eigenvalues!" << std::endl;

  // PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
  // PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));

  // PetscCall(MatTranspose(R, MAT_INPLACE_MATRIX, &R));

  // KSP ksp;
  // PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  // PetscCall(KSPSetOperators(ksp, A, A));
  // PetscCall(KSPSetType(ksp, KSPCG));
  // PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  // PetscCall(
  //     KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT,
  //     PETSC_DEFAULT));

  // PetscCall(KSPSetFromOptions(ksp));

  // Vec rhs;
  // PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, nrows, &rhs));
  // PetscCall(VecSet(rhs, 1.0));

  // Vec x;
  // PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, nrows, &x));
  // PetscCall(VecSet(x, 0.0));

  // PC pc;
  // PetscCall(KSPGetPC(ksp, &pc));

  // // PetscCall(PCSetType(pc, PCGAMG));

  // KSP kspCoarse, kspSmoother;
  // PC pcCoarse, pcSmoother;
  // // 设置二层multigrid
  // PetscCall(PCSetType(pc, PCMG));
  // PetscCall(PCMGSetLevels(pc, 2, NULL));
  // // 设为V-cycle
  // PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  // PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
  // PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
  // // 设置coarse solver
  // PetscCall(PCMGGetCoarseSolve(pc, &kspCoarse));
  // PetscCall(KSPSetType(kspCoarse, KSPPREONLY));
  // PetscCall(KSPGetPC(kspCoarse, &pcCoarse));
  // PetscCall(PCSetType(pcCoarse, PCLU));
  // // PetscCall(PCFactorSetMatSolverType(pcCoarse, MATSOLVERSUPERLU_DIST));
  // PetscCall(KSPSetErrorIfNotConverged(kspCoarse, PETSC_TRUE));
  // // 设置一阶smoother
  // PetscCall(PCMGGetSmoother(pc, 1, &kspSmoother));
  // PetscCall(KSPGetPC(kspSmoother, &pcSmoother));
  // PetscCall(PCSetType(pcSmoother, PCBJACOBI));
  // PetscCall(KSPSetErrorIfNotConverged(kspSmoother, PETSC_TRUE));
  // // 设置Prolongation
  // PetscCall(PCMGSetInterpolation(pc, 1, R));
  // PetscCall(
  //     PCShellSetName(pc, "2levels-MG-via-GMsFEM-with-velocity-elimination"));

  // PetscCall(KSPSolve(ksp, rhs, x));

  // int num;
  // PetscCall(KSPGetIterationNumber(ksp, &num));
  // std::cout << "Number of iterations: " << num << std::endl;

  // KSPConvergedReason reason;
  // PetscCall(KSPGetConvergedReason(ksp, &reason));
  // std::cout << "Converged reason: " << reason << std::endl;

  // PetscCall(MatDestroy(&A));
  // PetscCall(VecDestroy(&x));
  // PetscCall(VecDestroy(&rhs));
  // PetscCall(MatDestroy(&R));
  // PetscCall(KSPDestroy(&ksp));
  // PetscCall(VecDestroy(&rowsum));

  PetscCall(SlepcFinalize());
}