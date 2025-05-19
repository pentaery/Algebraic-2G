#include "matCPU.hh"
#include "petsc.h"
#include "petscsys.h"
#include <petscis.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <vector>

int main(int argc, char *argv[]) {

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    Mat A;

    int nrows, nnz;
    std::vector<PetscScalar> values(1);
    std::vector<PetscInt> col_indices(1);
    std::vector<PetscInt> row_ptr(1);

    readMat(&nrows, &nnz, row_ptr, col_indices, values);

    PetscCall(MatCreateMPIAdj(PETSC_COMM_WORLD, nrows, nrows, row_ptr.data(), col_indices.data(), NULL, &A));

    MatPartitioning part;
    PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD, &part));
    PetscCall(MatPartitioningSetAdjacency(part, A));
    PetscCall(MatPartitioningSetFromOptions(part));// Set number of partitions
    IS is;
    PetscCall(MatPartitioningApply(part, &is));

    PetscCall(ISView(is, PETSC_VIEWER_STDOUT_WORLD));


    // PetscInt row_index[5] = {0,3,5,7,9};
    // PetscInt col_index[9] = {0,1,2,0,1,0,2,1,3};
    // PetscScalar values[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, 4, 4, row_index, col_index, values, &A));

    // PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    // PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    // PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

    // IS is;
    // PetscInt idx[4]={0,2,1,3};
    // PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 4, idx, PETSC_COPY_VALUES, &is));
    // Mat B;
    // PetscCall(MatPermute(A, is, is, &B));
    // PetscCall(MatView(B, PETSC_VIEWER_STDOUT_WORLD));
    

    // PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
}