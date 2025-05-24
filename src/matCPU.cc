#include "matCPU.hh"
#include <cmath>
#include <iomanip>
#include <petscerror.h>
#include <petscmat.h>

void sortCSRRows(int m, int nnz, int *csrRowPtr, int *csrColInd,
                 double *csrVal) {
  // 遍历每一行
  for (int row = 0; row < m; ++row) {
    // 获取当前行的起始和结束位置
    int start = csrRowPtr[row];
    int end = csrRowPtr[row + 1];
    int row_nnz = end - start; // 当前行的非零元素个数

    if (row_nnz <= 1) {
      // 如果行内元素少于 2 个，无需排序
      continue;
    }

    // 将列索引和值绑定为 pair 进行排序
    std::vector<std::pair<int, double>> row_pairs(row_nnz);
    for (int i = start; i < end; ++i) {
      row_pairs[i - start] = std::make_pair(csrColInd[i], csrVal[i]);
    }

    // 按列索引升序排序
    std::sort(row_pairs.begin(), row_pairs.end(),
              [](const std::pair<int, float> &a,
                 const std::pair<int, float> &b) { return a.first < b.first; });

    // 将排序后的结果写回原始数组
    for (int i = start; i < end; ++i) {
      csrColInd[i] = row_pairs[i - start].first;
      csrVal[i] = row_pairs[i - start].second;
    }
  }
}

// int generateMat(int *nrows, int *nnz, std::vector<int> &row_ptr,
//                 std::vector<int> &col_index, std::vector<double> &values) {
//   // 1. Parse command line options.
//   std::string mesh_file = "../../data/rect.msh";
//   int order = 1;

//   // OptionsParser args(argc, argv);
//   // args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
//   // args.AddOption(&order, "-o", "--order", "Finite element polynomial
//   // degree"); args.ParseCheck();

//   // 2. Read the mesh from the given mesh file, and refine once uniformly.
//   mfem::Mesh mesh(mesh_file);
//   // mesh.UniformRefinement();

//   // 3. Define a finite element space on the mesh. Here we use H1 continuous
//   //    high-order Lagrange finite elements of the given order.
//   mfem::H1_FECollection fec(order, mesh.Dimension());
//   mfem::FiniteElementSpace fespace(&mesh, &fec);
//   std::cout << "Number of unknowns: " << fespace.GetTrueVSize() << std::endl;

//   // 4. For Neumann conditions, we don't need to get boundary DOFs
//   //    Array<int> boundary_dofs;
//   //    fespace.GetBoundaryTrueDofs(boundary_dofs);

//   // 5. Define the solution x as a finite element grid function in fespace.
//   mfem::GridFunction x(&fespace);
//   x = 0.0;

//   // 6. Set up the linear form b(.) corresponding to the right-hand side.
//   mfem::ConstantCoefficient one(1.0);
//   mfem::LinearForm b(&fespace);
//   b.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));

//   // Add Neumann boundary condition (if needed)
//   // ConstantCoefficient neumann_bc(0.0); // Zero flux for this example
//   // b.AddBoundaryIntegrator(new BoundaryLFIntegrator(neumann_bc));

//   b.Assemble();

//   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
//   mfem::BilinearForm a(&fespace);
//   a.AddDomainIntegrator(new mfem::DiffusionIntegrator);
//   a.Assemble();

//   // 8. Form the linear system A X = B. For Neumann problem, we don't eliminate
//   // boundary conditions
//   mfem::SparseMatrix A;
//   mfem::Vector B, X;

//   // For pure Neumann problem, the matrix is singular - we need to fix one DOF
//   // Here we fix the first DOF to zero (arbitrary choice)
//   mfem::Array<int> ess_tdof_list(1);
//   ess_tdof_list[0] = 0; // Fix first DOF to make system non-singular

//   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

//   // Get CSR components
//   const int *i = A.GetI();            // row pointers
//   const int *j = A.GetJ();            // column indices
//   const double *a_data = A.GetData(); // values
//   *nnz = A.NumNonZeroElems();         // number of non-zero elements
//   *nrows = A.Height();
//   row_ptr.resize(*nrows + 1);
//   col_index.resize(*nnz);
//   values.resize(*nnz);
//   std::copy(i, i + *nrows + 1, row_ptr.begin());
//   std::copy(j, j + *nnz, col_index.begin());
//   std::copy(a_data, a_data + *nnz, values.begin());

//   sortCSRRows(*nrows, *nnz, row_ptr.data(), col_index.data(), values.data());
//   return 0;
// }

int readMat(int *nrows, int *nnz, std::vector<int> &row_ptr,
            std::vector<int> &col_index, std::vector<double> &values) {
  std::ifstream in("../../data/A.bin");
  if (!in) {
    std::cerr << "Error: Could not open A.bin for reading!" << std::endl;
    return 1;
  }

  in >> *nrows >> *nnz;
  in.ignore();

  row_ptr.resize(*nrows + 1);
  col_index.resize(*nnz);
  values.resize(*nnz);

  std::string line;
  std::getline(in, line);
  std::istringstream iss_values(line);
  for (int i = 0; i < *nnz; i++) {
    if (!(iss_values >> values[i])) {
      std::cerr << "Error reading values at index " << i << std::endl;
      return 1;
    }
  }

  std::getline(in, line);
  std::istringstream iss_cols(line);
  for (int i = 0; i < *nnz; i++) {
    if (!(iss_cols >> col_index[i])) {
      std::cerr << "Error reading column indices at index " << i << std::endl;
      return 1;
    }
  }

  std::getline(in, line);
  std::istringstream iss_rows(line);
  for (int i = 0; i <= *nrows; i++) {
    if (!(iss_rows >> row_ptr[i])) {
      std::cerr << "Error reading row pointers at index " << i << std::endl;
      return 1;
    }
  }

  in.close();

  return 0;
}

void matDecompose2LM(int *nrows, int *nnz, std::vector<int> &row_ptr,
                     std::vector<int> &col_index, std::vector<double> &values) {
  // Step 1: Calculate row sums and update diagonal elements
  std::vector<double> row_sums(*nrows, 0.0f);

  // Calculate row sums
  for (int i = 0; i < *nrows; i++) {
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    for (int j = row_start; j < row_end; j++) {
      row_sums[i] += values[j];
    }
  }

  // Update diagonal elements
  for (int i = 0; i < *nrows; i++) {
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    for (int j = row_start; j < row_end; j++) {
      if (col_index[j] == i) { // Diagonal element
        values[j] = row_sums[i];
      } else {
        values[j] = -values[j];
      }
    }
  }

  // Step 2: In-place removal of lower triangular elements and scaling

  // int new_nnz = 0;
  // int current_pos = 0;

  // for (int i = 0; i < *nrows; i++) {
  //   int row_start = row_ptr[i];
  //   int row_end = row_ptr[i + 1];
  //   int new_row_start = current_pos;

  //   for (int j = row_start; j < row_end; j++) {
  //     if (col_index[j] >= i) { // Keep upper triangular elements
  //       // Move elements to their new positions
  //       if (current_pos != j) {
  //         col_index[current_pos] = col_index[j];
  //         values[current_pos] = values[j] * -2.0f;
  //       } else {
  //         values[current_pos] *= -2.0f;
  //       }
  //       current_pos++;
  //     }
  //   }

  //   row_ptr[i] = new_row_start;
  //   new_nnz = current_pos;
  // }

  // // Update the last row_ptr entry
  // row_ptr[*nrows] = new_nnz;

  // // Resize the vectors to remove unused space
  // col_index.resize(new_nnz);
  // values.resize(new_nnz);
  // *nnz = new_nnz;
}

// PetscErrorCode formAUX(std::vector<int> &row_ptr, std::vector<int>
// &col_index,
//                        std::vector<T> &values, int nrows, int nnz) {
//   PetscFunctionBegin;

//   Mat A;
//   PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nrows, nrows,
//   row_ptr.data(), col_index.data(), values.data(), &A));

//   PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
//   PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

//   PetscCall(MatConvert(A, MATSEQAIJCUSPARSE, MAT_INPLACE_MATRIX, &A));

//   EPS eps;
//   PetscInt nconv;

//   PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
//   PetscCall(EPSSetOperators(eps, A, NULL));
//   PetscCall(EPSSetProblemType(eps, EPS_HEP));
//   PetscCall(EPSSetDimensions(eps, 4, PETSC_DEFAULT,
//                              PETSC_DEFAULT));
//   // ST st;
//   // PetscCall(EPSGetST(eps, &st));
//   // PetscCall(STSetType(st, STSHIFT));

//   PetscCall(EPSSetTarget(eps, -1e-5));
//   PetscCall(EPSSetOptionsPrefix(eps, "epsl2_"));
//   PetscCall(EPSSetFromOptions(eps));
//   PetscCall(EPSSolve(eps));
//   PetscCall(EPSGetConverged(eps, &nconv));

//   PetscCall(MatDestroy(&A));

//   PetscFunctionReturn(0);
// }