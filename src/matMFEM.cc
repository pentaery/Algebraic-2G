// #include "fem/coefficient.hpp"
// #include "fem/gridfunc.hpp"
#include "matCPU.hh"
#include <cmath>
// #include "mfem.hpp"

double coefficient_func(const mfem::Vector &x) {

  // double x_min = 0.33, x_max = 0.66;
  int dim = x.Size(); // 获取维度（2 或 3）
  bool in_region = true;
  for (int i = 0; i < dim; ++i) {
    // if (std::fmod(x[i] * 4, 2) < 1.0) {
    //   in_region = false;
    // }

    if ((x[i] > 0.1 && x[i] < 0.2) || (x[i] > 0.3 && x[i] < 0.4) ||
        (x[i] > 0.6 && x[i] < 0.7) || (x[i] > 0.8 && x[i] < 0.9)) {
      in_region = false;
    }
  }
  return in_region ? 1e6 : 1.0;
}

void ComputeTranspose(const mfem::SparseMatrix &A, mfem::SparseMatrix &At) {
  // 获取原始矩阵的维度
  int num_rows = A.Height();
  int num_cols = A.Width();

  // 创建转置矩阵 (列数为原始矩阵的行数，行数为原始矩阵的列数)
  At = mfem::SparseMatrix(num_cols, num_rows);

  // 获取原始矩阵的 CSR 数据
  const int *I = A.GetI();          // 行指针
  const int *J = A.GetJ();          // 列索引
  const double *Data = A.GetData(); // 非零值

  // 遍历原始矩阵的非零元素
  for (int i = 0; i < num_rows; ++i) {
    for (int j_ptr = I[i]; j_ptr < I[i + 1]; ++j_ptr) {
      int j = J[j_ptr];
      double value = Data[j_ptr];
      At.Add(j, i, value); // 交换行和列
    }
  }

  // 完成转置矩阵的构造
  At.Finalize();
}

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

class DiagonalMassIntegrator : public mfem::BilinearFormIntegrator {
private:
  mfem::Coefficient &k; // 空间依赖的系数（可能是一个 GridFunction）
                        // 存储每个元素的 k 值（分片常数）

public:
  DiagonalMassIntegrator(mfem::Coefficient &k_) : k(k_) {}

  virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                     mfem::ElementTransformation &Trans,
                                     mfem::DenseMatrix &elmat) override {
    int dof = el.GetDof(); // RT0: dof = 边数（2D 三角形为 3）
    int dim = el.GetDim();
    if (dof <= 0 || dim <= 0) {
      throw std::runtime_error("Invalid element dimensions or DOFs");
    }

    elmat.SetSize(dof, dof);
    elmat = 0.0;

    // 为每个自由度（边）计算 k 的倒数平均值
    mfem::Vector k_avg(dof); // 存储每个自由度的 k_avg
    k_avg = 0.0;
    for (int j = 0; j < dof; j++) {
      double k_current = k.Eval(Trans, mfem::IntegrationPoint());
      k_avg(j) = 1.0 / (2 * k_current); // 转换为 k_avg
    }

    // 使用默认积分规则（RT0 基函数为常数，order=0 足够）
    const mfem::IntegrationRule *ir = IntRule;
    if (!ir) {
      int order = 0; // RT0 基函数是常数
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);
    }

    double area_or_volume = 0.0;

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      area_or_volume += ip.weight * Trans.Weight();
    }
    mfem::DenseMatrix vshape(dof, dim);
    mfem::Vector vshape_sq(dof);
    for (int j = 0; j < dof; ++j) {
      elmat(j, j) += area_or_volume * k_avg(j);
    }
  }
};

class CustomRT0P0Integrator : public mfem::BilinearFormIntegrator {
private:
  mfem::Mesh *mesh;

public:
  CustomRT0P0Integrator(mfem::Mesh *mesh_) : mesh(mesh_) {}

  virtual void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe,
                                      const mfem::FiniteElement &test_fe,
                                      mfem::ElementTransformation &Trans,
                                      mfem::DenseMatrix &elmat) override {
    int trial_dof = trial_fe.GetDof(); // RT0 的自由度数
    int test_dof = test_fe.GetDof();   // P0 的自由度数 (通常为 1)
    int elem_idx = Trans.ElementNo;

    mfem::Array<int> edges, orientations;
    mesh->GetElementEdges(elem_idx, edges, orientations);

    // 初始化元素矩阵
    elmat.SetSize(test_dof, trial_dof);
    elmat = 0.0;

    // 获取网格维度
    int dim = mesh->Dimension();

    // 存储边长（2D）或面积（3D）
    mfem::Vector measure(edges.Size());

    if (dim == 2) {
      // 2D: Compute edge lengths
      mfem::Array<int> edges, orientations;
      mesh->GetElementEdges(elem_idx, edges, orientations);

      const mfem::IntegrationRule *ir = &mfem::IntRules.Get(
          mfem::Geometry::SEGMENT, 1); // 1D integration for edges

      for (int i = 0; i < edges.Size(); i++) {
        mfem::ElementTransformation *edge_trans =
            mesh->GetEdgeTransformation(edges[i]);
        double meas = 0.0;
        for (int j = 0; j < ir->GetNPoints(); j++) {
          const mfem::IntegrationPoint &ip = ir->IntPoint(j);
          edge_trans->SetIntPoint(&ip);
          meas += ip.weight * edge_trans->Weight();
        }
        measure(i) = meas;
      }
    } else if (dim == 3) {
      // 3D: Compute face areas
      mfem::Array<int> faces, orientations;
      mesh->GetElementFaces(elem_idx, faces, orientations);

      for (int i = 0; i < faces.Size(); i++) {
        // Get face geometry and transformation
        int face_idx = faces[i];
        mfem::Geometry::Type face_geom = mesh->GetFaceGeometry(face_idx);
        mfem::ElementTransformation *face_trans =
            mesh->GetFaceTransformation(face_idx);

        // Select integration rule based on face geometry
        const mfem::IntegrationRule *ir =
            &mfem::IntRules.Get(face_geom, 1); // 2D integration for faces

        double meas = 0.0;
        for (int j = 0; j < ir->GetNPoints(); j++) {
          const mfem::IntegrationPoint &ip = ir->IntPoint(j);
          face_trans->SetIntPoint(&ip);
          meas += ip.weight * face_trans->Weight();
        }
        measure(i) = meas;
      }
    }

    for (int i = 0; i < trial_dof; i++) {
      for (int j = 0; j < test_dof; j++) {
        elmat(j, i) += measure(i);
      }
    }
  }
};

int generateMatMFEM(int *nrows, int *nnz, std::vector<int> &row_ptr,
                    std::vector<int> &col_index, std::vector<double> &values,
                    int meshsize = 30) {

  // 1. Parse command-line options.
  // const char *mesh_file = "../../data/structured3d.mesh";
  int order = 0;
  const char *device_config = "cpu";
  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.

  mfem::Device device(device_config);
  device.Print();

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  // mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 0);
  int dim = 3;
  double sx = 1.0, sy = 1.0, sz = 1.0;
  mfem::Mesh *mesh =
      new mfem::Mesh(meshsize, meshsize, meshsize, mfem::Element::HEXAHEDRON,
                     true, sx, sy, sz);

  // int dim = mesh->Dimension();

  // 5. Define a finite element space on the mesh. Here we use the
  //    Raviart-Thomas finite elements of the specified order.
  mfem::FiniteElementCollection *hdiv_coll(
      new mfem::RT_FECollection(order, dim));
  mfem::FiniteElementCollection *l2_coll(new mfem::L2_FECollection(order, dim));

  mfem::FiniteElementSpace *R_space =
      new mfem::FiniteElementSpace(mesh, hdiv_coll);
  mfem::FiniteElementSpace *W_space =
      new mfem::FiniteElementSpace(mesh, l2_coll);

  mfem::Array<int> boundary_dofs;
  R_space->GetBoundaryTrueDofs(boundary_dofs);

  std::cout << "***********************************************************\n";
  std::cout << "dim(R) = " << R_space->GetVSize() << "\n";
  std::cout << "dim(W) = " << W_space->GetVSize() << "\n";
  std::cout << "***********************************************************\n";

  // 7. Define the coefficients of the PDE.
  // mfem::FunctionCoefficient k_coeff(coefficient_func);
  mfem::ConstantCoefficient k_coeff(1.0);

  // 9. Assemble the finite element matrices for the Darcy operator
  //
  //                            D = [ M  B^T ]
  //                                [ B   0  ]
  //     where:
  //
  //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
  //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
  mfem::BilinearForm *mVarf(new mfem::BilinearForm(R_space));
  mfem::MixedBilinearForm *bVarf(new mfem::MixedBilinearForm(R_space, W_space));

  mVarf->AddDomainIntegrator(new DiagonalMassIntegrator(k_coeff));
  mVarf->Assemble();
  mVarf->Finalize();

  // bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
  bVarf->AddDomainIntegrator(
      new CustomRT0P0Integrator(mesh)); // 自定义的 RT0-P0 积分器
  bVarf->Assemble();
  bVarf->Finalize();

  mfem::SparseMatrix &M(mVarf->SpMat());
  mfem::Vector diag;
  M.GetDiag(diag);

  mfem::SparseMatrix &B(bVarf->SpMat());
  for (int i = 0; i < boundary_dofs.Size(); i++) {
    M(boundary_dofs[i], boundary_dofs[i]) = 0.0;
  }

  mfem::SparseMatrix BT;
  ComputeTranspose(B, BT);
  mfem::SparseMatrix *C = Mult(B, M);
  mfem::SparseMatrix *A = Mult(*C, BT);
  //   FINAL->Print(std::cout);

  const int *i = A->GetI();            // row pointers
  const int *j = A->GetJ();            // column indices
  const double *a_data = A->GetData(); // values
  *nnz = A->NumNonZeroElems();         // number of non-zero elements
  *nrows = A->Height();
  row_ptr.resize(*nrows + 1);
  col_index.resize(*nnz);
  values.resize(*nnz);
  std::copy(i, i + *nrows + 1, row_ptr.begin());
  std::copy(j, j + *nnz, col_index.begin());
  std::copy(a_data, a_data + *nnz, values.begin());

  sortCSRRows(*nrows, *nnz, row_ptr.data(), col_index.data(), values.data());

  // 17. Free the used memory.
  delete mVarf;
  delete bVarf;
  delete W_space;
  delete R_space;
  delete l2_coll;
  delete hdiv_coll;
  delete mesh;
  delete C;
  delete A;

  return 0;
}
