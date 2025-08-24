#include "matCPU.hh"
#include <cmath>
#include <petscoptions.h>

double coefficient_func(const mfem::Vector &x) {
  int dim = x.Size();
  int partition = 10, contrast = 6;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-partition", &partition, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-contrast", &contrast, NULL));
  int count = 0;
  for (int i = 0; i < dim; ++i) {
    if (std::fmod(x(i), (double)(2.0 / partition)) <
        ((double)(1.0 / partition))) {
      count++;
    }
  }

  if (count >= dim - 1) {
    return pow(10, -contrast);
  } else {
    return 1.0;
  }
}

class PermeabilityCoefficient : public mfem::Coefficient {
private:
  double high_k;
  int n;

public:
  PermeabilityCoefficient(double high_k_ = 6, int n_ = 1)
      : high_k(high_k_), n(n_) {
    PetscOptionsGetReal(NULL, NULL, "-contrast", &high_k, NULL);
    PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
  }

  virtual double Eval(mfem::ElementTransformation &T,
                      const mfem::IntegrationPoint &ip) {
    // 获取归一化坐标 [0, 1]
    mfem::Vector x;
    T.Transform(ip, x);
    int dim = x.Size(); // 维度（应为 3）

    if (dim != 3) {
      return 1.0; // 只支持 3D
    }

    // 将归一化坐标 [0, 1] 映射到索引 [0, 64*n)
    double idx_x = x(0) * (64.0 * n - 1); // x 坐标对应索引
    double idx_y = x(1) * (64.0 * n - 1); // y 坐标对应索引
    double idx_z = x(2) * (64.0 * n - 1); // z 坐标对应索引

    // MATLAB 索引范围转换为 MFEM 归一化坐标范围（从 1:64 映射到 [0, 63] 再到
    // [0, 64*n-1]） MATLAB 索引 i:j 对应归一化坐标 [(i-1)/63, (j-1)/63] 调整为
    // n 倍网格：[(i-1)/(63*n), (j-1)/(63*n)]
    auto in_range = [](double idx, int start, int end, int n) {
      double start_norm =
          (start - 1.0) / 63.0; // MATLAB 索引 start 转换为归一化坐标
      double end_norm = (end - 1.0) / 63.0; // MATLAB 索引 end 转换为归一化坐标
      double idx_norm = idx / (64.0 * n - 1); // 当前索引转换为归一化坐标
      return idx_norm >= start_norm && idx_norm <= end_norm;
    };

    // 检查是否在高渗透率区域
    if (
        // Region 1: perm(8:55, 30:33, 27:30)
        (in_range(idx_x, 8, 55, n) && in_range(idx_y, 30, 33, n) &&
         in_range(idx_z, 27, 30, n)) ||
        // Region 2: perm(5:9, 45:48, 10:55)
        (in_range(idx_x, 5, 9, n) && in_range(idx_y, 45, 48, n) &&
         in_range(idx_z, 10, 55, n)) ||
        // Region 3: perm(11:13, 45:48, 10:55)
        (in_range(idx_x, 11, 13, n) && in_range(idx_y, 45, 48, n) &&
         in_range(idx_z, 10, 55, n)) ||
        // Region 4: perm(10:14, 21:26, 20:55)
        (in_range(idx_x, 10, 14, n) && in_range(idx_y, 21, 26, n) &&
         in_range(idx_z, 20, 55, n)) ||
        // Region 5: perm(15:19, 22:28, 40:60)
        (in_range(idx_x, 15, 19, n) && in_range(idx_y, 22, 28, n) &&
         in_range(idx_z, 40, 60, n)) ||
        // Region 6: perm(28:33, 42:47, 44:48)
        (in_range(idx_x, 28, 33, n) && in_range(idx_y, 42, 47, n) &&
         in_range(idx_z, 44, 48, n)) ||
        // Region 7: perm(31:34, 32:37, 44:48)
        (in_range(idx_x, 31, 34, n) && in_range(idx_y, 32, 37, n) &&
         in_range(idx_z, 44, 48, n)) ||
        // Region 8: perm(42:48, 12:17, 42:48)
        (in_range(idx_x, 42, 48, n) && in_range(idx_y, 12, 17, n) &&
         in_range(idx_z, 42, 48, n)) ||
        // Region 9: perm(50:54, 20:25, 44:50)
        (in_range(idx_x, 50, 54, n) && in_range(idx_y, 20, 25, n) &&
         in_range(idx_z, 44, 50, n)) ||
        // Region 10: perm(15:20, 11:15, 20:25)
        (in_range(idx_x, 15, 20, n) && in_range(idx_y, 11, 15, n) &&
         in_range(idx_z, 20, 25, n)) ||
        // Region 11: perm(36:40, 21:25, 20:25)
        (in_range(idx_x, 36, 40, n) && in_range(idx_y, 21, 25, n) &&
         in_range(idx_z, 20, 25, n)) ||
        // Region 12: perm(20:25, 38:43, 25:30)
        (in_range(idx_x, 20, 25, n) && in_range(idx_y, 38, 43, n) &&
         in_range(idx_z, 25, 30, n)) ||
        // Region 13: perm(25:30, 44:49, 25:30)
        (in_range(idx_x, 25, 30, n) && in_range(idx_y, 44, 49, n) &&
         in_range(idx_z, 25, 30, n)) ||
        // Region 14: perm(42:46, 42:50, 25:30)
        (in_range(idx_x, 42, 46, n) && in_range(idx_y, 42, 50, n) &&
         in_range(idx_z, 25, 30, n)) ||
        // Region 15: perm(15:25, 22:28, 8:13)
        (in_range(idx_x, 15, 25, n) && in_range(idx_y, 22, 28, n) &&
         in_range(idx_z, 8, 13, n)) ||
        // Region 16: perm(28:33, 42:47, 9:15)
        (in_range(idx_x, 28, 33, n) && in_range(idx_y, 42, 47, n) &&
         in_range(idx_z, 9, 15, n)) ||
        // Region 17: perm(31:35, 42:45, 11:15)
        (in_range(idx_x, 31, 35, n) && in_range(idx_y, 42, 45, n) &&
         in_range(idx_z, 11, 15, n)) ||
        // Region 18: perm(42:46, 12:17, 18:23)
        (in_range(idx_x, 42, 46, n) && in_range(idx_y, 12, 17, n) &&
         in_range(idx_z, 18, 23, n)) ||
        // Region 19: perm(50:56, 20:25, 26:31)
        (in_range(idx_x, 50, 56, n) && in_range(idx_y, 20, 25, n) &&
         in_range(idx_z, 26, 31, n)) ||
        // Region 20: perm(40:45, 10:50, 45:50)
        (in_range(idx_x, 40, 45, n) && in_range(idx_y, 10, 50, n) &&
         in_range(idx_z, 45, 50, n)) ||
        // Region 21: perm(50:54, 10:50, 52:54)
        (in_range(idx_x, 50, 54, n) && in_range(idx_y, 10, 50, n) &&
         in_range(idx_z, 52, 54, n)) ||
        // Region 22: perm(40:45, 10:60, 15:20)
        (in_range(idx_x, 40, 45, n) && in_range(idx_y, 10, 60, n) &&
         in_range(idx_z, 15, 20, n)) ||
        // Region 23: perm(55:60, 15:55, 11:14)
        (in_range(idx_x, 55, 60, n) && in_range(idx_y, 15, 55, n) &&
         in_range(idx_z, 11, 14, n)) ||
        // Region 24: perm(15:45, 10:50, 11:14)
        (in_range(idx_x, 15, 45, n) && in_range(idx_y, 10, 50, n) &&
         in_range(idx_z, 11, 14, n)) ||
        // Region 25: perm(15:50, 10:15, 30:34)
        (in_range(idx_x, 15, 50, n) && in_range(idx_y, 10, 15, n) &&
         in_range(idx_z, 30, 34, n)) ||
        // Region 26: perm(15:50, 10:15, 50:54)
        (in_range(idx_x, 15, 50, n) && in_range(idx_y, 10, 15, n) &&
         in_range(idx_z, 50, 54, n)) ||
        // Region 27: perm(15:30, 50:55, 10:14)
        (in_range(idx_x, 15, 30, n) && in_range(idx_y, 50, 55, n) &&
         in_range(idx_z, 10, 14, n)) ||
        // Region 28: perm(15:50, 50:55, 10:14)
        (in_range(idx_x, 15, 50, n) && in_range(idx_y, 50, 55, n) &&
         in_range(idx_z, 10, 14, n)) ||
        // Region 29: perm(15:25, 52:56, 50:54)
        (in_range(idx_x, 15, 25, n) && in_range(idx_y, 52, 56, n) &&
         in_range(idx_z, 50, 54, n)) ||
        // Region 30: perm(35:50, 52:56, 50:54)
        (in_range(idx_x, 35, 50, n) && in_range(idx_y, 52, 56, n) &&
         in_range(idx_z, 50, 54, n))) {
      return pow(10, high_k); // 高渗透率区域
    }
    return 1.0; // 默认渗透率
  }
};

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

  // 先创建均匀网格
  double sx = 1.0, sy = 1.0, sz = 1.0;
  mfem::Mesh *mesh = new mfem::Mesh(
      mfem::Mesh::MakeCartesian3D(meshsize, meshsize, meshsize,
                                  mfem::Element::HEXAHEDRON, sx, sy, sz, true));

  // 创建非均匀网格：手动调整x、y、z三个方向的节点坐标
  // 所有三个方向都使用1/128和3/128交替的元素长度
  std::vector<double> x_coords, y_coords, z_coords;

  // 生成交替长度的坐标数组（1/128和3/128交替）
  auto generate_alternating_coords = [](int num_elements) {
    std::vector<double> coords;
    coords.push_back(0.0);
    double current_pos = 0.0;
    for (int i = 0; i < num_elements; ++i) {
      // 交替使用1/128和3/128的长度
      double element_length = (i % 2 == 0) ? (1.0 / 128.0) : (3.0 / 128.0);
      current_pos += element_length;
      coords.push_back(current_pos);
    }
    // 归一化到[0, 1]
    double total_length = coords.back();
    for (auto &coord : coords) {
      coord /= total_length;
    }
    return coords;
  };

  x_coords = generate_alternating_coords(meshsize);
  y_coords = generate_alternating_coords(meshsize);
  z_coords = generate_alternating_coords(meshsize);

  // 调整网格节点坐标
  for (int i = 0; i < mesh->GetNV(); ++i) {
    double *vertex = mesh->GetVertex(i);
    int ix = (int)round(vertex[0] * meshsize); // 获取x方向索引
    int iy = (int)round(vertex[1] * meshsize); // 获取y方向索引
    int iz = (int)round(vertex[2] * meshsize); // 获取z方向索引

    if (ix < x_coords.size()) {
      vertex[0] = x_coords[ix]; // 设置新的x坐标
    }
    if (iy < y_coords.size()) {
      vertex[1] = y_coords[iy]; // 设置新的y坐标
    }
    if (iz < z_coords.size()) {
      vertex[2] = z_coords[iz]; // 设置新的z坐标
    }
  }

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
  // mfem::ConstantCoefficient k_coeff(1.0);
  PermeabilityCoefficient k_coeff;

  // 将k_coeff投影到GridFunction，并与网格一起导出到同一个Paraview支持的文件
  mfem::GridFunction k_gridfunc(W_space);
  k_gridfunc.ProjectCoefficient(k_coeff);
  mfem::ParaViewDataCollection dcoll("output", mesh);
  dcoll.RegisterField("Permeability", &k_gridfunc);
  dcoll.SetLevelsOfDetail(1);
  dcoll.SetHighOrderOutput(false);
  dcoll.SetOwnData(false);
  dcoll.Save();

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
  for (int i = 0; i < R_space->GetVSize(); i++) {
    M(i, i) = 1.0 / M(i, i); // 将 M 的对角线元素取倒数
  }
  for (int i = 0; i < boundary_dofs.Size(); i++) {
    M(boundary_dofs[i], boundary_dofs[i]) = 0.0;
  }

  mfem::SparseMatrix BT;
  ComputeTranspose(B, BT);
  mfem::SparseMatrix *C = Mult(B, M);
  mfem::SparseMatrix *A = Mult(*C, BT);
  // A->Print(std::cout);

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
  // delete mesh;

  return 0;
}
