//                                MFEM Example 5
//
// Compile with: make ex5
//
// Sample runs:  ex5 -m ../data/square-disc.mesh
//               ex5 -m ../data/star.mesh
//               ex5 -m ../data/star.mesh -pa
//               ex5 -m ../data/beam-tet.mesh
//               ex5 -m ../data/beam-hex.mesh
//               ex5 -m ../data/beam-hex.mesh -pa
//               ex5 -m ../data/escher.mesh
//               ex5 -m ../data/fichera.mesh
//
// Device sample runs:
//               ex5 -m ../data/star.mesh -pa -d cuda
//               ex5 -m ../data/star.mesh -pa -d raja-cuda
//               ex5 -m ../data/star.mesh -pa -d raja-omp
//               ex5 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//
//                                 k*u + grad p = f
//                                 - div u      = g
//
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockOperator class, as
//               well as the collective saving of several grid functions in
//               VisIt (visit.llnl.gov) and ParaView (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "fem/coefficient.hpp"
#include "linalg/operator.hpp"
#include "linalg/sparsemat.hpp"
#include "linalg/vector.hpp"
#include "mfem.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <type_traits>

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

    std::cout << "Processing element " << Trans.ElementNo << std::endl;
    // 为每个自由度（边）计算 k 的倒数平均值
    mfem::Vector k_avg(dof); // 存储每个自由度的 k_avg
    k_avg = 0.0;
    double k_current = k.Eval(Trans, mfem::IntegrationPoint());
    std::cout << "k_current = " << k_current << std::endl;
    for (int j = 0; j < dof; j++) {
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

double coefficient_func(const mfem::Vector &x) {
  double x_min = 0.33, x_max = 0.66;
  int dim = x.Size(); // 获取维度（2 或 3）
  bool in_region = true;

  // 检查每个维度的坐标是否在 [1/3, 2/3] 内
  for (int i = 0; i < dim; i++) {
    if (x[i] < x_min || x[i] > x_max) {
      in_region = false;
      break;
    }
  }

  return in_region ? 1000.0 : 1.0; // 中心区域返回 100，其他返回 1
}

int main(int argc, char *argv[]) {

  // 1. Parse command-line options.
  const char *mesh_file = "../../data/structured3d.mesh";
  int order = 0;
  const char *device_config = "cpu";
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return 1;
  }
  args.PrintOptions(std::cout);

  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  mfem::Device device(device_config);
  device.Print();

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 0);
  int dim = mesh->Dimension();

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

  // 7. Define the coefficients, analytical solution, and rhs of the PDE.
  // mfem::ConstantCoefficient k_coeff(1.0);

  mfem::FunctionCoefficient k_coeff(coefficient_func);

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
  M.Print(std::cout);
  mfem::SparseMatrix &B(bVarf->SpMat());
  // B.Print(std::cout);
  for (int i = 0; i < boundary_dofs.Size(); i++) {
    B.EliminateCol(boundary_dofs[i], mfem::Operator::DIAG_ZERO);
  }

  mfem::SparseMatrix BT;
  ComputeTranspose(B, BT);
  mfem::SparseMatrix *C = Mult(B, M);
  mfem::SparseMatrix *FINAL = Mult(*C, BT);
  // FINAL->Print(std::cout);

  // 17. Free the used memory.
  delete mVarf;
  delete bVarf;
  delete W_space;
  delete R_space;
  delete l2_coll;
  delete hdiv_coll;
  delete mesh;
  delete C;
  delete FINAL;

  return 0;
}
