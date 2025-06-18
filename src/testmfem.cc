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

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector &x, Vector &u);
real_t pFun_ex(const Vector &x);
void fFun(const Vector &x, Vector &f);
real_t gFun(const Vector &x);
real_t f_natural(const Vector &x);

void ComputeTranspose(const SparseMatrix &A, SparseMatrix &At) {
  // 获取原始矩阵的维度
  int num_rows = A.Height();
  int num_cols = A.Width();

  // 创建转置矩阵 (列数为原始矩阵的行数，行数为原始矩阵的列数)
  At = SparseMatrix(num_cols, num_rows);

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

class DiagonalMassIntegrator : public BilinearFormIntegrator {
private:
  Coefficient &k;          // 空间依赖的系数（可能是一个 GridFunction）
  Mesh *mesh;              // 网格，用于访问邻居信息
  Array<double> *k_values; // 存储每个元素的 k 值（分片常数）

public:
  DiagonalMassIntegrator(Coefficient &k_, Mesh *mesh_,
                         Array<double> *k_values_ = nullptr)
      : k(k_), mesh(mesh_), k_values(k_values_) {}

  virtual void AssembleElementMatrix(const FiniteElement &el,
                                     ElementTransformation &Trans,
                                     DenseMatrix &elmat) override {
    int dof = el.GetDof(); // RT0: dof = 边数（2D 三角形为 3）
    int dim = el.GetDim();
    if (dof <= 0 || dim <= 0) {
      throw std::runtime_error("Invalid element dimensions or DOFs");
    }

    elmat.SetSize(dof, dof);
    elmat = 0.0;

    // 获取当前元素索引
    int elem_idx = Trans.ElementNo;

    // 为每个自由度（边）计算 k 的倒数平均值
    Vector k_avg(dof); // 存储每个自由度的 k_avg
    k_avg = 0.0;

    // 获取当前元素的所有边
    Array<int> edges, orientations;
    mesh->GetElementEdges(elem_idx, edges, orientations);
    if (edges.Size() != dof) {
      throw std::runtime_error("Number of edges does not match DOFs");
    }

    // 遍历每个自由度（边）

    std::cout << "Processing element " << elem_idx << " with " << dof
              << " DOFs (edges), k = " << k.Eval(Trans, IntegrationPoint())
              << std::endl;
    for (int j = 0; j < dof; j++) {
      int face_idx = edges[j]; // 第 j 个自由度对应的边索引

      // 获取边连接的两个元素
      int elem1, elem2;
      mesh->GetFaceElements(face_idx, &elem1, &elem2);

      // 获取当前元素的 k 值
      double k_current =
          k_values ? (*k_values)[elem_idx] : k.Eval(Trans, IntegrationPoint());
      // // 计算 k_avg
      // double k_inv_avg = 1.0 / k_current;
      // int neighbor_count = 1;
      // if (elem2 >= 0) { // 如果有邻居元素（非边界边）
      //   double k_neighbor = k_values
      //                           ? (*k_values)[elem2]
      //                           :
      //                           k.Eval(*mesh->GetElementTransformation(elem2),
      //                                    IntegrationPoint());
      //   k_inv_avg = 0.5 * (1.0 / k_current + 1.0 / k_neighbor); // 倒数平均
      //   neighbor_count = 2;
      // }
      k_avg(j) = 1.0 / (2 * k_current); // 转换为 k_avg

      std::cout << "Edge " << j << ", k_current: " << k_current
                << " k_avg: " << k_avg(j) << std::endl;
    }

    // 使用默认积分规则（RT0 基函数为常数，order=0 足够）
    const IntegrationRule *ir = IntRule;
    if (!ir) {
      int order = 0; // RT0 基函数是常数
      ir = &IntRules.Get(el.GetGeomType(), order);
    }

    DenseMatrix vshape(dof, dim);
    Vector vshape_sq(dof);

    // 遍历积分点
    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      el.CalcVShape(Trans, vshape);

      double w = ip.weight * Trans.Weight();

      // 计算对角项，使用每个自由度对应的 k_avg
      vshape_sq = 0.0;
      for (int j = 0; j < dof; j++) {
        for (int d = 0; d < dim; d++) {
          vshape_sq(j) += vshape(j, d) * vshape(j, d);
        }
        elmat(j, j) += w * k_avg(j) * vshape_sq(j); // 使用自由度对应的 k_avg
      }
    }
  }
};

int main(int argc, char *argv[]) {
  StopWatch chrono;

  // 1. Parse command-line options.
  const char *mesh_file = "../../data/structured.mesh";
  int order = 0;
  bool pa = false;
  const char *device_config = "cpu";
  bool visualization = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                 "--no-partial-assembly", "Enable Partial Assembly.");
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(device_config);
  device.Print();

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  Mesh *mesh = new Mesh(mesh_file, 1, 0);
  int dim = mesh->Dimension();

  // 5. Define a finite element space on the mesh. Here we use the
  //    Raviart-Thomas finite elements of the specified order.
  FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
  FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

  FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
  FiniteElementSpace *W_space = new FiniteElementSpace(mesh, l2_coll);

  Array<int> boundary_dofs;
  R_space->GetBoundaryTrueDofs(boundary_dofs);

  // 6. Define the BlockStructure of the problem, i.e. define the array of
  //    offsets for each variable. The last component of the Array is the sum
  //    of the dimensions of each block.
  Array<int> block_offsets(3); // number of variables + 1
  block_offsets[0] = 0;
  block_offsets[1] = R_space->GetVSize();
  block_offsets[2] = W_space->GetVSize();
  block_offsets.PartialSum();

  std::cout << "***********************************************************\n";
  std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
  std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
  std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
  std::cout << "***********************************************************\n";

  // 7. Define the coefficients, analytical solution, and rhs of the PDE.
  ConstantCoefficient k(1.0);

  VectorFunctionCoefficient fcoeff(dim, fFun);
  FunctionCoefficient fnatcoeff(f_natural);
  FunctionCoefficient gcoeff(gFun);

  VectorFunctionCoefficient ucoeff(dim, uFun_ex);
  FunctionCoefficient pcoeff(pFun_ex);

  // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
  //    side.  Define the GridFunction u,p for the finite element solution and
  //    linear forms fform and gform for the right hand side.  The data
  //    allocated by x and rhs are passed as a reference to the grid functions
  //    (u,p) and the linear forms (fform, gform).
  MemoryType mt = device.GetMemoryType();
  BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

  LinearForm *fform(new LinearForm);
  fform->Update(R_space, rhs.GetBlock(0), 0);
  fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
  fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
  fform->Assemble();
  fform->SyncAliasMemory(rhs);

  LinearForm *gform(new LinearForm);
  gform->Update(W_space, rhs.GetBlock(1), 0);
  gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
  gform->Assemble();
  gform->SyncAliasMemory(rhs);

  // 9. Assemble the finite element matrices for the Darcy operator
  //
  //                            D = [ M  B^T ]
  //                                [ B   0  ]
  //     where:
  //
  //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
  //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
  BilinearForm *mVarf(new BilinearForm(R_space));
  MixedBilinearForm *bVarf(new MixedBilinearForm(R_space, W_space));

  mVarf->AddDomainIntegrator(new DiagonalMassIntegrator(k, mesh));
  mVarf->Assemble();
  mVarf->Finalize();

  bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
  bVarf->Assemble();
  bVarf->Finalize();

  BlockOperator darcyOp(block_offsets);

  TransposeOperator *Bt = NULL;

  SparseMatrix &M(mVarf->SpMat());
  M.Print(std::cout);
  Vector diag;
  M.GetDiag(diag);

  SparseMatrix &B(bVarf->SpMat());
  for (int i = 0; i < boundary_dofs.Size(); i++) {
    B.EliminateCol(boundary_dofs[i], mfem::Operator::DIAG_ZERO);
  }
  B.Print(std::cout);

  // B *= -1.;
  Bt = new TransposeOperator(&B);
  SparseMatrix BT;
  ComputeTranspose(B, BT);
  // BT.Print(std::cout);

  SparseMatrix *C = Mult(B, M);

  SparseMatrix *FINAL = Mult(*C, BT);
  // FINAL->Print(std::cout);

  darcyOp.SetBlock(0, 0, &M);
  darcyOp.SetBlock(0, 1, Bt);
  darcyOp.SetBlock(1, 0, &B);

  // 10. Construct the operators for preconditioner
  //
  //                 P = [ diag(M)         0         ]
  //                     [  0       B diag(M)^-1 B^T ]
  //
  //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
  //     pressure Schur Complement
  //   SparseMatrix *MinvBt = NULL;
  //   Vector Md(mVarf->Height());

  //   BlockDiagonalPreconditioner darcyPrec(block_offsets);
  //   Solver *invM, *invS;
  //   SparseMatrix *S = NULL;

  //   M.GetDiag(Md);
  //   Md.HostReadWrite();

  //   MinvBt = Transpose(B);

  //   for (int i = 0; i < Md.Size(); i++) {
  //     MinvBt->ScaleRow(i, 1. / Md(i));
  //   }

  //   S = Mult(B, *MinvBt);

  //   invM = new DSmoother(M);

  // #ifndef MFEM_USE_SUITESPARSE
  //   invS = new GSSmoother(*S);
  // #else
  //   invS = new UMFPackSolver(*S);
  // #endif

  //   invM->iterative_mode = false;
  //   invS->iterative_mode = false;

  //   darcyPrec.SetDiagonalBlock(0, invM);
  //   darcyPrec.SetDiagonalBlock(1, invS);

  // 11. Solve the linear system with MINRES.
  //     Check the norm of the unpreconditioned residual.
  int maxIter(1000);
  real_t rtol(1.e-6);
  real_t atol(1.e-10);

  chrono.Clear();
  chrono.Start();
  MINRESSolver solver;
  solver.SetAbsTol(atol);
  solver.SetRelTol(rtol);
  solver.SetMaxIter(maxIter);
  solver.SetOperator(darcyOp);
  // solver.SetPreconditioner(darcyPrec);
  solver.SetPrintLevel(1);
  x = 0.0;
  solver.Mult(rhs, x);
  if (device.IsEnabled()) {
    x.HostRead();
  }
  chrono.Stop();

  if (solver.GetConverged()) {
    std::cout << "MINRES converged in " << solver.GetNumIterations()
              << " iterations with a residual norm of " << solver.GetFinalNorm()
              << ".\n";
  } else {
    std::cout << "MINRES did not converge in " << solver.GetNumIterations()
              << " iterations. Residual norm is " << solver.GetFinalNorm()
              << ".\n";
  }
  std::cout << "MINRES solver took " << chrono.RealTime() << "s.\n";

  // 12. Create the grid functions u and p. Compute the L2 error norms.
  GridFunction u, p;
  u.MakeRef(R_space, x.GetBlock(0), 0);
  p.MakeRef(W_space, x.GetBlock(1), 0);

  int order_quad = max(2, 2 * order + 1);
  const IntegrationRule *irs[Geometry::NumGeom];
  for (int i = 0; i < Geometry::NumGeom; ++i) {
    irs[i] = &(IntRules.Get(i, order_quad));
  }

  real_t err_u = u.ComputeL2Error(ucoeff, irs);
  real_t norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);
  real_t err_p = p.ComputeL2Error(pcoeff, irs);
  real_t norm_p = ComputeLpNorm(2., pcoeff, *mesh, irs);

  std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
  std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";

  // 17. Free the used memory.
  delete fform;
  delete gform;
  // delete invM;
  // delete invS;
  // delete S;
  delete Bt;
  // delete MinvBt;
  delete mVarf;
  delete bVarf;
  delete W_space;
  delete R_space;
  delete l2_coll;
  delete hdiv_coll;
  delete mesh;

  return 0;
}

void uFun_ex(const Vector &x, Vector &u) {
  real_t xi(x(0));
  real_t yi(x(1));
  real_t zi(0.0);
  if (x.Size() == 3) {
    zi = x(2);
  }

  u(0) = -exp(xi) * sin(yi) * cos(zi);
  u(1) = -exp(xi) * cos(yi) * cos(zi);

  if (x.Size() == 3) {
    u(2) = exp(xi) * sin(yi) * sin(zi);
  }
}

// Change if needed
real_t pFun_ex(const Vector &x) {
  real_t xi(x(0));
  real_t yi(x(1));
  real_t zi(0.0);

  if (x.Size() == 3) {
    zi = x(2);
  }

  return exp(xi) * sin(yi) * cos(zi);
}

void fFun(const Vector &x, Vector &f) { f = 0.0; }

real_t gFun(const Vector &x) {
  if (x.Size() == 3) {
    return -pFun_ex(x);
  } else {
    return 0;
  }
}

real_t f_natural(const Vector &x) { return (-pFun_ex(x)); }

double k_function(const Vector &x) {
  double x_coord = x(0);
  double y_coord = x(1);
  // 示例：k 随 x 坐标线性变化
  return 1.0 + 0.5 * x_coord;
}
