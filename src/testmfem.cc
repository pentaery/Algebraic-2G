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

class DiagonalMassIntegrator : public BilinearFormIntegrator {
private:
  Coefficient &k; // 空间依赖的系数
  Mesh *mesh;     // 网格对象，用于访问共享面和相邻单元
public:
  DiagonalMassIntegrator(Coefficient &k_, Mesh *mesh_) : k(k_), mesh(mesh_) {}

  virtual void AssembleElementMatrix(const FiniteElement &el,
                                     ElementTransformation &Trans,
                                     DenseMatrix &elmat) override {
    int dof = el.GetDof();
    elmat.SetSize(dof, dof);
    elmat = 0.0;

    // 确保是 RT0 元（Raviart-Thomas 最低阶）
    if (el.GetMapType() != FiniteElement::H_DIV) {
      MFEM_ABORT(
          "This integrator is designed for H(div) elements (e.g., RT0).");
    }

    // 获取当前单元索引
    int elem_idx = Trans.ElementNo;

    // 获取单元的几何类型和面数（自由度数 = 面数）
    Geometry::Type geom_type = el.GetGeomType();
    int nfaces = mesh->GetElement(elem_idx)->GetNFaces();

    // 确保自由度数与面数一致
    MFEM_VERIFY(dof == nfaces,
                "Number of DoFs must equal number of faces for RT0.");

    // 遍历单元的每个面（对应一个自由度）
    for (int face = 0; face < nfaces; face++) {
      // 获取面在网格中的全局索引
      int face_idx = mesh->GetElementFace(elem_idx, face);

      // 获取面上的积分规则（面上的高斯积分）
      const IntegrationRule *ir =
          &IntRules.Get(mesh->GetFaceGeometry(face_idx), 1);
      // RT0 元在面上法向分量为常数，积分阶数 1 足以准确计算

      // 获取共享面两侧的单元
      Array<int> elem_pair;
      mesh->GetFaceElements(face_idx, &elem_pair);
      int elem1_idx = elem_pair[0]; // 当前单元
      int elem2_idx =
          (elem_pair.Size() == 2) ? elem_pair[1] : -1; // 相邻单元（若存在）

      // 获取面上的法向量和面积
      FaceElementTransformations *face_trans =
          mesh->GetFaceElementTransformations(face_idx);
      MFEM_VERIFY(face_trans, "Failed to get face transformations.");

      // 遍历面上的积分点
      double face_contrib = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++) {
        const IntegrationPoint &ip = ir->IntPoint(i);
        face_trans->SetIntPoint(&ip);

        // 计算当前单元的 k 值
        ElementTransformation *Trans1 =
            mesh->GetElementTransformation(elem1_idx);
        Trans1->SetIntPoint(&ip); // 映射到单元1的积分点
        double k1 = k.Eval(*Trans1, ip);

        // 计算相邻单元的 k 值（如果存在）
        double k2 = k1; // 默认值，防止无相邻单元
        if (elem2_idx >= 0) {
          ElementTransformation *Trans2 =
              mesh->GetElementTransformation(elem2_idx);
          Trans2->SetIntPoint(&ip); // 映射到单元2的积分点
          k2 = k.Eval(*Trans2, ip);
        }

        // 计算倒数的平均值 k_eff = 1 / (1/k1 + 1/k2)
        double k_eff =
            (k1 > 0.0 && k2 > 0.0) ? 1.0 / (1.0 / k1 + 1.0 / k2) : k1;

        // 计算积分权重（面上的权重）
        double w = ip.weight * face_trans->Weight();

        // RT0 基函数在对应面上的法向分量为 1，其他面为 0
        // 对角项贡献：k_eff * (v_j · n)^2 * w，其中 v_j · n = 1
        face_contrib += w * k_eff;
      }

      // 将贡献加到对角项 elmat(face, face)
      elmat(face, face) += face_contrib;
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

  mVarf->AddDomainIntegrator(new DiagonalMassIntegrator(k));
  mVarf->Assemble();
  mVarf->Finalize();

  bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
  bVarf->Assemble();
  bVarf->Finalize();

  BlockOperator darcyOp(block_offsets);

  TransposeOperator *Bt = NULL;

  SparseMatrix &M(mVarf->SpMat());
  M.Print(std::cout);
  SparseMatrix &B(bVarf->SpMat());
  B *= -1.;
  Bt = new TransposeOperator(&B);

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