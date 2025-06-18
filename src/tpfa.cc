#include "fem/bilinearform.hpp"
#include "fem/coefficient.hpp"
#include "fem/fespace.hpp"
#include "mfem.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>

class CustomBilinearIntegrator : public mfem::BilinearFormIntegrator {
private:
  mfem::Coefficient *k; // 空间依赖的系数（可能是一个 GridFunction）
public:
  // 构造函数
  CustomBilinearIntegrator(mfem::Coefficient *k_) : k(k_) {}

  // 实现组装元素矩阵的函数
  virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                     mfem::ElementTransformation &Trans,
                                     mfem::DenseMatrix &elmat) override {
    // P0元：每个单元上基函数为常数1，自由度为1
    int dof = el.GetDof();   // 应为1（P0元每个单元1个自由度）
    elmat.SetSize(dof, dof); // 元素矩阵为1x1

    // 获取当前单元的测度（面积）
    double measure = Trans.Weight();

    // 假设双线性形式为 \int u v dx
    // 对于P0元，基函数在单元内为1，因此元素矩阵为单元面积
    elmat(0, 0) = measure; // 1x1矩阵，值为单元面积
  }
};

int main() {

  const char *mesh_file = "../../data/structured.mesh";
  mfem::Device device("cpu");
  device.Print();

  mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 0);
  int dim = mesh->Dimension();

  mfem::FiniteElementCollection *hdiv_coll(new mfem::RT_FECollection(0, dim));
  mfem::FiniteElementSpace *R_space =
      new mfem::FiniteElementSpace(mesh, hdiv_coll);

  mfem::Array<int> boundary_dofs;
  R_space->GetBoundaryTrueDofs(boundary_dofs);
  std::cout << "R_space size: " << R_space->GetVSize() << "\n";
  std::cout << "Boundary dofs size: " << boundary_dofs.Size() << "\n";
  for(int i = 0; i < boundary_dofs.Size(); ++i) {
    std::cout << "Boundary dof " << i << ": " << boundary_dofs[i] << "\n";
  }


  return 0;
}