/*

My own FESpace for linear and quadratic triangular elements.

A fe-space provides the connection between the local reference
element, and the global mesh.

*/


#include <comp.hpp>    // provides FESpace, ...
#include <python_comp.hpp>

#include "myElement.hpp"
#include "myFESpace.hpp"
#include "myDiffOp.hpp"


namespace ngcomp
{
    
  /*
     the MeshAccess object provides information about the finite element mesh,
     see: https://github.com/NGSolve/ngsolve/blob/master/comp/meshaccess.hpp
          
     base class FESpace is here:
     https://github.com/NGSolve/ngsolve/blob/master/comp/fespace.hpp
  */
    
  MyFESpace :: MyFESpace (shared_ptr<MeshAccess> ama, const Flags & flags)
    : FESpace (ama, flags)
  {
    cout << "Constructor of MyFESpace" << endl;
    cout << "Flags = " << flags << endl;

    type = "myfespace";

    secondorder = flags.GetDefineFlag ("secondorder");

    if (!secondorder)
      cout << "You have chosen first order elements" << endl;
    else
      cout << "You have chosen second order elements" << endl;

    // diffops are needed for function evaluation and evaluation of the gradient:
    evaluator[VOL] = make_shared<T_DifferentialOperator<MyDiffOpId>>();
    // evaluator[BND] = make_shared<T_DifferentialOperator<MyDiffOpId>>(); 
    flux_evaluator[VOL] = make_shared<T_DifferentialOperator<MyDiffOpGradient>>();
  }

  DocInfo MyFESpace :: GetDocu()
  {
    auto docu = FESpace::GetDocu();
    docu.short_docu = "My own FESpace.";
    docu.long_docu =
      R"raw_string(My own FESpace provides first and second order triangular elements.
)raw_string";      
      
    docu.Arg("secondorder") = "bool = False\n"
      "  Use second order basis functions";
    return docu;
  }

  void MyFESpace :: Update()
  {
    // some global update:
    cout << "Update MyFESpace, #vert = " << ma->GetNV()
         << ", #edge = " << ma->GetNEdges() << endl;

    // number of vertices
    nvert = ma->GetNV();

    // number of dofs:
    size_t ndof = nvert;
    if (secondorder)
      ndof += ma->GetNEdges();  // num vertics + num edges

    SetNDof (ndof);
  }

  /*
    returns dof-numbers of element with ElementId ei
    element may be a volume element, or boundary element
  */
  void MyFESpace :: GetDofNrs (ElementId ei, Array<DofId> & dnums) const
  {
    dnums.SetSize(0);

    // first dofs are vertex numbers:
    for (auto v : ma->GetElement(ei).Vertices())
      dnums.Append (v);

    if (secondorder)
      {
        // more dofs on edges:
        for (auto e : ma->GetElement(ei).Edges())
          dnums.Append (nvert+e);
      }
  }

  /*
    Allocate finite element class, using custom allocator alloc
  */
  FiniteElement & MyFESpace :: GetFE (ElementId ei, Allocator & alloc) const
  {
    switch (ma->GetElement(ei).GetType())
      {
        case ET_TRIG:
          if (!secondorder)
            return * new (alloc) MyLinearTrig;
          else
            return * new (alloc) MyQuadraticTrig;
        default:
          throw Exception("MyFESpace: Element of type "+ToString(ma->GetElement(ei).GetType()) + 
                               " not available\n");
      }
  }

}


void ExportMyFESpace(py::module m)
{
  using namespace ngcomp;

  cout << "called ExportMyFESpace" << endl;

  ExportFESpace<MyFESpace>(m, "MyFESpace", true)
    .def("GetNVert", &MyFESpace::GetNVert, 
            "return number of vertices")
    ;
}
