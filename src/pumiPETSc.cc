#include "pumiPETSc.h"
#include <petsc.h>
#include <petscksp.h>
#include <petscsnes.h>
namespace las
{
  // todo : create variant using nnz structure
  las::Mat * createPetscMatrix(int g, int l)
  {
    ::Mat * m = new ::Mat;
    MatCreateAIJ(PETSC_COMM_WORLD,
                 l,l,g,g,
                 sqrt(g), // dnz approximation
                 PETSC_NULL,
                 sqrt(g), // onz approximation
                 PETSC_NULL,
                 m);
    MatSetOption(*m,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);
    return reinterpret_cast<las::Mat*>(m);
  }
  las::Vec * createPetscVector(int g, int l)
  {
    ::Vec * v = new ::Vec;
    VecCreateMPI(PETSC_COMM_WORLD,l,g,v);
    VecSetOption(*v,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);
    return reinterpret_cast<las::Vec*>(v);
  }
  ::Mat * getPetscMat(las::Mat * m)
  {
    return reinterpret_cast< ::Mat* >(m);
  }
  ::Vec * getPetscVec(las::Vec * v)
  {
    return reinterpret_cast< ::Vec* >(v);
  }
  LasOps * getPetscOps()
  {
    static PetscOps * ops = NULL;
    if(ops == NULL)
      ops = new PetscOps;
    return ops;
  }
  class PetscDofOps : public PetscOps
  {
    void zero(las::Mat * m)
    {
      MatZeroEntries(*getPetscMat(m));
    }
    void zero(las::Vec * v)
    {
      VecZeroEntries(*getPetscVec(v));
    }
    void assemble(las::Vec * v, int cnt, int * rws, double * vls)
    {
      VecSetValues(*getPetscVec(v),cnt,rws,vls,ADD_VALUES);
    }
    void assemble(las::Mat * m, int cntr, int * rws, int cntc, int * cls, double * vls)
    {
      MatSetValues(*getPetscMat(m),cntr,rws,cntc,cls,vls,ADD_VALUES);
    }
    void set(las::Vec * v, int cnt, int * rws, double * vls)
    {
      VecSetValues(*getPetscVec(v),cnt,rws,vls,INSERT_VALUES);
    }
    void set(las::Mat * m, int cntr, int * rws, int cntc, int * cls, double * vls)
    {
      MatSetValues(*getPetscMat(m),cntr,rws,cntc,cls,vls,INSERT_VALUES);
    }
    double norm(las::Vec * v)
    {
      double n = 0.0;
      VecNorm(*getPetscVec(v),NORM_2,&n);
      return n;
    }
    double dot(las::Vec * v0, las::Vec * v1)
    {
      double d = 0.0;
      VecDot(*getPetscVec(v0),*getPetscVec(v1),&d);
      return d;
    }
    void axpy(double a, Vec * x, Vec * y)
    {
      VecAXPY(*getPetscVec(y),a,*getPetscVec(x));
    }
    void get(las::Vec * v, double *& vls)
    {
      VecGetArray(*getPetscVec(v),&vls);
    }
    void restore(las::Vec * v, double *& vls)
    {
      VecRestoreArray(*getPetscVec(v),&vls);
    }
  }
  class PetscDefaultSolve : public PetscSolve
  {
  public:
    virtual void solve(las::Mat * k, las::Vec * u, las::Vec * f)
    {
      ::Mat * pk = getPetscMat(k);
      ::Vec * pu = getPetscVec(u);
      ::Vec * pf = getPetscVec(f);
      ::KSP s;
      KSPCreate(PETSC_COMM_WORLD,&s);
      VecAssemblyBegin(*pf);
      VecAssemblyEnd(*pf);
      MatAssemblyBegin(*pk,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(*pk,MAT_FINAL_ASSEMBLY);
      KSPSetOperators(s,*pk,*pk);
      KSPSetFromOptions(s);
      KSPSolve(s,*pf,*pu);
      KSPDestroy(&s);
    }
  };
  LasSolve * createPetscSolver()
  {
    return new PetscLUSolve;
  }
}
