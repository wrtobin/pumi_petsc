#include "pumiPETSc.h"
#include <apf.h>
#include <PCU.h>
#include <gmi.h>
#include <cassert>
#include <iostream>

char * mdl_fl = NULL;
char * msh_fl = NULL;
char * petsc_opts = NULL;

bool parse_options(int * ac, char * av[])
{
  if(*ac != 4)
  {
    std::cout << "Usage: " << argv[0] << " [model] [mesh] [petsc_options]" << std::endl;
    return false;
  }
  else
  {
    mdl_fl = av[1];
    msh_fl = av[2];
    petsc_opts = av[3];
  }
}

class HookesIntegrator : public apf::Integrator
{
private:
  las::Mat * K;
  las::Vec * F;
  las::PetscOps * ops;
  apf::DynamicMatrix C;
  apf::DynamicMatrix ke;
  apf::DynamicVector fe;
  apf::MeshElement * me;
  apf::Element * e;
  apf::Field * f;
  int nedofs;
  int nenodes;
  int nfcmps;
public:
  HookesIntegrator(pField fld,
                   Mat * k,
                   Vec * f,
                   int o,
                   double E,
                   double v)
    : apf::Integrator(o)
    , K(k)
    , F(f)
    , ops(las::getPetscDofOps()),
    , C(isotropicLinearElasticityTensor(E,v))
    , ke()
    , fe()
    , me(NULL)
    , e(NULL)
    , f(fld)
    , nedofs(0)
    , nenodes(0)
    , nfcmps(apf::countComponents(fld))
  { }
  void inElement(apf::MeshElement * ME)
  {
    me = ME;
    e = apf::createElement(f,me);
    nenodes = apf::countNodes(e);
    int new_nedofs = nenodes * nfcmps;
    bool realloc = nedofs != new_nedofs;
    nedofs = new_nedofs;
    if(realloc)
    {
      ke.setSize(nedofs,nedofs);
      fe.setSize(nedofs);
    }
    ke.zero();
    fe.zero();
  }
  void atPoint(apf::Vector3 const & p, double w, double dV)
  {
    for(int ii = 0; ii < nenodes; ii++)
    {
      apf::NewArray<apf::Vector3> grads;
      apf::getShapeGrads(e,p,grads);
      apf::DynamicMatrix B(6,nedofs);
      B(0,3*ii)   = grads[ii][0]; // N_(ii,1)
      B(0,3*ii+1) = B(0,3*ii+2) = 0.0;
      B(1,3*ii+1) = grads[ii][1]; // N_(ii,2)
      B(1,3*ii)   = B(1,3*ii+2) = 0.0;
      B(2,3*ii+2) = grads[ii][2]; // N_(ii,3)
      B(2,3*ii)   = B(2,3*ii+1) = 0.0;
      B(3,3*ii)   = grads[ii][1]; // N_(ii,2)
      B(3,3*ii+1) = grads[ii][0]; // N_(ii,1)
      B(3,3*ii+2) = 0.0;
      B(4,3*ii)   = 0.0;
      B(4,3*ii+1) = grads[ii][2]; // N_(ii,3)
      B(4,3*ii+2) = grads[ii][1]; // N_(ii,2)
      B(5,3*ii)   = grads[ii][2]; // N_(ii,3)
      B(5,3*ii+1) = 0.0;
      B(5,3*ii+2) = grads[ii][0]; // N_(ii,1)
    }
    apf::DynamicMatrix kt(nedofs,nedofs);
    apf::DynamicMatrix CB(6,nedofs);
    apf::multiply(C,B,CB);
    apf::DynamicMatrix BT(nedofs,6);
    apf::transpose(B,BT);
    apf::multiply(BT,CB,kt);
    // numerical integration
    kt *= w * dV;
    Ke += kt;

    // TODO : generate any contributions to the force vector
  }
  void outElement()
  {
    int dof_ids[nedofs] = {0};
    // TODO: need to actually get the dof ids for the element in their
    // canonical order
    ops->assemble(K,nedofs,&dof_ids,nedofs,&dof_ids,&ke(0,0));
    ops->assemble(F,nedofs,&dof_ids,&fe(0));
    apf::destroyElement(e);
  }
};

apf::DynamicMatrix isotropicLinearElasticityTensor(double E, double v)
{
  apf::DynamicMatrix result(6,6);
  double lambda = ( v * E ) / ( ( 1 + v ) * ( 1 - 2 * v ) );
  double mu = E / ( 2 * ( 1 + v ) );
  result(0,0) = lambda + (2 * mu);
  result(0,1) = lambda;
  result(0,2) = lambda;
  result(0,3) = result(0,4) = result(0,5) = 0.0;
  result(1,0) = lambda;
  result(1,1) = lambda + (2 * mu);
  result(1,2) = lambda;
  result(1,3) = result(1,4) = result(1,5) = 0.0;
  result(2,0) = lambda;
  result(2,1) = lambda;
  result(2,2) = lambda + (2 * mu);
  result(2,3) = result(2,4) = result(2,5) = 0.0;
  result(3,0) = result(3,1) = result(3,2) = result(3,4) = result(3,5) = 0.0;
  result(3,3) = mu;
  result(4,0) = result(4,1) = result(4,2) = result(4,3) = result(4,5) = 0.0;
  result(4,4) = mu;
  result(5,0) = result(5,1) = result(5,2) = result(5,3) = result(5,4) = 0.0;
  result(5,5) = mu;
  return result;
}

int main(int * ac, char * av[])
{
  int errs = 0;
  parse_options(ac,av);
  // init libs
  MPI_Init(&argc,&argv);
  PCU_Comm_Init();
  PetscInitialize(&argc,&argv,petsc_opts,PETSC_NULL);
  pumi_start();

  pGeom mdl = pumi_geom_load(mdl_fl,"mesh");
  pMesh msh = pumi_mesh_loadAll(mdl,msh_fl);
  int dofs_per_node = 3;
  pField u = pumi_field_create(msh,"u",dofs_per_node);

  // number all the dofs or use an implicit numbering

  // apply/update boundary conditions
  // this may need to be done before/after creating
  // the linear structs depending on whether we are
  // removing homogeneous dirichlet bcs from the linear
  // system

  // fixed dofs should have a negative dof_id

  // create linear structures
  int n = pumi_mesh_getNumOwnEnt(msh,0) * dofs_per_node;
  las::Mat * k = las::createPetscMatrix(-1,n);
  las::Vec * uv = las::createPetscVector(-1,n);
  las::Vec * f = las::createPetscVector(-1,n);

  // preallocate matrix for better performance
  //  (more important in nonlinear problems)

  // loop over mesh and assemble linear structures
  HookesIntegrator integrator(u,k,f,1,3000,0.3);
  integrator.process(msh);

  // solve linear system
  las::PetscSolve * slv = las::createPetscSolve();
  slv->solve(k,uv,f);

  // apply solution
  las::PetscOps * ops = las::getPetscDofOps();
  double * u_dat = NULL;
  ops->get(uv,u_dat);

  // use some dof numbering to index into the u_dat array
  // and set/accumulate field values in the u field

  // delete structures
  pumi_field_delete(u);
  pumi_mesh_delete(msh);
  pumi_geom_delete(mdl);

  // finalize libs
  pumi_finalize();
  PetscFinalize();
  PCU_Comm_Free();
  MPI_Finalize();
  return errs;
}

