#!/usr/bin/env python3

from fenics import FunctionSpace, TrialFunction, TestFunction, Function
from fenics import DirichletBC, interpolate, File, Expression
from fenics import Constant, dot, grad, dx, derivative, near
from fenics import Point, plot, set_log_level
from fenics import BoxMesh, MeshFunction, SubDomain
from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver
import OpenPelt

import numpy as np
import matplotlib.pylab as plt

tol = 1e-14
set_log_level(20)


class Omega0(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0 + tol


if __name__ == '__main__':

    T         = 100
    num_steps = 100
    dt        = T / num_steps

    rho = Constant(1039)        # Mass density
    C = Constant(3680)          # Specific heat C
    K = Constant(0.565)         # Thermal conductivity
    B0 = Constant(35000)        # Blod perfusion
    A0 = Constant(10000)        # Metabolic rate
    u_blood = Constant(37.0)
    u_basal = Constant(37.0)

    RC = 1.0 / (rho * C)

    # Create mesh and define function space
    length = 0.002
    nx = ny = nz = 10
    mesh = BoxMesh(Point(-length, -length, -length),
                   Point(length, length, length),
                   nx, ny, nz)

    # Define the auxiliary function space
    V0 = FunctionSpace(mesh, 'DG', 0)

    plate_select = OpenPelt.TECPlate.HOT_SIDE
    pC = OpenPelt.tec_plant("TEC", None, OpenPelt.Signal.VOLTAGE, plate_select = plate_select)
    cbs = OpenPelt.circular_buffer_sequencer([50.00], pC.get_ncs())
    pidc = OpenPelt.pid_controller(cbs, 8.00, 0.00, 0.00, plate_select = plate_select)
    pC.set_controller_f(pidc.controller_f)
    pC.run_sim()

    # Define subdomains
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim())
    subdomain0 = Omega0()
    subdomain0.mark(subdomains, 0)
    subdomain1 = pC.subdomain
    subdomain1.mark(subdomains, 1)

    # kappa = K(areas, 0.565, 0.3, degree=0)
    kappa = Function(V0)
    # Brain gray matter sits on TEC
    k_values = [0.565, pC.get_k_val()]
    for cell_no in range(subdomains.array().size):
        subdomain_no = subdomains.array()[cell_no]
        kappa.vector()[cell_no] = k_values[subdomain_no]

    # Define the function space
    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary conditions
    # u_D = [Constant(37.0), Constant(37.0), Constant(37.0)]
    # boundaries = [LeftBoundary(), RightBoundary(), BottomBoundary()]
    u_D = [pC.u_D]
    boundaries = [pC.subdomain]
    bc = [DirichletBC(V, u_D[i], boundaries[i]) for i in range(len(u_D))]
    # Define boundary conditions

    # Define initial value
    u_0 = Constant(23.0)
    # u_0 = Expression('(x[0]>-0.002)&&(x[0]<-0.001)&&\
    #                   (x[1]>-0.002)&&(x[1]<-0.001)?1000:37',
    #                  degree=2)
    u_n = interpolate(u_0, V)

    # Define variational problem
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    F = (u_*v*dx + kappa * RC * dt*dot(grad(u_), grad(v))*dx
         - (u_n + RC*A0*dt*(1.1**(u_ - u_basal))
         - RC*B0*dt*(u_ - u_blood))*v*dx)
    J = derivative(F, u_, u)
    problem = NonlinearVariationalProblem(F, u_, bc, J)
    solver = NonlinearVariationalSolver(problem)

    # Create VTK file for storing the solution
    vtkfile = File("results/solution_3d_subdomains.pvd")

    # Time-stepping
    t = 0
    for u_d in u_D:
        u_d.t = 0
    for n in range(num_steps):

        # Compute solutions
        solver.solve()

        # Save to file and plot solution
        vtkfile << (u_, t)

        # Compute error at vertices
        e_ = np.zeros_like(np.array(u_.vector()))
        for u_d in u_D:
            u_e = interpolate(u_d, V)
            e_ += np.array(u_e.vector()) - np.array(u_.vector())
        err = e_.max()
        # err = np.abs(np.array(u_e.vector()) - np.array(u_.vector())).max()
        print("t = %.2f: error = %.7g" % (t, err))

        # Update previous solutions
        u_n.assign(u_)

        # Update current time
        t += dt
        for u_d in u_D:
            u_d.t = t

        pC.time_update()

    plot(u_)
    plt.show()
