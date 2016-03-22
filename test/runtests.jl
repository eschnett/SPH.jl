using SPH
using Base.Test

# Test initial conditions
const idpars = SPH.InitialBoxParameters(0.0,
                                        -1, -1, -1,
                                        +1, +1, +1,
                                        2, 2, 2)
p = SPH.initial(idpars)
@test p.nparts == 8
@test p.time == 0
@test p.id == [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]
@test p.posx == [-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5]
@test p.posy == [-0.5,-0.5,0.5,0.5,-0.5,-0.5,0.5,0.5]
@test p.posz == [-0.5,-0.5,-0.5,-0.5,0.5,0.5,0.5,0.5]
@test p.vol == [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
@test p.mass == [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]
@test p.velx == [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
@test p.vely == [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
@test p.velz == [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
@test p.uint == [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

# Test EOS
press = SPH.eos(p)
@test press == [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]

# Test kernel
const sphpars = SPH.SPHParameters(1.5,
                                  -2.0, -2.0, -2.0,
                                  +2.0, +2.0, +2.0)
@test SPH.kernel_nonzero(sphpars, 0.0, 0.0, 0.0)
@test SPH.kernel_nonzero(sphpars, 1.0, 0.0, 0.0)
@test SPH.kernel_nonzero(sphpars, 1.499, 0.0, 0.0)
@test !SPH.kernel_nonzero(sphpars, 1.501, 0.0, 0.0)
for r in rand(0.0:1.0e-8:+2.0, 20)
    w = SPH.kernel(sphpars, r, 0.0, 0.0)
    @test SPH.kernel_nonzero(sphpars, r, 0.0, 0.0) == (w!=0)
    @test SPH.kernel(sphpars, r, 0.0, 0.0) == w
    @test SPH.kernel(sphpars, 0.0, r, 0.0) == w
    @test SPH.kernel(sphpars, 0.0, 0.0, r) == w
    @test SPH.kernel(sphpars, -r, 0.0, 0.0) == w
    @test SPH.kernel(sphpars, 0.0, -r, 0.0) == w
    @test SPH.kernel(sphpars, 0.0, 0.0, -r) == w
    wx,wy,wz = SPH.grad_kernel(sphpars, r, 0.0, 0.0)
    @test SPH.kernel_nonzero(sphpars, r, 0.0, 0.0) == (wx!=0)
    @test wy == 0
    @test wz == 0
    @test SPH.grad_kernel(sphpars, r, 0.0, 0.0) == (wx,0.0,0.0)
    @test SPH.grad_kernel(sphpars, 0.0, r, 0.0) == (0.0,wx,0.0)
    @test SPH.grad_kernel(sphpars, 0.0, 0.0, r) == (0.0,0.0,wx)
    @test SPH.grad_kernel(sphpars, -r, 0.0, 0.0) == (-wx,0.0,0.0)
    @test SPH.grad_kernel(sphpars, 0.0, -r, 0.0) == (0.0,-wx,0.0)
    @test SPH.grad_kernel(sphpars, 0.0, 0.0, -r) == (0.0,0.0,-wx)
end

# Test interation search
iacs = SPH.interactions(sphpars, p)
@test iacs.niacs == 48
@test iacs.iacs[1:14] == [SPH.Interaction(1,7),
                          SPH.Interaction(1,6),
                          SPH.Interaction(1,5),
                          SPH.Interaction(1,4),
                          SPH.Interaction(1,3),
                          SPH.Interaction(1,2),
                          SPH.Interaction(2,8),
                          SPH.Interaction(2,6),
                          SPH.Interaction(2,5),
                          SPH.Interaction(2,4),
                          SPH.Interaction(2,3),
                          SPH.Interaction(2,1),
                          SPH.Interaction(3,8),
                          SPH.Interaction(3,7)]

# Test RHS calculation
rhs = SPH.rhs(sphpars, p, press, iacs)
@test rhs.nparts == p.nparts
@test rhs.time == 1
@test rhs.id == p.id
@test rhs.posx == p.velx
@test rhs.posy == p.vely
@test rhs.posz == p.velz
@test rhs.mass == [0,0,0,0,0,0,0,0]

# TODO: Test state

# TODO: Test ODE integrator
