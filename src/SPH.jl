module SPH

using HDF5



# Parameters

immutable InitialBoxParameters
    time::Float64
    xmin::Float64; ymin::Float64; zmin::Float64
    xmax::Float64; ymax::Float64; zmax::Float64
    ni::Int; nj::Int; nk::Int
end
const idpars = InitialBoxParameters(0.0,
                                    -0.5, -0.5, -0.5,
                                    +0.5, +0.5, +0.5,
                                    10, 10, 10)

immutable SPHParameters
    hsml::Float64
    xmin::Float64; ymin::Float64; zmin::Float64
    xmax::Float64; ymax::Float64; zmax::Float64
    hsml_1::Float64
    function SPHParameters(hsml,
                           xmin, ymin, zmin,
                           xmax, ymax, zmax)
        new(hsml,
            xmin, ymin, zmin,
            xmax, ymax, zmax,
            1.0 / hsml)
    end
end
const sphpars = SPHParameters(0.3,
                              -2.0, -2.0, -2.0,
                              +2.0, +2.0, +2.0)



immutable SimulationParameters
    tmax::Float64
    dt::Float64
end
const simpars = SimulationParameters(1.0,
                                     0.01)



type Particles
    nparts::Int
    time::Float64

    id::Vector{Float64}
    posx::Vector{Float64}
    posy::Vector{Float64}
    posz::Vector{Float64}
    vol::Vector{Float64}
    mass::Vector{Float64}
    velx::Vector{Float64}
    vely::Vector{Float64}
    velz::Vector{Float64}
    uint::Vector{Float64}

    Particles() = new(0,
                      0.0,
                      Float64[],
                      Float64[],Float64[],Float64[],
                      Float64[],
                      Float64[],
                      Float64[],Float64[],Float64[],
                      Float64[])
end
import Base: resize!
function resize!(p::Particles, np::Int)
    p.nparts = np
    resize!(p.id, np)
    resize!(p.posx, np)
    resize!(p.posy, np)
    resize!(p.posz, np)
    resize!(p.vol, np)
    resize!(p.mass, np)
    resize!(p.velz, np)
    resize!(p.velx, np)
    resize!(p.vely, np)
    resize!(p.uint, np)
end

function axpy(a, x::Vector, y::Vector)
    r = similar(y)
    @assert length(x) == length(y)
    @inbounds @simd for i in eachindex(y)
        r[i] = a * x[i] + y[i]
    end
    r
end

function axpy(a::Float64, x::Particles, y::Particles)
    @assert x.nparts == y.nparts
    p = Particles()
    resize!(p, y.nparts)
    p.time = a * x.time + y.time
    p.id[:] = y.id[:]           # ids are not modified
    p.posx[:] = a * x.posx[:] + y.posx[:]
    p.posy[:] = a * x.posy[:] + y.posy[:]
    p.posz[:] = a * x.posz[:] + y.posz[:]
    p.vol[:] = a * x.vol[:] + y.vol[:]
    p.mass[:] = a * x.mass[:] + y.mass[:]
    p.velx[:] = a * x.velx[:] + y.velx[:]
    p.vely[:] = a * x.vely[:] + y.vely[:]
    p.velz[:] = a * x.velz[:] + y.velz[:]
    p.uint[:] = a * x.uint[:] + y.uint[:]
    p
end

function permute(p::Particles, perm::Vector{Int})
    p1 = Particles()
    resize!(p1, p.nparts)
    p1.time = p.time
    p1.id = p.id[perm]
    p1.posx = p.posx[perm]
    p1.posy = p.posy[perm]
    p1.posz = p.posz[perm]
    p1.vol = p.vol[perm]
    p1.mass = p.mass[perm]
    p1.velx = p.velx[perm]
    p1.vely = p.vely[perm]
    p1.velz = p.velz[perm]
    p1.uint = p.uint[perm]
    p1
end

function sort(p::Particles)
    permute(p, sortperm(p.posz))
end






immutable Interaction
    iaci::Int
    iacj::Int
end

type Interactions
    niacs::Int                  # number of interactions
    iacs::Vector{Interaction}
    maxniacs::Int               # max interactions per particle
    nzeroiacs::Int              # num particles without interaction
    Interactions() = new(0, Interaction[], 0, 0)
end



function kernel_nonzero(par::SPHParameters,
                        dx::Float64, dy::Float64, dz::Float64)
    dr2 = dx^2 + dy^2 + dz^2
    dr2 < par.hsml^2
end

function kernel(par::SPHParameters,
                dx::Float64, dy::Float64, dz::Float64)
    dr2 = dx^2 + dy^2 + dz^2
    if dr2 >= par.hsml^2
        return 0.0
    end
    dr = sqrt(dr2)
    q = dr * par.hsml_1
    if q < 0.5
        w = 8.0/pi * (1.0 - 6.0 * q^2 + 6.0 * q^3)
    else
        w = 16.0/pi * (1.0 - 3.0 * q + 3.0 * q^2 - q^3)
    end
    w * par.hsml_1^3
end

function grad_kernel(par::SPHParameters,
                     dx::Float64, dy::Float64, dz::Float64)
    dr2 = dx^2 + dy^2 + dz^2
    if dr2 >= par.hsml^2
        return 0.0, 0.0, 0.0
    end
    dr = sqrt(dr2)
    q = dr * par.hsml_1
    if q < 0.5
        wr = -48.0/pi * (2.0 * q - 3.0 * q^2)
    else
        wr = -48.0/pi * (1.0 - 2.0 * q + q^2)
    end
    wr *= par.hsml_1^4
    wr_dr = wr / dr
    wr_dr * dx, wr_dr * dy, wr_dr * dz
end



function initial(par::InitialBoxParameters)
    dx = (par.xmax - par.xmin) / par.ni
    dy = (par.ymax - par.ymin) / par.nj
    dz = (par.zmax - par.zmin) / par.nk
    dV = dx * dy * dz
    ncells = par.ni * par.nj * par.nk
    p = Particles()
    resize!(p, ncells)
    p.time = par.time
    for gk in 1:par.nk, gj in 1:par.nj, gi in 1:par.ni
        i = gi-1 + par.ni * (gj-1 + par.nj * (gk-1)) + 1
        p.id[i] = i - 1
        p.posx[i] = par.xmin + (gi - 0.5) * dx
        p.posy[i] = par.ymin + (gj - 0.5) * dy
        p.posz[i] = par.zmin + (gk - 0.5) * dz
        p.vol[i] = dV
        p.mass[i] = 1.0 / p.nparts
        p.velx[i] = 0.0
        p.vely[i] = 0.0
        p.velz[i] = 0.0
        p.uint[i] = 1.0
    end
    p
end



function interactions(par::SPHParameters, p::Particles)
    dx_1 = 1.0 / par.hsml
    dy_1 = 1.0 / par.hsml
    dz_1 = 1.0 / par.hsml
    ni = floor(Int, (par.xmax - par.xmin) * dx_1) + 1
    nj = floor(Int, (par.ymax - par.ymin) * dy_1) + 1
    nk = floor(Int, (par.zmax - par.zmin) * dz_1) + 1
    ncells = ni * nj * nk

    grid = Array{Int}(ni, nj, nk)
    fill!(grid, 0)
    next = Vector{Int}(p.nparts)
    for i=1:p.nparts
        gi = floor(Int, (p.posx[i] - par.xmin) * dx_1) + 1
        gj = floor(Int, (p.posy[i] - par.ymin) * dy_1) + 1
        gk = floor(Int, (p.posz[i] - par.zmin) * dz_1) + 1
        @assert 0 < gi <= ni
        @assert 0 < gj <= nj
        @assert 0 < gk <= nk
        next[i] = grid[gi,gj,gk]
        grid[gi,gj,gk] = i
    end

    maxnparts = 0               # maximum particles per cell
    nzeroparts = 0              # num cells without particle
    for i in grid
        nparts = 0
        while i > 0
            nparts += 1
            i = next[i]
        end
        maxnparts = max(maxnparts, nparts)
        nzeroparts += nparts == 0
    end

    iacs = Interactions()
    for i=1:p.nparts
        gi = floor(Int, (p.posx[i] - par.xmin) * dx_1) + 1
        gj = floor(Int, (p.posy[i] - par.ymin) * dy_1) + 1
        gk = floor(Int, (p.posz[i] - par.zmin) * dz_1) + 1
        simin = max(1, gi-1)
        sjmin = max(1, gj-1)
        skmin = max(1, gk-1)
        simax = min(ni, gi+1)
        sjmax = min(nj, gj+1)
        skmax = min(nk, gk+1)
        oldniacs = iacs.niacs
        for sk=skmin:skmax, sj=sjmin:sjmax, si=simin:simax
            j = grid[si,sj,sk]
            while j > 0
                if j != i
                    xij = p.posx[i] - p.posx[j]
                    yij = p.posy[i] - p.posy[j]
                    zij = p.posz[i] - p.posz[j]
                    if kernel_nonzero(par, xij, yij, zij)
                        iacs.niacs += 1
                        push!(iacs.iacs, Interaction(i,j))
                    end
                end
                j = next[j]
            end
        end
        niacs_i = iacs.niacs - oldniacs
        iacs.maxniacs = max(iacs.maxniacs, niacs_i)
        iacs.nzeroiacs += niacs_i == 0
    end

    println("Interaction statistics:")
    println("    grid cells:                        $ncells")
    println("    particles:                         $(p.nparts)")
    println("    interactions:                      $(iacs.niacs)")
    println("    average interactions per particle: " *
            "$(round(iacs.niacs / p.nparts, 1))")
    println("    maximum interactions per particle: $(iacs.maxniacs)")
    println("    particles without interactions:    " *
            "$(iacs.nzeroiacs) " *
            "($(round(iacs.nzeroiacs / p.nparts * 100.0, 2))%)")
    println("    average particles per cell:        " *
            "$(round(p.nparts / ncells, 2))")
    println("    maximum particles per cell:        $(maxnparts)")
    println("    cells without particles:           " *
            "$(nzeroparts) ($(round(nzeroparts / ncells * 100.0, 2))%)")

    iacs
end



function eos(p::Particles)
    press = Vector{Float64}(p.nparts)
    for i = 1:p.nparts
        press[i] = p.mass[i] / p.vol[i] * p.uint[i]
    end
    press
end



function rhs(par::SPHParameters,
             p::Particles, press::Vector{Float64}, iacs::Interactions)
    rhs = Particles()
    resize!(rhs, p.nparts)

    rhs.time = 1.0

    for i = 1:p.nparts
        rhs.id[i] = p.id[i]     # ids are not modified
        rhs.posx[i] = p.velx[i]
        rhs.posy[i] = p.vely[i]
        rhs.posz[i] = p.velz[i]
        rhs.vol[i] = 0.0
        rhs.mass[i] = 0.0
        rhs.velx[i] = 0.0
        rhs.vely[i] = 0.0
        rhs.velz[i] = 0.0
        rhs.uint[i] = 0.0
    end

    for iac in iacs.iacs
        i, j = iac.iaci, iac.iacj
        
        xij = p.posx[i] - p.posx[j]
        yij = p.posy[i] - p.posy[j]
        zij = p.posz[i] - p.posz[j]
        gradWij_x, gradWij_y, gradWij_z = grad_kernel(par, xij, yij, zij)

        div_vij = ((p.velx[j] - p.velx[i]) * gradWij_x +
                   (p.vely[j] - p.vely[i]) * gradWij_y +
                   (p.velz[j] - p.velz[i]) * gradWij_z)
        rhs.vol[i] += p.vol[i]^2 * div_vij

        vol_press_ij = (p.vol[i] * p.vol[j] * (press[j] + press[i]) /
                        (p.mass[i] * p.mass[j]))
        rhs.velx[i] -= p.mass[j] * vol_press_ij * gradWij_x
        rhs.vely[i] -= p.mass[j] * vol_press_ij * gradWij_y
        rhs.velz[i] -= p.mass[j] * vol_press_ij * gradWij_z

        vol_press_div_vij = vol_press_ij / 2.0 * div_vij
        rhs.uint[i] -= p.mass[j] * vol_press_div_vij
    end

    rhs
end



type State
    iter::Int
    p::Particles
    press::Vector{Float64}
    iacs::Interactions    
    rhs::Particles
    State() = new()
end

function State(p::Particles, iter::Int)
    state = State()
    state.iter = iter
    state.p = p
    state.press = eos(state.p)
    state.iacs = interactions(state.p)
    state.rhs = rhs(state.p, state.press, state.iacs)
    state
end



function rk2(state0::State)
    iter0 = state0.iter
    p1 = axpy(dt/2, state0.rhs, state0.p)
    state1 = State(p1, iter0)
    p2 = axpy(dt, state1.rhs, state0.p)
    p = sort(p2)
    state = State(p, iter0+1)
    state
end



function output(state::State)
    println("Iteration $(state.iter), time $(state.p.time)")
    filename = "sph.h5"
    # Truncate before first iteration
    mode = state.iter == 0 ? "w" : "r+"
    options = ("libver_bounds",
               (HDF5.H5F_LIBVER_EARLIEST, HDF5.H5F_LIBVER_LATEST))
    h5open(filename, mode, options...) do h5file
        group = @sprintf "%08d" state.iter
        p = state.p
        active = 1:p.nparts
        h5file["$group/time"] = p.time
        h5file["$group/id"] = p.id[active]
        h5file["$group/posx"] = p.posx[active]
        h5file["$group/posy"] = p.posy[active]
        h5file["$group/posz"] = p.posz[active]
        h5file["$group/vol"] = p.vol[active]
        h5file["$group/mass"] = p.mass[active]
        h5file["$group/velx"] = p.velx[active]
        h5file["$group/vely"] = p.vely[active]
        h5file["$group/velz"] = p.velz[active]
        h5file["$group/uint"] = p.uint[active]
    end
end



function main()
    info("SPH")
    p = initial()
    p = sort(p)
    state = State(p, 0)
    output(state)
    while state.p.time < tmax
        state = rk2(state)
        output(state)
    end
    info("Done.")
end

end
