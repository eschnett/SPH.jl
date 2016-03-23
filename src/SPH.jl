module SPH

using HDF5



# Parameters

immutable InitialBoxParameters
    time::Float64
    xmin::Float64; ymin::Float64; zmin::Float64
    xmax::Float64; ymax::Float64; zmax::Float64
    ni::Int; nj::Int; nk::Int
end
# const idpars = InitialBoxParameters(0.0,
#                                     -0.5, -0.5, -0.5,
#                                     +0.5, +0.5, +0.5,
#                                     10, 10, 10)
const idpars = InitialBoxParameters(0.0,
                                    -0.5, -0.5, -0.5,
                                    +0.5, +0.5, +0.5,
                                    100, 100, 100)

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
# const sphpars = SPHParameters(0.3,
#                               -2.0, -2.0, -2.0,
#                               +2.0, +2.0, +2.0)
const sphpars = SPHParameters(0.03,
                              -2.0, -2.0, -2.0,
                              +2.0, +2.0, +2.0)



immutable SimulationParameters
    tmax::Float64
    dt::Float64
    outfile::AbstractString
    outevery::Int
end
# const simpars = SimulationParameters(1.0,
#                                      0.01,
#                                      "sph", 10)
const simpars = SimulationParameters(1.0,
                                     0.001,
                                     "sph", 100)



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

@fastmath function lincom!{T}(r::Vector{T}, v0::Vector{T}, s1::T, v1::Vector{T})
    @assert length(v0) == length(r)
    @assert length(v1) == length(r)
    @inbounds @simd for i in eachindex(r)
        r[i] = v0[i] + s1 * v1[i]
    end
    r
end

function lincom(p0::Particles, s1::Float64, p1::Particles)
    @assert p1.nparts == p0.nparts
    p = Particles()
    resize!(p, p0.nparts)
    p.time = p0.time + s1 * p1.time
    copy!(p.id, p0.id)          # ids are not modified
    lincom!(p.posx, p0.posx, s1, p1.posx)
    lincom!(p.posy, p0.posy, s1, p1.posy)
    lincom!(p.posz, p0.posz, s1, p1.posz)
    lincom!(p.vol, p0.vol, s1, p1.vol)
    lincom!(p.mass, p0.mass, s1, p1.mass)
    lincom!(p.velx, p0.velx, s1, p1.velx)
    lincom!(p.vely, p0.vely, s1, p1.vely)
    lincom!(p.velz, p0.velz, s1, p1.velz)
    lincom!(p.uint, p0.uint, s1, p1.uint)
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



@fastmath function kernel_nonzero(par::SPHParameters,
                                  dx::Float64, dy::Float64, dz::Float64)
    dr2 = dx^2 + dy^2 + dz^2
    dr2 < par.hsml^2
end

@fastmath function kernel(par::SPHParameters,
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

@fastmath function grad_kernel(par::SPHParameters,
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



@fastmath function initial(par::InitialBoxParameters)
    dx = (par.xmax - par.xmin) / par.ni
    dy = (par.ymax - par.ymin) / par.nj
    dz = (par.zmax - par.zmin) / par.nk
    dV = dx * dy * dz
    ncells = par.ni * par.nj * par.nk
    p = Particles()
    resize!(p, ncells)
    p.time = par.time
    @inbounds for gk in 1:par.nk, gj in 1:par.nj
        @simd for gi in 1:par.ni
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
    end
    p
end



@fastmath function interactions(par::SPHParameters, p::Particles)
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
    @inbounds for i=1:p.nparts
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
    @inbounds for i in grid
        nparts = 0
        while i > 0
            nparts += 1
            i = next[i]
        end
        maxnparts = max(maxnparts, nparts)
        nzeroparts += nparts == 0
    end

    iacs = Interactions()
    @inbounds for i=1:p.nparts
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



@fastmath function eos(p::Particles)
    press = Vector{Float64}(p.nparts)
    @inbounds for i = 1:p.nparts
        press[i] = p.mass[i] / p.vol[i] * p.uint[i]
    end
    press
end



@fastmath function rhs(par::SPHParameters,
                       p::Particles, press::Vector{Float64}, iacs::Interactions)
    rhs = Particles()
    resize!(rhs, p.nparts)

    rhs.time = 1.0

    @inbounds for i = 1:p.nparts
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

    @inbounds for iac in iacs.iacs
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
    state.iacs = interactions(sphpars, state.p)
    state.rhs = rhs(sphpars, state.p, state.press, state.iacs)
    state
end



function rk2(par::SimulationParameters, state0::State)
    iter0 = state0.iter
    p1 = lincom(state0.p, par.dt/2, state0.rhs)
    state1 = State(p1, iter0)
    p2 = lincom(state0.p, par.dt, state1.rhs)
    p2 = sort(p2)
    state2 = State(p2, iter0+1)
    state2
end



function output(par::SimulationParameters, state::State, islast::Bool=false)
    println("Iteration $(state.iter), time $(state.p.time)")
    if par.outevery > 0 && (islast || state.iter % par.outevery == 0)
        filename = "$(par.outfile).h5"
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
end



function main()
    info("SPH")
    p = initial(idpars)
    p = sort(p)
    state = State(p, 0)
    while state.p.time < simpars.tmax
        output(simpars, state)
        state = rk2(simpars, state)
        break
    end
    output(simpars, state, true)
    info("Done.")
end

end
