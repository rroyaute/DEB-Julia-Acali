import DifferentialEquations as DE
using Plots

# === DEB Model Parameters (Standard notation) ===
Base.@kwdef struct DEBParams
    # Primary parameters
    p_Am::Float64 = 1712.41      # Surface-area-specific max assimilation rate (J/d/cm²)
    # r_pAm_pM = 1.5;   # Ratio between pAm and pM (pM = pAm * r_pAm_pM)
    v::Float64 = 0.018505         # Energy conductance (cm/d)
    κ::Float64 = 0.4373          # Allocation fraction to soma

    p_M::Float64 = 1680       # Volume-specific somatic maintenance (J/d/cm³)
    E_G::Float64 = 6880.33     # Specific cost of structure (J/cm³)
    k_J::Float64 = 0.002793      # Maturity maintenance rate coefficient (1/d)
    κ_R::Float64 = 0.475       # Reproduction efficiency
    
    # Maturity thresholds
    # TO CHANGE
    E_Hb::Float64 = 275.0     # Maturity at birth (J)
    E_Hp::Float64 = 1500.0    # Maturity at puberty (J)
    
    # Food/environment
    # TO CHANGE
    f::Float64 = 1.0          # Scaled functional response (0-1)
end

# === DEB ODE System ===
function deb_ode!(du, u, p::DEBParams, t)
    E, V, E_H, E_R = u
    
    # Unpack frequently used parameters
    (; p_Am, v, κ, p_M, E_G, k_J, κ_R, E_Hb, E_Hp, f) = p
    
    # Structural length (cm)
    L = V^(1/3)
    
    # === Fluxes ===
    # Assimilation (only after birth)
    p_A = (E_H >= E_Hb) ? f * p_Am * L^2 : 0.0
    
    # Mobilization (from reserve)
    p_C = E * (v * E_G * L^2 + p_M * V) / (κ * E + E_G * V)
    
    # Somatic maintenance
    p_S = p_M * V
    
    # Maturity maintenance
    p_J = k_J * E_H
    
    # === Allocation ===
    # Somatic branch (κ fraction)
    p_somatic = κ * p_C
    
    # Growth (what remains after somatic maintenance)
    p_G = max(0.0, p_somatic - p_S)
    
    # Reproductive/maturity branch (1-κ fraction)
    p_repro = (1 - κ) * p_C
    
    # === State equations ===
    # Reserve dynamics
    dE = p_A - p_C
    
    # Structure dynamics
    dV = p_G / E_G
    
    # Maturity dynamics (before puberty) or reproduction buffer (after)
    if E_H < E_Hp
        dE_H = max(0.0, p_repro - p_J)
        dE_R = 0.0
    else
        dE_H = 0.0
        dE_R = κ_R * (p_repro - p_J)
    end
    
    du[1] = dE
    du[2] = dV
    du[3] = dE_H
    du[4] = dE_R
end

# === Initial Conditions & Simulation ===
params = DEBParams()

# Start at birth: small reserve, tiny structure, maturity at birth threshold
E0 = 1.0        # Initial reserve (J)
V0 = 1e-6       # Initial structural volume (cm³)
E_H0 = params.E_Hb  # Start at birth
E_R0 = 0.0

u0 = [E0, V0, E_H0, E_R0]
tspan = (0.0, 365.0)  # One year simulation

prob = DE.ODEProblem(deb_ode!, u0, tspan, params)
sol = DE.solve(prob, DE.Tsit5(), saveat=1.0)

# === Visualization ===
p1 = plot(sol.t, sol[1,:], label="Reserve (E)", ylabel="J", lw=2)
p2 = plot(sol.t, sol[2,:], label="Structure (V)", ylabel="cm³", lw=2)
p3 = plot(sol.t, sol[3,:], label="Maturity (E_H)", ylabel="J", lw=2)
hline!(p3, [params.E_Hp], ls=:dash, label="Puberty threshold", color=:red)
p4 = plot(sol.t, sol[4,:], label="Repro buffer (E_R)", xlabel="Time (d)", ylabel="J", lw=2)

plot(p1, p2, p3, p4, layout=(4,1), size=(700, 600))