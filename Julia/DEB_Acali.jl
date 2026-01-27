using DifferentialEquations
using Plots

# ============================================================================
# DEB Model for Apporectodea caliginosa (Earthworm)
# Translated from MCSim implementation
# ============================================================================

# === Parameter Structure ===
Base.@kwdef struct DEBEarthwormParams
    # --- Environment ---
    muOM::Float64 = 11700.0       # Energy in 1g organic matter from horse dung (J/g)
    rOM_ClxHorse::Float64 = 0.3   # Ratio of energy: soil OM vs horse dung OM
    
    # --- Energy Assimilation ---
    Fm::Float64 = 0.5             # Max specific searching rate (g/d/g)
    kapX::Float64 = 0.258         # Digestion efficiency of food to reserve (-)
    pAm::Float64 = 1712.41        # {p_Am}, specific assimilation flux (J/d/cm²)
    r_pAm_pM::Float64 = 1.5       # Ratio pAm/pM
    v::Float64 = 0.018505         # Energy conductance (cm/d)
    
    # --- Allocation ---
    kap::Float64 = 0.4373         # Allocation fraction to soma (-)
    
    # --- Maintenance and Growth ---
    pT::Float64 = 0.0             # {p_T}, surface-specific somatic maintenance (J/d/cm²)
    Eg::Float64 = 6880.33         # [E_G], specific cost for structure (J/cm³)
    Shape::Float64 = 0.066193     # Shape coefficient (-)
    w::Float64 = 27.24            # Contribution of reserve to body weight (g/cm³)
    
    # --- Maturity ---
    kJ::Float64 = 0.002793        # Maturity maintenance rate coefficient (1/d)
    alpha_Ehb::Float64 = 0.8      # Ratio of Ehb to E_coc ([0,1))
    beta_Ehp::Float64 = 10.0      # Ratio of Ehp to Ehb (>1)
    
    # --- Reproduction ---
    L_coc::Float64 = 0.23         # Structural length of a cocoon (cm)
    E_coc::Float64 = 467.0        # Energy in a cocoon (J)
    kapR::Float64 = 0.475         # Reproduction efficiency (-)
    
    # --- Temperature Response (Arrhenius) ---
    TA::Float64 = 7976.0          # Arrhenius temperature (K)
    TAH::Float64 = 28750.0        # Arrhenius temperature for upper boundary (K)
    TH::Float64 = 293.2           # Upper boundary temperature (K)
    Tref::Float64 = 293.15        # Reference temperature (K)
    
    # --- Experimental Conditions ---
    Texp::Float64 = 20.0          # Experimental temperature (°C)
    Dens::Float64 = 1.0           # Number of earthworms in cosm (#)
end

# === Derived Parameters ===
struct DEBDerivedParams
    pM::Float64       # [p_M], volume-specific somatic maintenance (J/d/cm³)
    Ehb::Float64      # Maturity at birth (J)
    Ehp::Float64      # Maturity at puberty (J)
    Em::Float64       # Maximum reserve density (J/cm³)
    Km::Float64       # Somatic maintenance rate coefficient (1/d)
end

function derive_params(p::DEBEarthwormParams)
    pM = p.pAm * p.r_pAm_pM
    Ehb = p.alpha_Ehb * p.E_coc
    Ehp = p.beta_Ehp * Ehb
    Em = p.pAm / p.v
    Km = pM / p.Eg
    return DEBDerivedParams(pM, Ehb, Ehp, Em, Km)
end

# === Temperature Correction ===
function temperature_correction(Texp_C, p::DEBEarthwormParams)
    T = Texp_C + 273.15  # Convert to Kelvin
    
    # Standard Arrhenius
    sA = exp(p.TA / p.Tref - p.TA / T)
    
    # Upper boundary correction
    srH = (1 + exp(p.TAH / p.TH - p.TAH / p.Tref)) / 
          (1 + exp(p.TAH / p.TH - p.TAH / T))
    
    # Apply upper correction only if T >= Tref
    Tc = sA * (T >= p.Tref ? srH : 1.0)
    
    return Tc
end

# === Functional Response ===
function functional_response(Weight, OM_soil, OM_horse, L, pAm_t, p::DEBEarthwormParams, d::DEBDerivedParams)
    # Maximum energy that can be ingested (J/d)
    Qfmax = pAm_t * L^2 / p.kapX
    
    # Energy available from soil and horse dung (J/d)
    Qf_soil = p.Fm * Weight * (p.muOM * p.rOM_ClxHorse * OM_soil / p.Dens)
    Qf_horse = p.Fm * Weight * (p.muOM * OM_horse / p.Dens)
    Qf = Qf_soil + Qf_horse
    
    # Type II functional response
    X = Qf / (0.5 * Qfmax)
    freal = X / (1 + X)
    
    return freal, Qf_soil, Qf_horse
end

# === DEB ODE System ===
function deb_earthworm!(du, u, params, t)
    # Unpack state variables
    E, L, Eh, R, OM_soil, OM_horse = u
    
    # Unpack parameters
    p, d = params
    
    # Temperature correction
    Tc = temperature_correction(p.Texp, p)
    
    # Temperature-corrected rates
    pAm_t = p.pAm * Tc
    v_t = p.v * Tc
    kJ_t = p.kJ * Tc
    Km_t = d.Km * Tc
    
    # Allocation fraction: switches at puberty, and depends on density
    # In the original: after puberty with Dens > 1, allocation flips to (1-kap)
    kapreal = Eh < d.Ehp ? p.kap : (p.Dens > 1 ? (1 - p.kap) : p.kap)
    
    # Compound parameters (depend on kapreal)
    g = p.Eg / (kapreal * d.Em)
    Lm = p.v / (d.Km * g)
    
    # Current weight (g)
    Weight = L^3 * (1 + E / d.Em * p.w)
    
    # === Functional Response & Resource Dynamics ===
    freal, Qf_soil, Qf_horse = functional_response(Weight, OM_soil, OM_horse, L, pAm_t, p, d)
    
    # Organic matter consumption rates (g/d for entire cosm)
    OM_soil_out = (Qf_soil / (p.muOM * p.rOM_ClxHorse)) * p.Dens
    OM_horse_out = (Qf_horse / p.muOM) * p.Dens
    
    # === State Equations ===
    
    # Reserve dynamics (scaled by structural length)
    # dE/dt in terms of reserve density E (J/cm³)
    dE = (pAm_t / L) * (freal - E / d.Em)
    dE = E < 1e-12 ? -E : dE  # Prevent negative reserve
    
    # Structural length dynamics
    dL = (v_t / (3 * ((E / d.Em) + g))) * ((E / d.Em) - (L / Lm))
    
    # Mobilization flux
    pC = (g * E / (g + E / d.Em)) * (v_t * L^2 + Km_t * L^3)
    
    # Maturity dynamics (only increases until puberty)
    if Eh < d.Ehp
        dEh = (1 - kapreal) * pC - kJ_t * Eh
    else
        dEh = 0.0
    end
    
    # Reproduction (only after puberty)
    if Eh >= d.Ehp
        dR = p.kapR * ((1 - kapreal) * pC - kJ_t * d.Ehp) / p.E_coc
    else
        dR = 0.0
    end
    
    # Organic matter depletion (prevent going negative)
    dOM_horse = (OM_horse - OM_horse_out) < 1e-12 ? -OM_horse : -OM_horse_out
    dOM_soil = (OM_soil - OM_soil_out) < 1e-12 ? -OM_soil : -OM_soil_out
    
    # Pack derivatives
    du[1] = dE
    du[2] = dL
    du[3] = dEh
    du[4] = dR
    du[5] = dOM_soil
    du[6] = dOM_horse
end

# === Initialization Function ===
function initialize_earthworm(Winit, init_status, p::DEBEarthwormParams, d::DEBDerivedParams;
                               OM_soil_init=100.0, OM_horse_init=50.0)
    # Initial reserve density at maximum
    E0 = d.Em
    
    # Initial structural length from weight (assuming E/Em = 1)
    L0 = (Winit / (1 + 1 * p.w))^(1/3)
    
    # Initial maturity depends on life stage
    # init_status: 3 = adult (at puberty), otherwise juvenile (at birth)
    Eh0 = init_status == 3 ? d.Ehp : d.Ehb
    
    # Initial reproduction buffer
    R0 = 1e-6
    
    return [E0, L0, Eh0, R0, OM_soil_init, OM_horse_init]
end

# === Output Calculations ===
function calc_outputs(sol, p::DEBEarthwormParams, d::DEBDerivedParams)
    E = sol[1, :]
    L = sol[2, :]
    Eh = sol[3, :]
    R = sol[4, :]
    OM_soil = sol[5, :]
    OM_horse = sol[6, :]
    
    # Physical length (cm)
    Size = L ./ p.Shape
    
    # Weight (g)
    Weight = L.^3 .* (1 .+ E ./ d.Em .* p.w)
    
    # Total organic matter
    OM_total = OM_soil .+ OM_horse
    
    return (
        t = sol.t,
        Energy = E,
        Length_struct = L,
        Size = Size,
        Maturity = max.(Eh, 1e-9),
        Reproduction = max.(R, 1e-9),
        Weight = Weight,
        OM_soil = OM_soil,
        OM_horse = OM_horse,
        OM_total = OM_total
    )
end

# ============================================================================
# Example Simulation
# ============================================================================

# Create parameters
params = DEBEarthwormParams(
    Texp = 20.0,    # Temperature (°C)
    Dens = 1.0      # Single earthworm
)
derived = derive_params(params)

# Initialize: adult earthworm, 0.5g initial weight
Winit = 0.5  # grams
u0 = initialize_earthworm(Winit, 3, params, derived; 
                          OM_soil_init = 100.0,   # g soil OM
                          OM_horse_init = 50.0)   # g horse dung OM

# Simulation timespan (200 days)
tspan = (0.0, 200.0)

# Solve
prob = DE.ODEProblem(deb_earthworm!, u0, tspan, (params, derived))
sol = DE.solve(prob, DE.Tsit5(), saveat=1.0)

# Calculate outputs
outputs = calc_outputs(sol, params, derived)

# === Plotting ===
p1 = plot(outputs.t, outputs.Weight, label="Weight (g)", 
          ylabel="Weight (g)", lw=2, color=:brown)

p2 = plot(outputs.t, outputs.Size, label="Physical length", 
          ylabel="Length (cm)", lw=2, color=:green)

p3 = plot(outputs.t, outputs.Reproduction, label="Cocoons produced", 
          ylabel="# Cocoons", lw=2, color=:purple)

p4 = plot(outputs.t, outputs.OM_soil, label="Soil OM", lw=2)
plot!(p4, outputs.t, outputs.OM_horse, label="Horse dung OM", lw=2)
plot!(p4, outputs.t, outputs.OM_total, label="Total OM", lw=2, ls=:dash)
ylabel!(p4, "OM (g)")
xlabel!(p4, "Time (days)")

plot(p1, p2, p3, p4, layout=(4,1), size=(800, 700), 
     plot_title="DEB Model: Apporectodea caliginosa")