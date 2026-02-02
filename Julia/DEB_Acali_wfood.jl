import DifferentialEquations as DE
using Plots

# ============================================================================
# DEB Model for Apporectodea caliginosa (Earthworm)
# With Periodic Feeding Events
# ============================================================================

Base.@kwdef struct DEBEarthwormParams
    # --- Environment ---
    muOM::Float64 = 11700.0       # Energy in 1g OM from horse dung (J/g)
    rOM_ClxHorse::Float64 = 0.3   # Ratio of energy: soil OM vs horse dung OM
    
    # --- Energy Assimilation ---
    Fm::Float64 = 0.5             # Max specific searching rate (g/d/g)
    kapX::Float64 = 0.258         # Digestion efficiency (-)
    pAm::Float64 = 1712.41        # {p_Am}, spec assimilation flux (J/d/cm²)
    r_pAm_pM::Float64 = 1.5       # Ratio pAm/pM
    v::Float64 = 0.018505         # Energy conductance (cm/d)
    
    # --- Allocation ---
    kap::Float64 = 0.4373         # Allocation fraction to soma (-)
    
    # --- Maintenance and Growth ---
    pT::Float64 = 0.0             # {p_T}, surface-spec somatic maint (J/d/cm²)
    Eg::Float64 = 6880.33         # [E_G], spec cost for structure (J/cm³)
    Shape::Float64 = 0.066193     # Shape coefficient (-)
    w::Float64 = 27.24            # Contribution of reserve to weight (g/cm³)
    
    # --- Maturity ---
    kJ::Float64 = 0.002793        # Maturity maint rate coefficient (1/d)
    alpha_Ehb::Float64 = 0.8      # Ratio Ehb/E_coc
    beta_Ehp::Float64 = 10.0      # Ratio Ehp/Ehb
    
    # --- Reproduction ---
    L_coc::Float64 = 0.23         # Structural length of cocoon (cm)
    E_coc::Float64 = 467.0        # Energy in a cocoon (J)
    kapR::Float64 = 0.475         # Reproduction efficiency (-)
    
    # --- Temperature Response ---
    TA::Float64 = 7976.0          # Arrhenius temperature (K)
    TAH::Float64 = 28750.0        # Arrhenius temp upper boundary (K)
    TH::Float64 = 293.2           # Upper boundary temp (K)
    Tref::Float64 = 293.15        # Reference temperature (K)
    
    # --- Experimental Conditions ---
    Texp::Float64 = 20.0          # Temperature (°C)
    Dens::Float64 = 1.0           # Number of earthworms in cosm
end

struct DEBDerivedParams
    pM::Float64
    Ehb::Float64
    Ehp::Float64
    Em::Float64
    Km::Float64
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
    T = Texp_C + 273.15
    sA = exp(p.TA / p.Tref - p.TA / T)
    srH = (1 + exp(p.TAH / p.TH - p.TAH / p.Tref)) / 
          (1 + exp(p.TAH / p.TH - p.TAH / T))
    
    if T >= p.Tref
        Tc = sA * srH
    else
        Tc = sA
    end
    
    return Tc
end

# === Functional Response ===
function functional_response(Weight, OM_soil, OM_horse, L, pAm_t, 
                              p::DEBEarthwormParams, d::DEBDerivedParams)
    Qfmax = pAm_t * L^2 / p.kapX
    Qf_soil = p.Fm * Weight * (p.muOM * p.rOM_ClxHorse * OM_soil / p.Dens)
    Qf_horse = p.Fm * Weight * (p.muOM * OM_horse / p.Dens)
    Qf = Qf_soil + Qf_horse
    X = Qf / (0.5 * Qfmax)
    freal = X / (1 + X)
    return freal, Qf_soil, Qf_horse
end

# === DEB ODE System ===
function deb_earthworm!(du, u, params, t)
    E, L, Eh, R, OM_soil, OM_horse = u
    p, d = params
    
    # Temperature correction
    Tc = temperature_correction(p.Texp, p)
    pAm_t = p.pAm * Tc
    v_t = p.v * Tc
    kJ_t = p.kJ * Tc
    Km_t = d.Km * Tc
    
    # Allocation fraction
    if Eh < d.Ehp
        kapreal = p.kap
    else
        if p.Dens > 1
            kapreal = 1 - p.kap
        else
            kapreal = p.kap
        end
    end
    
    # Compound parameters
    g = p.Eg / (kapreal * d.Em)
    Lm = p.v / (d.Km * g)
    
    # Current weight (g)
    Weight = L^3 * (1 + E / d.Em * p.w)
    
    # Functional response
    freal, Qf_soil, Qf_horse = functional_response(Weight, OM_soil, OM_horse, 
                                                    L, pAm_t, p, d)
    
    # OM consumption rates
    OM_soil_out = (Qf_soil / (p.muOM * p.rOM_ClxHorse)) * p.Dens
    OM_horse_out = (Qf_horse / p.muOM) * p.Dens
    
    # === State Equations ===
    
    # Reserve density dynamics
    tmp_E = (pAm_t / L) * (freal - E / d.Em)
    if E < 1e-12
        dE = -E
    else
        dE = tmp_E
    end
    
    # Structural length dynamics
    dL = (v_t / (3 * ((E / d.Em) + g))) * ((E / d.Em) - (L / Lm))
    
    # Mobilization flux
    pC = (g * E / (g + E / d.Em)) * (v_t * L^2 + Km_t * L^3)
    
    # Maturity dynamics
    tmp_Eh = (1 - kapreal) * pC - kJ_t * Eh
    if Eh < d.Ehp
        dEh = tmp_Eh
    else
        dEh = 0.0
    end
    
    # Reproduction
    tmp_R = p.kapR * ((1 - kapreal) * pC - kJ_t * d.Ehp)
    if Eh >= d.Ehp
        dR = tmp_R / p.E_coc
    else
        dR = 0.0
    end
    
    # OM depletion
    if (OM_horse - OM_horse_out) < 1e-12
        dOM_horse = -OM_horse
    else
        dOM_horse = -OM_horse_out
    end
    
    if (OM_soil - OM_soil_out) < 1e-12
        dOM_soil = -OM_soil
    else
        dOM_soil = -OM_soil_out
    end
    
    du[1] = dE
    du[2] = dL
    du[3] = dEh
    du[4] = dR
    du[5] = dOM_soil
    du[6] = dOM_horse
end

# === Initialization ===
function initialize_earthworm(Winit, init_status, p::DEBEarthwormParams, d::DEBDerivedParams;
                               OM_soil_init=10.0, OM_horse_init=5.0)
    E0 = d.Em
    L0 = (Winit / (1 + p.w))^(1/3)
    
    if init_status == 3
        Eh0 = d.Ehp
    else
        Eh0 = d.Ehb
    end
    
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
    
    Size = L ./ p.Shape
    Weight = L.^3 .* (1 .+ E ./ d.Em .* p.w)
    
    Tc = temperature_correction(p.Texp, p)
    pAm_t = p.pAm * Tc
    freal = similar(L)
    for i in eachindex(L)
        freal[i], _, _ = functional_response(Weight[i], OM_soil[i], OM_horse[i], 
                                              L[i], pAm_t, p, d)
    end
    
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
        OM_total = OM_soil .+ OM_horse,
        freal = freal
    )
end

# ============================================================================
# FEEDING EVENT SYSTEM
# ============================================================================

"""
    FeedingSchedule

Defines the periodic feeding regime for the earthworm experiment.

Fields:
- horse_interval: Days between horse manure additions (default: 14)
- horse_amount: Amount of horse manure added each time in grams (default: 20)
- horse_start: Day of first horse manure addition (default: 14)
- soil_interval: Days between soil changes (default: 28)
- soil_amount: Amount of fresh soil at each change in grams (default: 100)
- soil_start: Day of first soil change (default: 28)
- verbose: Print event messages (default: true)
"""
Base.@kwdef struct FeedingSchedule
    horse_interval::Float64 = 14.0
    horse_amount::Float64 = 20.0
    horse_start::Float64 = 14.0
    soil_interval::Float64 = 28.0
    soil_amount::Float64 = 100.0
    soil_start::Float64 = 28.0
    verbose::Bool = true
end

"""
    create_feeding_callbacks(tspan, schedule)

Create DifferentialEquations.jl callbacks for periodic feeding events.

Arguments:
- tspan: Simulation time span (t_start, t_end)
- schedule: FeedingSchedule struct defining the feeding regime

Returns:
- CallbackSet containing horse manure and soil change callbacks

State variable indices:
- u[5] = OM_soil
- u[6] = OM_horse
"""
function create_feeding_callbacks(tspan, schedule::FeedingSchedule)
    t_start, t_end = tspan
    
    # Generate event times
    horse_times = collect(schedule.horse_start : schedule.horse_interval : t_end)
    soil_times = collect(schedule.soil_start : schedule.soil_interval : t_end)
    
    # Horse manure addition callback (additive)
    function add_horse_manure!(integrator)
        integrator.u[6] += schedule.horse_amount
        if schedule.verbose
            @info "Day $(round(integrator.t, digits=1)): Added $(schedule.horse_amount)g horse manure → Total: $(round(integrator.u[6], digits=1))g"
        end
    end
    horse_callback = DE.PresetTimeCallback(horse_times, add_horse_manure!)
    
    # Soil change callback (replacement)
    function change_soil!(integrator)
        old_soil = integrator.u[5]
        integrator.u[5] = schedule.soil_amount
        if schedule.verbose
            @info "Day $(round(integrator.t, digits=1)): Soil changed $(round(old_soil, digits=1))g → $(schedule.soil_amount)g"
        end
    end
    soil_callback = DE.PresetTimeCallback(soil_times, change_soil!)
    
    return DE.CallbackSet(horse_callback, soil_callback)
end

"""
    create_custom_feeding_callbacks(horse_events, soil_events; verbose=true)

Create callbacks for irregular/custom feeding schedules.

Arguments:
- horse_events: Vector of (time, amount) tuples for horse manure additions
- soil_events: Vector of (time, amount) tuples for soil replacements
- verbose: Print event messages

Example:
    horse = [(7.0, 15.0), (21.0, 25.0), (50.0, 30.0)]
    soil = [(14.0, 80.0), (42.0, 120.0)]
    cb = create_custom_feeding_callbacks(horse, soil)
"""
function create_custom_feeding_callbacks(horse_events::Vector{Tuple{Float64, Float64}},
                                          soil_events::Vector{Tuple{Float64, Float64}};
                                          verbose::Bool = true)
    callbacks = DiscreteCallback[]
    
    # Horse manure events
    for (event_time, amount) in horse_events
        condition(u, t, integrator) = t == event_time
        function affect!(integrator)
            integrator.u[6] += amount
            if verbose
                @info "Day $(round(integrator.t, digits=1)): Added $(amount)g horse manure"
            end
        end
        cb = DE.PresetTimeCallback([event_time], affect!)
        push!(callbacks, cb)
    end
    
    # Soil replacement events
    for (event_time, amount) in soil_events
        function affect_soil!(integrator)
            integrator.u[5] = amount
            if verbose
                @info "Day $(round(integrator.t, digits=1)): Soil replaced with $(amount)g"
            end
        end
        cb = DE.PresetTimeCallback([event_time], affect_soil!)
        push!(callbacks, cb)
    end
    
    return DE.CallbackSet(callbacks...)
end

"""
    get_event_times(tspan, schedule)

Return vectors of event times for plotting/reference.
"""
function get_event_times(tspan, schedule::FeedingSchedule)
    t_end = tspan[2]
    horse_times = collect(schedule.horse_start : schedule.horse_interval : t_end)
    soil_times = collect(schedule.soil_start : schedule.soil_interval : t_end)
    return horse_times, soil_times
end

# ============================================================================
# VISUALIZATION
# ============================================================================

"""
    plot_deb_results(outputs, feeding, tspan, derived)

Create a comprehensive visualization of DEB simulation results with event markers.
"""
function plot_deb_results(outputs, feeding::FeedingSchedule, tspan, derived::DEBDerivedParams)
    horse_times, soil_times = get_event_times(tspan, feeding)
    
    # Weight
    p1 = plot(outputs.t, outputs.Weight .* 1000, 
              label="Weight", ylabel="Weight (mg)", 
              lw=2, color=:brown, legend=:topright)
    
    # Physical length
    p2 = plot(outputs.t, outputs.Size, 
              label="Physical length", ylabel="Length (cm)", 
              lw=2, color=:green)
    
    # Maturity with puberty threshold
    p3 = plot(outputs.t, outputs.Maturity,
              label="Maturity (Eh)", ylabel="Maturity (J)", 
              lw=2, color=:blue)
    hline!(p3, [derived.Ehp], ls=:dash, color=:red, label="Puberty", lw=1)
    
    # Reproduction
    p4 = plot(outputs.t, outputs.Reproduction, 
              label="Cocoons", ylabel="# Cocoons", 
              lw=2, color=:purple)
    
    # Organic matter with event markers
    p5 = plot(outputs.t, outputs.OM_soil, 
              label="Soil OM", lw=2, color=:sienna)
    plot!(p5, outputs.t, outputs.OM_horse, 
          label="Horse OM", lw=2, color=:darkgreen)
    plot!(p5, outputs.t, outputs.OM_total, 
          label="Total OM", lw=2, ls=:dash, color=:black)
    
    # Add event markers
    for t in horse_times
        vline!(p5, [t], color=:darkgreen, alpha=0.4, lw=1, label=false)
    end
    for t in soil_times
        vline!(p5, [t], color=:sienna, alpha=0.6, lw=2, ls=:dot, label=false)
    end
    ylabel!(p5, "OM (g)")
    
    # Functional response
    p6 = plot(outputs.t, outputs.freal, 
              label="f (func. response)", ylabel="f (-)", 
              lw=2, color=:orange, ylims=(0, 1.1))
    xlabel!(p6, "Time (days)")
    
    plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(900, 750),
         plot_title="DEB A. caliginosa with Feeding Events")
end

# ============================================================================
# RUN SIMULATION WITH EVENTS
# ============================================================================
# Simulate Bart2019_XP1_1
# Init= 1;
# WeightSoilCosm = 400;
# Winit=0.01;
# Soil=1;
# Texp=15;

# Model parameters
params = DEBEarthwormParams(Texp = 15.0, Dens = 1.0)
derived = derive_params(params)

println("=== Derived Parameters ===")
println("Ehb (maturity at birth): $(derived.Ehb) J")
println("Ehp (maturity at puberty): $(derived.Ehp) J")
println("Em (max reserve density): $(round(derived.Em, digits=1)) J/cm³")

# Initial conditions: juvenile earthworm
Winit = 0.015  # 15 mg hatchling
init_status = 1  # Juvenile

L_init = (Winit / (1 + params.w))^(1/3)
Size_init = L_init / params.Shape
println("\n=== Initial Conditions ===")
println("Initial weight: $(Winit * 1000) mg")
println("Initial physical length: $(round(Size_init, digits=2)) cm")

u0 = initialize_earthworm(Winit, init_status, params, derived;
                          OM_soil_init = 10.0,
                          OM_horse_init = 3.0)

# Simulation time
tspan = (0.0, 365.0)

# Define feeding schedule
feeding = FeedingSchedule(
    horse_interval = 14.0,    # Every 2 weeks
    horse_amount = 2.7,      # 2.7g horse manure (additive)
    horse_start = 14.0,       # First addition at day 14
    soil_interval = 28.0,     # Every 4 weeks
    soil_amount = 13.0,      # Replace with 13g of soil OM matter
    soil_start = 28.0,        # First change at day 28
    verbose = true            # Print event messages
)

println("\n=== Feeding Schedule ===")
println("Horse manure: $(feeding.horse_amount)g every $(Int(feeding.horse_interval)) days")
println("Soil change: $(feeding.soil_amount)g every $(Int(feeding.soil_interval)) days")

# Create callbacks
callbacks = create_feeding_callbacks(tspan, feeding)

# Solve ODE with callbacks
prob = DE.ODEProblem(deb_earthworm!, u0, tspan, (params, derived))
sol = DE.solve(prob, DE.Tsit5(), callback=callbacks, saveat=1.0)

# Calculate outputs
outputs = calc_outputs(sol, params, derived)

# Print final state
println("\n=== Final State (Day $(Int(tspan[2]))) ===")
println("Weight: $(round(outputs.Weight[end] * 1000, digits=1)) mg")
println("Physical length: $(round(outputs.Size[end], digits=2)) cm")
println("Cocoons produced: $(round(outputs.Reproduction[end], digits=2))")
println("Final maturity: $(round(outputs.Maturity[end], digits=1)) J")

# Plot results
plot_deb_results(outputs, feeding, tspan, derived)
