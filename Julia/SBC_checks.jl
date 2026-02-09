# ============================================================================
# DIAGNOSTIC SCRIPT - Run this to identify the failure point
# ============================================================================

include("DEB_Acali_wfood.jl")

using Turing
using Distributions
using Random

# ============================================================================
# TEST 1: Can the simplified ODE solve at all?
# ============================================================================

println("=" ^ 70)
println("TEST 1: Basic ODE solution")
println("=" ^ 70)

# Simplified ODE (copy from SBC file)
function deb_simple!(du, u, params, t)
    E, L, Eh, R = u
    p, d = params
    
    Tc = temperature_correction(p.Texp, p)
    pAm_t = p.pAm * Tc
    v_t = p.v * Tc
    kJ_t = p.kJ * Tc
    Km_t = d.Km * Tc
    
    kapreal = p.kap
    g = p.Eg / (kapreal * d.Em)
    Lm = p.v / (d.Km * g)
    freal = 1.0
    
    tmp_E = (pAm_t / L) * (freal - E / d.Em)
    if E < 1e-12
        dE = -E
    else
        dE = tmp_E
    end
    
    dL = (v_t / (3 * ((E / d.Em) + g))) * ((E / d.Em) - (L / Lm))
    pC = (g * E / (g + E / d.Em)) * (v_t * L^2 + Km_t * L^3)
    
    tmp_Eh = (1 - kapreal) * pC - kJ_t * Eh
    if Eh < d.Ehp
        dEh = tmp_Eh
    else
        dEh = 0.0
    end
    
    tmp_R = p.kapR * ((1 - kapreal) * pC - kJ_t * d.Ehp)
    if Eh >= d.Ehp
        dR = tmp_R / p.E_coc
    else
        dR = 0.0
    end
    
    du[1] = dE
    du[2] = dL
    du[3] = dEh
    du[4] = dR
end

function initialize_simple(Winit, init_status, p, d)
    E0 = d.Em
    L0 = (Winit / (1 + p.w))^(1/3)
    Eh0 = init_status == 3 ? d.Ehp : d.Ehb
    R0 = 1e-6
    return [E0, L0, Eh0, R0]
end

# Test with default parameters
base_params = DEBEarthwormParams(Texp = 15.0, Dens = 1.0)
derived = derive_params(base_params)

Winit = 0.015
init_status = 1
tspan = (0.0, 180.0)
obs_times = collect(0.0:14.0:180.0)

u0 = initialize_simple(Winit, init_status, base_params, derived)
println("Initial conditions: $u0")

prob = DE.ODEProblem(deb_simple!, u0, tspan, (base_params, derived))
sol = DE.solve(prob, DE.Tsit5(), saveat=obs_times)

println("Solution status: $(sol.retcode)")
if sol.retcode == :Success
    println("✓ ODE solves with default parameters")
    println("Final state: $(sol.u[end])")
    
    # Calculate weight
    E = sol[1, :]
    L = sol[2, :]
    Weight = L.^3 .* (1 .+ E ./ derived.Em .* base_params.w)
    println("Weight trajectory: $(round.(Weight, digits=4))")
else
    println("✗ ODE FAILED with default parameters!")
end

# ============================================================================
# TEST 2: Test with parameters sampled from prior
# ============================================================================

println("\n" * "=" ^ 70)
println("TEST 2: ODE with sampled prior parameters")
println("=" ^ 70)

prior_pAm = truncated(Normal(1172.0, 117.2), 800.0, 2000.0)
prior_kap = Uniform(0.05, 0.25)

Random.seed!(42)
n_test = 20
n_success = 0

for i in 1:n_test
    test_pAm = rand(prior_pAm)
    test_kap = rand(prior_kap)
    
    test_params = DEBEarthwormParams(
        pAm = test_pAm,
        kap = test_kap,
        Texp = base_params.Texp,
        Dens = base_params.Dens,
        # Keep other parameters at defaults
        muOM = base_params.muOM,
        rOM_ClxHorse = base_params.rOM_ClxHorse,
        Fm = base_params.Fm,
        kapX = base_params.kapX,
        r_pAm_pM = base_params.r_pAm_pM,
        v = base_params.v,
        pT = base_params.pT,
        Eg = base_params.Eg,
        Shape = base_params.Shape,
        w = base_params.w,
        kJ = base_params.kJ,
        alpha_Ehb = base_params.alpha_Ehb,
        beta_Ehp = base_params.beta_Ehp,
        L_coc = base_params.L_coc,
        E_coc = base_params.E_coc,
        kapR = base_params.kapR,
        TA = base_params.TA,
        TAH = base_params.TAH,
        TH = base_params.TH,
        Tref = base_params.Tref
    )
    
    test_derived = derive_params(test_params)
    u0_test = initialize_simple(Winit, init_status, test_params, test_derived)
    
    prob_test = DE.ODEProblem(deb_simple!, u0_test, tspan, (test_params, test_derived))
    
    try
        sol_test = DE.solve(prob_test, DE.Tsit5(), saveat=obs_times)
        if sol_test.retcode == :Success
            n_success += 1
            print("✓")
        else
            print("✗($(sol_test.retcode))")
        end
    catch e
        print("✗(error)")
        if i == 1
            println("\n  Error details: $e")
        end
    end
end

println("\nPrior parameter ODE success rate: $n_success / $n_test")

# ============================================================================
# TEST 3: Test data generation
# ============================================================================

println("\n" * "=" ^ 70)
println("TEST 3: Synthetic data generation")
println("=" ^ 70)

function generate_test_data(pAm, kap, sigma_w, obs_times, Winit, init_status, tspan, base_params)
    params = DEBEarthwormParams(
        pAm = pAm,
        kap = kap,
        Texp = base_params.Texp,
        Dens = base_params.Dens,
        muOM = base_params.muOM,
        rOM_ClxHorse = base_params.rOM_ClxHorse,
        Fm = base_params.Fm,
        kapX = base_params.kapX,
        r_pAm_pM = base_params.r_pAm_pM,
        v = base_params.v,
        pT = base_params.pT,
        Eg = base_params.Eg,
        Shape = base_params.Shape,
        w = base_params.w,
        kJ = base_params.kJ,
        alpha_Ehb = base_params.alpha_Ehb,
        beta_Ehp = base_params.beta_Ehp,
        L_coc = base_params.L_coc,
        E_coc = base_params.E_coc,
        kapR = base_params.kapR,
        TA = base_params.TA,
        TAH = base_params.TAH,
        TH = base_params.TH,
        Tref = base_params.Tref
    )
    
    derived = derive_params(params)
    u0 = initialize_simple(Winit, init_status, params, derived)
    
    prob = DE.ODEProblem(deb_simple!, u0, tspan, (params, derived))
    sol = DE.solve(prob, DE.Tsit5(), saveat=obs_times)
    
    if sol.retcode != :Success
        println("  ODE failed: $(sol.retcode)")
        return nothing
    end
    
    E = sol[1, :]
    L = sol[2, :]
    Weight = L.^3 .* (1 .+ E ./ derived.Em .* params.w)
    
    # Check for invalid weights
    if any(isnan.(Weight)) || any(isinf.(Weight)) || any(Weight .<= 0)
        println("  Invalid weights: NaN=$(any(isnan.(Weight))), Inf=$(any(isinf.(Weight))), <=0=$(any(Weight .<= 0))")
        return nothing
    end
    
    obs_weight = Weight .+ rand(Normal(0, sigma_w), length(Weight))
    obs_weight = max.(obs_weight, 0.001)
    
    return obs_weight
end

# Test with known good parameters
test_pAm = 1172.0
test_kap = 0.15
test_sigma = 0.02

obs_data = generate_test_data(test_pAm, test_kap, test_sigma, 
                               obs_times, Winit, init_status, tspan, base_params)

if !isnothing(obs_data)
    println("✓ Data generation works")
    println("  Observations: $(round.(obs_data, digits=4))")
else
    println("✗ Data generation FAILED")
end

# ============================================================================
# TEST 4: Test Turing model (single sample)
# ============================================================================

println("\n" * "=" ^ 70)
println("TEST 4: Turing model sampling")
println("=" ^ 70)

# Define a simpler Turing model for testing
@model function deb_test_model(obs_weight, obs_times, Winit, init_status, tspan, base_params)
    # Priors
    pAm ~ truncated(Normal(1172.0, 117.2), 800.0, 2000.0)
    kap ~ Uniform(0.05, 0.25)
    Sigma_W ~ truncated(Normal(0.05, 0.02), 0.001, 0.5)
    
    # Build parameters
    params = DEBEarthwormParams(
        pAm = pAm,
        kap = kap,
        Texp = base_params.Texp,
        Dens = base_params.Dens,
        muOM = base_params.muOM,
        rOM_ClxHorse = base_params.rOM_ClxHorse,
        Fm = base_params.Fm,
        kapX = base_params.kapX,
        r_pAm_pM = base_params.r_pAm_pM,
        v = base_params.v,
        pT = base_params.pT,
        Eg = base_params.Eg,
        Shape = base_params.Shape,
        w = base_params.w,
        kJ = base_params.kJ,
        alpha_Ehb = base_params.alpha_Ehb,
        beta_Ehp = base_params.beta_Ehp,
        L_coc = base_params.L_coc,
        E_coc = base_params.E_coc,
        kapR = base_params.kapR,
        TA = base_params.TA,
        TAH = base_params.TAH,
        TH = base_params.TH,
        Tref = base_params.Tref
    )
    
    derived = derive_params(params)
    u0 = initialize_simple(Winit, init_status, params, derived)
    
    prob = DE.ODEProblem(deb_simple!, u0, tspan, (params, derived))
    sol = DE.solve(prob, DE.Tsit5(), saveat=obs_times, 
                   abstol=1e-6, reltol=1e-3)  # Relaxed tolerances
    
    if sol.retcode != :Success
        Turing.@addlogprob! -Inf
        return
    end
    
    E = sol[1, :]
    L = sol[2, :]
    predicted_weight = L.^3 .* (1 .+ E ./ derived.Em .* params.w)
    
    # Check for valid predictions
    if any(isnan.(predicted_weight)) || any(predicted_weight .<= 0)
        Turing.@addlogprob! -Inf
        return
    end
    
    # Likelihood
    for i in eachindex(obs_weight)
        obs_weight[i] ~ Normal(predicted_weight[i], Sigma_W)
    end
end

# Generate test data with known parameters
if !isnothing(obs_data)
    println("Testing Turing model with generated data...")
    
    model = deb_test_model(obs_data, obs_times, Winit, init_status, tspan, base_params)
    
    try
        # Try just a few samples first
        println("  Attempting 10 NUTS samples...")
        chain = sample(model, NUTS(0.65), 10, progress=true)
        println("✓ Turing sampling works!")
        println("  Chain summary:")
        display(chain)
    catch e
        println("✗ Turing sampling FAILED")
        println("  Error: $e")
        println("\n  Trying with MH sampler instead...")
        
        try
            chain = sample(model, MH(), 100, progress=true)
            println("✓ MH sampling works (NUTS might need tuning)")
        catch e2
            println("✗ MH also failed: $e2")
        end
    end
else
    println("Skipping Turing test - data generation failed")
end

# ============================================================================
# TEST 5: Check parameter space boundaries
# ============================================================================

println("\n" * "=" ^ 70)
println("TEST 5: Parameter space exploration")
println("=" ^ 70)

# Test extreme corners of parameter space
test_cases = [
    (pAm=800.0, kap=0.05, name="low pAm, low kap"),
    (pAm=800.0, kap=0.25, name="low pAm, high kap"),
    (pAm=2000.0, kap=0.05, name="high pAm, low kap"),
    (pAm=2000.0, kap=0.25, name="high pAm, high kap"),
    (pAm=1172.0, kap=0.15, name="center"),
]

for tc in test_cases
    params = DEBEarthwormParams(
        pAm = tc.pAm,
        kap = tc.kap,
        Texp = base_params.Texp,
        Dens = base_params.Dens,
        muOM = base_params.muOM,
        rOM_ClxHorse = base_params.rOM_ClxHorse,
        Fm = base_params.Fm,
        kapX = base_params.kapX,
        r_pAm_pM = base_params.r_pAm_pM,
        v = base_params.v,
        pT = base_params.pT,
        Eg = base_params.Eg,
        Shape = base_params.Shape,
        w = base_params.w,
        kJ = base_params.kJ,
        alpha_Ehb = base_params.alpha_Ehb,
        beta_Ehp = base_params.beta_Ehp,
        L_coc = base_params.L_coc,
        E_coc = base_params.E_coc,
        kapR = base_params.kapR,
        TA = base_params.TA,
        TAH = base_params.TAH,
        TH = base_params.TH,
        Tref = base_params.Tref
    )
    
    derived = derive_params(params)
    
    # Check derived parameters
    println("\n$(tc.name):")
    println("  Em = $(round(derived.Em, digits=2)), Km = $(round(derived.Km, digits=6))")
    println("  Ehb = $(round(derived.Ehb, digits=2)), Ehp = $(round(derived.Ehp, digits=2))")
    
    u0 = initialize_simple(Winit, init_status, params, derived)
    println("  u0 = $(round.(u0, digits=4))")
    
    prob = DE.ODEProblem(deb_simple!, u0, tspan, (params, derived))
    
    try
        sol = DE.solve(prob, DE.Tsit5(), saveat=obs_times)
        if sol.retcode == :Success
            E = sol[1, :]
            L = sol[2, :]
            W = L.^3 .* (1 .+ E ./ derived.Em .* params.w)
            println("  ✓ Success: W_final = $(round(W[end], digits=4)) g")
        else
            println("  ✗ Failed: $(sol.retcode)")
        end
    catch e
        println("  ✗ Error: $e")
    end
end

println("\n" * "=" ^ 70)
println("DIAGNOSTIC COMPLETE")
println("=" ^ 70)