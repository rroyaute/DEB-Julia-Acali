# ============================================================================
# Simulation Based Calibration (SBC) for DEB Apporectodea caliginosa
# Using Turing.jl - FIXED VERSION
# ============================================================================

include("DEB_Acali_wfood.jl")

using Turing
using Distributions
using Random
using DataFrames
using Statistics
using LinearAlgebra
using StatsPlots
using ProgressMeter
using SciMLBase  # For successful_retcode

# ============================================================================
# AD-COMPATIBLE DEB MODEL
# ============================================================================
# Key changes:
# 1. No Float64 type annotations (allow Dual numbers)
# 2. Parameters passed as a simple vector, not a typed struct
# 3. Use SciMLBase.successful_retcode() for checking solution status

"""
Simplified DEB ODE compatible with automatic differentiation.
Parameters are passed as a vector to avoid type restrictions.

param_vec = [pAm, kap, v, Eg, kJ, kapR, E_coc, w, Texp, TA, Tref, TAH, TH, 
             r_pAm_pM, alpha_Ehb, beta_Ehp]
"""
function deb_simple_ad!(du, u, param_vec, t)
    # Unpack state
    E, L, Eh, R = u
    
    # Unpack parameters (no type annotations!)
    pAm = param_vec[1]
    kap = param_vec[2]
    v = param_vec[3]
    Eg = param_vec[4]
    kJ = param_vec[5]
    kapR = param_vec[6]
    E_coc = param_vec[7]
    w = param_vec[8]
    Texp = param_vec[9]
    TA = param_vec[10]
    Tref = param_vec[11]
    TAH = param_vec[12]
    TH = param_vec[13]
    r_pAm_pM = param_vec[14]
    alpha_Ehb = param_vec[15]
    beta_Ehp = param_vec[16]
    
    # Derived parameters
    pM = pAm * r_pAm_pM
    Em = pAm / v
    Km = pM / Eg
    Ehb = alpha_Ehb * E_coc
    Ehp = beta_Ehp * Ehb
    
    # Temperature correction
    T = Texp + 273.15
    sA = exp(TA / Tref - TA / T)
    srH = (1 + exp(TAH / TH - TAH / Tref)) / (1 + exp(TAH / TH - TAH / T))
    Tc = T >= Tref ? sA * srH : sA
    
    # Temperature-corrected rates
    pAm_t = pAm * Tc
    v_t = v * Tc
    kJ_t = kJ * Tc
    Km_t = Km * Tc
    
    # Compound parameters (kap is now variable)
    g = Eg / (kap * Em)
    Lm = v / (Km * g)
    
    # Functional response (constant f = 1)
    freal = 1.0
    
    # State equations
    # Reserve density
    dE = (pAm_t / L) * (freal - E / Em)
    
    # Structural length
    dL = (v_t / (3 * ((E / Em) + g))) * ((E / Em) - (L / Lm))
    
    # Mobilization flux
    pC = (g * E / (g + (E / Em))) * (v_t * L^2 + Km_t * L^3)
    
    # Maturity (until puberty)
    if Eh < Ehp
        dEh = (1 - kap) * pC - kJ_t * Eh
    else
        dEh = zero(Eh)  # AD-compatible zero
    end
    
    # Reproduction (after puberty)
    if Eh >= Ehp
        dR = kapR * ((1 - kap) * pC - kJ_t * Ehp) / E_coc
    else
        dR = zero(R)  # AD-compatible zero
    end
    
    du[1] = dE
    du[2] = dL
    du[3] = dEh
    du[4] = dR
    
    return nothing
end

"""
Create parameter vector from individual values and base parameters.
"""
function make_param_vec(pAm, kap, base_params)
    return [
        pAm,                      # 1
        kap,                      # 2
        base_params.v,            # 3
        base_params.Eg,           # 4
        base_params.kJ,           # 5
        base_params.kapR,         # 6
        base_params.E_coc,        # 7
        base_params.w,            # 8
        base_params.Texp,         # 9
        base_params.TA,           # 10
        base_params.Tref,         # 11
        base_params.TAH,          # 12
        base_params.TH,           # 13
        base_params.r_pAm_pM,     # 14
        base_params.alpha_Ehb,    # 15
        base_params.beta_Ehp      # 16
    ]
end

"""
Initialize state vector (AD-compatible).
"""
function initialize_simple_ad(Winit, init_status, pAm, v, w, E_coc, alpha_Ehb, beta_Ehp)
    Em = pAm / v
    E0 = Em
    L0 = (Winit / (1 + w))^(1/3)
    
    Ehb = alpha_Ehb * E_coc
    Ehp = beta_Ehp * Ehb
    Eh0 = init_status == 3 ? Ehp : Ehb
    
    R0 = 1e-6
    return [E0, L0, Eh0, R0]
end

"""
Run DEB simulation with AD-compatible parameters.
Returns predicted weight at observation times.
"""
function run_deb_ad(pAm, kap, obs_times, Winit, init_status, tspan, base_params)
    # Create parameter vector
    param_vec = make_param_vec(pAm, kap, base_params)
    
    # Initial conditions (use base_params values for initialization)
    # Important: Don't use pAm here to avoid AD issues with initial conditions
    Em_base = base_params.pAm / base_params.v
    E0 = Em_base  # Fixed initial reserve density
    L0 = (Winit / (1 + base_params.w))^(1/3)
    Ehb = base_params.alpha_Ehb * base_params.E_coc
    Ehp = base_params.beta_Ehp * Ehb
    Eh0 = init_status == 3 ? Ehp : Ehb
    R0 = 1e-6
    
    u0 = [E0, L0, Eh0, R0]
    
    # Solve ODE
    prob = DE.ODEProblem(deb_simple_ad!, u0, tspan, param_vec)
    sol = DE.solve(prob, DE.Tsit5(), saveat=obs_times,
                   abstol=1e-6, reltol=1e-3, maxiters=1e5)
    
    return sol
end

"""
Calculate weight from ODE solution.
"""
function calc_weight_from_sol(sol, pAm, v, w)
    Em = pAm / v
    E = sol[1, :]
    L = sol[2, :]
    Weight = L.^3 .* (1 .+ E ./ Em .* w)
    return Weight
end

# ============================================================================
# TURING MODEL (AD-COMPATIBLE)
# ============================================================================

@model function deb_turing_model(obs_weight, obs_times, Winit, init_status, tspan, base_params)
    # Priors
    pAm ~ truncated(Normal(1172.0, 117.2), 800.0, 2000.0)
    kap ~ Uniform(0.05, 0.95)
    # Sigma_W ~ truncated(Normal(0.05, 0.02), 0.001, 0.5)
    Sigma_W ~ Exponential(1)
    
    # Run model
    sol = run_deb_ad(pAm, kap, obs_times, Winit, init_status, tspan, base_params)
    
    # Check solution success
    if !SciMLBase.successful_retcode(sol)
        Turing.@addlogprob! -Inf
        return
    end
    
    # Calculate predicted weight
    predicted_weight = calc_weight_from_sol(sol, pAm, base_params.v, base_params.w)
    
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

# ============================================================================
# SBC FUNCTIONS
# ============================================================================

struct SBCResult
    n_sbc::Int
    n_success::Int
    true_params::DataFrame
    posterior_summaries::DataFrame
    ranks::DataFrame
end

"""
Generate synthetic data from known parameters.
"""
function generate_sbc_data(true_pAm, true_kap, true_sigma_w,
                            obs_times, Winit, init_status, tspan, base_params)
    
    sol = run_deb_ad(true_pAm, true_kap, obs_times, Winit, init_status, tspan, base_params)
    
    if !SciMLBase.successful_retcode(sol)
        return nothing
    end
    
    Weight = calc_weight_from_sol(sol, true_pAm, base_params.v, base_params.w)
    
    if any(isnan.(Weight)) || any(Weight .<= 0)
        return nothing
    end
    
    # Add observation noise
    obs_weight = Weight .+ rand(Normal(0, true_sigma_w), length(Weight))
    obs_weight = max.(obs_weight, 0.001)
    
    return obs_weight
end

"""
Compute rank of true value in posterior samples.
"""
function compute_rank(true_value, posterior_samples)
    return sum(posterior_samples .< true_value)
end

"""
Run SBC with specified number of iterations.
"""
function run_sbc(n_sbc::Int,
                  n_posterior_samples::Int,
                  obs_times::Vector{Float64},
                  Winit::Float64,
                  init_status::Int,
                  tspan::Tuple{Float64, Float64},
                  base_params::DEBEarthwormParams;
                  n_warmup::Int = 200,
                  seed::Union{Int, Nothing} = nothing,
                  show_progress::Bool = true,
                  sampler = NUTS(0.65))  # Can switch to MH() if NUTS fails
    
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Priors for generating true values
    prior_pAm = truncated(Normal(1172.0, 117.2), 800.0, 2000.0)
    prior_kap = Uniform(0.05, 0.25)
    prior_sigma_w = truncated(Normal(0.05, 0.02), 0.001, 0.5)
    
    # Storage
    results = Dict(
        :true_pAm => Float64[],
        :true_kap => Float64[],
        :true_sigma_w => Float64[],
        :post_pAm_mean => Float64[],
        :post_kap_mean => Float64[],
        :post_sigma_w_mean => Float64[],
        :post_pAm_std => Float64[],
        :post_kap_std => Float64[],
        :post_sigma_w_std => Float64[],
        :rank_pAm => Int[],
        :rank_kap => Int[],
        :rank_sigma_w => Int[]
    )
    
    n_success = 0
    n_data_failed = 0
    n_mcmc_failed = 0
    
    println("=" ^ 70)
    println("SIMULATION BASED CALIBRATION")
    println("=" ^ 70)
    println("SBC iterations:        $n_sbc")
    println("Posterior samples:     $n_posterior_samples")
    println("Warmup samples:        $n_warmup")
    println("Sampler:               $(typeof(sampler))")
    println("=" ^ 70)
    
    @showprogress desc="SBC: " enabled=show_progress for i in 1:n_sbc
        
        # Step 1: Draw true parameters
        true_pAm = rand(prior_pAm)
        true_kap = rand(prior_kap)
        true_sigma_w = rand(prior_sigma_w)
        
        # Step 2: Generate synthetic data
        obs_weight = generate_sbc_data(true_pAm, true_kap, true_sigma_w,
                                        obs_times, Winit, init_status, 
                                        tspan, base_params)
        
        if isnothing(obs_weight)
            n_data_failed += 1
            continue
        end
        
        # Step 3: Fit model
        model = deb_turing_model(obs_weight, obs_times, Winit, init_status, 
                                  tspan, base_params)
        
        try
            chain = sample(model, sampler, n_posterior_samples + n_warmup,
                          progress=false)
            
            # Extract samples (skip warmup if using NUTS)
            if sampler isa NUTS
                chain = chain[(n_warmup+1):end, :, :]
            end
            
            pAm_samples = vec(Array(chain[:pAm]))
            kap_samples = vec(Array(chain[:kap]))
            sigma_w_samples = vec(Array(chain[:Sigma_W]))
            
            # Store results
            push!(results[:true_pAm], true_pAm)
            push!(results[:true_kap], true_kap)
            push!(results[:true_sigma_w], true_sigma_w)
            
            push!(results[:post_pAm_mean], mean(pAm_samples))
            push!(results[:post_kap_mean], mean(kap_samples))
            push!(results[:post_sigma_w_mean], mean(sigma_w_samples))
            
            push!(results[:post_pAm_std], std(pAm_samples))
            push!(results[:post_kap_std], std(kap_samples))
            push!(results[:post_sigma_w_std], std(sigma_w_samples))
            
            push!(results[:rank_pAm], compute_rank(true_pAm, pAm_samples))
            push!(results[:rank_kap], compute_rank(true_kap, kap_samples))
            push!(results[:rank_sigma_w], compute_rank(true_sigma_w, sigma_w_samples))
            
            n_success += 1
            
        catch e
            n_mcmc_failed += 1
            continue
        end
    end
    
    println("\n")
    println("Results:")
    println("  Successful:    $n_success / $n_sbc")
    println("  Data gen fail: $n_data_failed")
    println("  MCMC fail:     $n_mcmc_failed")
    
    # Create DataFrames
    true_params = DataFrame(
        iteration = 1:n_success,
        pAm = results[:true_pAm],
        kap = results[:true_kap],
        Sigma_W = results[:true_sigma_w]
    )
    
    posterior_summaries = DataFrame(
        iteration = 1:n_success,
        pAm_mean = results[:post_pAm_mean],
        kap_mean = results[:post_kap_mean],
        Sigma_W_mean = results[:post_sigma_w_mean],
        pAm_std = results[:post_pAm_std],
        kap_std = results[:post_kap_std],
        Sigma_W_std = results[:post_sigma_w_std]
    )
    
    ranks = DataFrame(
        iteration = 1:n_success,
        pAm = results[:rank_pAm],
        kap = results[:rank_kap],
        Sigma_W = results[:rank_sigma_w]
    )
    
    return SBCResult(n_sbc, n_success, true_params, posterior_summaries, ranks)
end

# ============================================================================
# SBC DIAGNOSTICS
# ============================================================================

"""
Print SBC summary with uniformity tests.
"""
function summarize_sbc(result::SBCResult, n_posterior_samples::Int)
    println("=" ^ 70)
    println("SBC SUMMARY")
    println("=" ^ 70)
    println("Successful iterations: $(result.n_success) / $(result.n_sbc)")
    println()
    
    if result.n_success == 0
        println("No successful iterations - cannot compute statistics")
        return
    end
    
    println("RANK UNIFORMITY (p > 0.05 indicates good calibration):")
    println("-" ^ 50)
    
    n_bins = min(10, result.n_success ÷ 5)  # Adjust bins to sample size
    expected_per_bin = result.n_success / n_bins
    
    for param in [:pAm, :kap, :Sigma_W]
        ranks = result.ranks[!, param]
        normalized = ranks ./ n_posterior_samples
        
        # Bin counts
        bin_counts = zeros(Int, n_bins)
        for r in normalized
            idx = min(n_bins, Int(floor(r * n_bins)) + 1)
            bin_counts[idx] += 1
        end
        
        # Chi-squared test
        chi_sq = sum((bin_counts .- expected_per_bin).^2 ./ expected_per_bin)
        p_value = 1 - cdf(Chisq(n_bins - 1), chi_sq)
        
        status = p_value > 0.05 ? "✓" : "✗"
        println("$status $(rpad(string(param), 12)) χ²=$(round(chi_sq, digits=2)), p=$(round(p_value, digits=3))")
    end
    
    println()
    println("PARAMETER RECOVERY:")
    println("-" ^ 50)
    
    for param in [:pAm, :kap, :Sigma_W]
        true_vals = result.true_params[!, param]
        mean_col = Symbol(string(param) * "_mean")
        post_means = result.posterior_summaries[!, mean_col]
        
        bias = mean(post_means .- true_vals)
        rmse = sqrt(mean((post_means .- true_vals).^2))
        correlation = cor(true_vals, post_means)
        
        println("$(rpad(string(param), 12)) Bias=$(round(bias, digits=4)), RMSE=$(round(rmse, digits=4)), r=$(round(correlation, digits=3))")
    end
end

"""
Plot SBC rank histograms.
"""
function plot_sbc_ranks(result::SBCResult, n_posterior_samples::Int)
    n_bins = min(10, result.n_success ÷ 5)
    expected = result.n_success / n_bins
    
    plots = []
    for param in [:pAm, :kap, :Sigma_W]
        ranks = result.ranks[!, param] ./ n_posterior_samples
        
        p = histogram(ranks, bins=n_bins, normalize=:false,
                      xlabel="Normalized rank", ylabel="Count",
                      title=string(param), label=false,
                      color=:steelblue, alpha=0.7)
        hline!(p, [expected], color=:red, lw=2, ls=:dash, label="Expected")
        
        push!(plots, p)
    end
    
    plot(plots..., layout=(1, 3), size=(900, 300),
         plot_title="SBC Rank Histograms")
end

"""
Plot parameter recovery scatter plots.
"""
function plot_sbc_recovery(result::SBCResult)
    plots = []
    
    for param in [:pAm, :kap, :Sigma_W]
        true_vals = result.true_params[!, param]
        mean_col = Symbol(string(param) * "_mean")
        post_means = result.posterior_summaries[!, mean_col]
        
        p = scatter(true_vals, post_means,
                    xlabel="True", ylabel="Estimated",
                    title=string(param), label=false,
                    alpha=0.6, markersize=4)
        
        # Add identity line
        lims = extrema([true_vals; post_means])
        plot!(p, [lims[1], lims[2]], [lims[1], lims[2]], 
              color=:red, lw=2, ls=:dash, label="y=x")
        
        push!(plots, p)
    end
    
    plot(plots..., layout=(1, 3), size=(900, 300),
         plot_title="Parameter Recovery")
end

# ============================================================================
# RUN SBC
# ============================================================================

println("Setting up SBC...")

base_params = DEBEarthwormParams(Texp = 15.0, Dens = 1.0)
Winit = 0.015
init_status = 1
tspan = (0.0, 180.0)
obs_times = collect(0.0:14.0:180.0)

println("Testing single MCMC run first...")

# Test with MH sampler (more robust, doesn't need gradients)
test_data = generate_sbc_data(1172.0, 0.15, 0.02, obs_times, Winit, init_status, tspan, base_params)

if !isnothing(test_data)
    println("✓ Data generation works")
    println("  Test data: $(round.(test_data[1:5], digits=4))...")
    
    model = deb_turing_model(test_data, obs_times, Winit, init_status, tspan, base_params)
    
    println("Testing MH sampler...")
    try
        chain = sample(model, MH(), 100, progress=true)
        println("✓ MH sampling works")
    catch e
        println("✗ MH failed: $e")
    end
else
    println("✗ Data generation failed")
end

# Run SBC with MH sampler (gradient-free, more robust)
println("\nRunning SBC with MH sampler...")

n_posterior = 20000
sbc_result = run_sbc(
    100,                     # SBC iterations
    n_posterior,            # Posterior samples
    obs_times,
    Winit,
    init_status,
    tspan,
    base_params;
    n_warmup = 5000,
    seed = 42,
    show_progress = true,
    sampler = MH()          # Use MH instead of NUTS
)

# Analyze results
if sbc_result.n_success > 0
    summarize_sbc(sbc_result, n_posterior)
    
    p1 = plot_sbc_ranks(sbc_result, n_posterior)
    display(p1)
    
    p2 = plot_sbc_recovery(sbc_result)
    display(p2)
else
    println("No successful SBC iterations - check model specification")
end

# Add this diagnostic function to check how sensitive predictions are to parameters

function check_parameter_sensitivity(base_params, Winit, init_status, tspan, obs_times)
    # Vary pAm while holding kap fixed
    pAm_values = range(800, 2000, length=10)
    kap_fixed = 0.15
    
    p1 = plot(title="Sensitivity to pAm (kap=0.15)", xlabel="Time", ylabel="Weight")
    
    for pAm in pAm_values
        sol = run_deb_ad(pAm, kap_fixed, obs_times, Winit, init_status, tspan, base_params)
        if SciMLBase.successful_retcode(sol)
            W = calc_weight_from_sol(sol, pAm, base_params.v, base_params.w)
            plot!(p1, obs_times, W, label="pAm=$(Int(pAm))", alpha=0.7)
        end
    end
    
    # Vary kap while holding pAm fixed
    kap_values = range(0.05, 0.25, length=10)
    pAm_fixed = 1172.0
    
    p2 = plot(title="Sensitivity to kap (pAm=1172)", xlabel="Time", ylabel="Weight")
    
    for kap in kap_values
        sol = run_deb_ad(pAm_fixed, kap, obs_times, Winit, init_status, tspan, base_params)
        if SciMLBase.successful_retcode(sol)
            W = calc_weight_from_sol(sol, pAm_fixed, base_params.v, base_params.w)
            plot!(p2, obs_times, W, label="kap=$(round(kap, digits=2))", alpha=0.7)
        end
    end
    
    plot(p1, p2, layout=(1,2), size=(1000, 400))
end

# Run sensitivity check
check_parameter_sensitivity(base_params, Winit, init_status, tspan, obs_times)