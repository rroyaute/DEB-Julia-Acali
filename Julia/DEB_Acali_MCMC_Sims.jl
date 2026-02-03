# ============================================================================
# Monte Carlo Simulations for DEB Apporectodea caliginosa
# ============================================================================

# Load the base DEB model
include("DEB_Acali_wfood.jl")

using Distributions
using Random
using DataFrames
using Statistics
using ProgressMeter

# ============================================================================
# PRIOR DISTRIBUTIONS (from MCSim .in file)
# ============================================================================

"""
    PriorDistributions

Prior distributions matching MCSim specification:
- pAm:         TruncNormal(1172, 117.2, 800, 8000)
- r_pAm_pM:    TruncNormal(1.5, 0.2, 0.01, 100)
- kap:         Uniform(0.05, 0.25)
- Ehb:         TruncNormal(0.5, 0.5, 0.1, 10)
- Ehp:         TruncNormal(100, 50, 1, 1000)
- E_coc:       TruncNormal(200, 100, 50, 5000)
- rOM_ClxHorse: TruncNormal(0.2, 0.1, 0, 1)

Error model:
- Sigma_W:     TruncNormal(5, 1, 0.01, 100)
"""
Base.@kwdef struct PriorDistributions
    # Energy Assimilation
    pAm::Distribution = truncated(Normal(1172.0, 117.2), 800.0, 8000.0)
    r_pAm_pM::Distribution = truncated(Normal(1.5, 0.2), 0.01, 100.0)
    
    # Allocation
    kap::Distribution = Uniform(0.05, 0.95)
    
    # Maturity (sampled directly, not derived)
    Ehb::Distribution = truncated(Normal(0.5, 0.5), 0.1, 10.0)
    Ehp::Distribution = truncated(Normal(100.0, 50.0), 1.0, 1000.0)
    
    # Reproduction
    E_coc::Distribution = truncated(Normal(200.0, 100.0), 50.0, 5000.0)
    
    # Environment
    rOM_ClxHorse::Distribution = truncated(Normal(0.2, 0.1), 0.0, 1.0)
    
    # Error model (for likelihood, not used in forward simulation)
    Sigma_W::Distribution = truncated(Normal(5.0, 1.0), 0.01, 100.0)
end

"""
    SampledParameters

Container for a single Monte Carlo sample.
Includes both sampled and derived parameters.
"""
struct SampledParameters
    # Sampled from priors
    pAm::Float64
    r_pAm_pM::Float64
    kap::Float64
    Ehb::Float64
    Ehp::Float64
    E_coc::Float64
    rOM_ClxHorse::Float64
    Sigma_W::Float64
    
    # Derived
    pM::Float64
    Em::Float64
    Km::Float64
end

"""
    sample_from_priors(priors::PriorDistributions, base_params::DEBEarthwormParams)

Sample parameters from prior distributions.
Returns (DEBEarthwormParams, SampledParameters) tuple.
"""
function sample_from_priors(priors::PriorDistributions, base_params::DEBEarthwormParams)
    # Sample from priors
    pAm = rand(priors.pAm)
    r_pAm_pM = rand(priors.r_pAm_pM)
    kap = rand(priors.kap)
    Ehb = rand(priors.Ehb)
    Ehp = rand(priors.Ehp)
    E_coc = rand(priors.E_coc)
    rOM_ClxHorse = rand(priors.rOM_ClxHorse)
    Sigma_W = rand(priors.Sigma_W)
    
    # Derive dependent parameters
    pM = pAm * r_pAm_pM
    Em = pAm / base_params.v
    Km = pM / base_params.Eg
    
    # Create new DEBEarthwormParams with sampled values
    # Note: Ehb and Ehp are sampled directly, so we need to back-calculate
    # alpha_Ehb and beta_Ehp for compatibility with the model
    alpha_Ehb = Ehb / E_coc
    beta_Ehp = Ehp / Ehb
    
    new_params = DEBEarthwormParams(
        # Sampled parameters
        pAm = pAm,
        r_pAm_pM = r_pAm_pM,
        kap = kap,
        E_coc = E_coc,
        rOM_ClxHorse = rOM_ClxHorse,
        alpha_Ehb = alpha_Ehb,
        beta_Ehp = beta_Ehp,
        
        # Fixed parameters (from base)
        muOM = base_params.muOM,
        Fm = base_params.Fm,
        kapX = base_params.kapX,
        v = base_params.v,
        pT = base_params.pT,
        Eg = base_params.Eg,
        Shape = base_params.Shape,
        w = base_params.w,
        kJ = base_params.kJ,
        L_coc = base_params.L_coc,
        kapR = base_params.kapR,
        TA = base_params.TA,
        TAH = base_params.TAH,
        TH = base_params.TH,
        Tref = base_params.Tref,
        Texp = base_params.Texp,
        Dens = base_params.Dens
    )
    
    sampled = SampledParameters(
        pAm, r_pAm_pM, kap, Ehb, Ehp, E_coc, rOM_ClxHorse, Sigma_W,
        pM, Em, Km
    )
    
    return new_params, sampled
end

# ============================================================================
# MONTE CARLO SIMULATION ENGINE
# ============================================================================

"""
    MCResult

Container for Monte Carlo simulation results.
"""
struct MCResult
    n_samples::Int
    n_success::Int
    sampled_params::Vector{SampledParameters}
    trajectories::Vector{Union{NamedTuple, Nothing}}
    summary::DataFrame
end

"""
    run_single_simulation(params, Winit, init_status, tspan, feeding; kwargs...)

Run a single DEB simulation with given parameters.
"""
function run_single_simulation(params::DEBEarthwormParams,
                                Winit::Float64,
                                init_status::Int,
                                tspan::Tuple{Float64, Float64},
                                feeding::FeedingSchedule;
                                OM_soil_init::Float64 = 10.0,
                                OM_horse_init::Float64 = 3.0,
                                save_trajectory::Bool = false,
                                saveat::Float64 = 1.0)
    
    # Derive parameters
    derived = derive_params(params)
    
    # Initialize state
    u0 = initialize_earthworm(Winit, init_status, params, derived;
                              OM_soil_init = OM_soil_init,
                              OM_horse_init = OM_horse_init)
    
    # Create quiet callbacks
    feeding_quiet = FeedingSchedule(
        horse_interval = feeding.horse_interval,
        horse_amount = feeding.horse_amount,
        horse_start = feeding.horse_start,
        soil_interval = feeding.soil_interval,
        soil_amount = feeding.soil_amount,
        soil_start = feeding.soil_start,
        verbose = false
    )
    callbacks = create_feeding_callbacks(tspan, feeding_quiet)
    
    # Solve ODE
    prob = DE.ODEProblem(deb_earthworm!, u0, tspan, (params, derived))
    
    try
        sol = DE.solve(prob, DE.Tsit5(), callback=callbacks, saveat=saveat,
                       abstol=1e-8, reltol=1e-6)
        
        if sol.retcode != :Success
            return nothing
        end
        
        outputs = calc_outputs(sol, params, derived)
        
        if save_trajectory
            return outputs
        else
            t_end = length(outputs.t)
            return (
                Weight_final = outputs.Weight[t_end],
                Size_final = outputs.Size[t_end],
                Reproduction_final = outputs.Reproduction[t_end],
                Maturity_final = outputs.Maturity[t_end]
            )
        end
    catch e
        return nothing
    end
end

"""
    run_monte_carlo(n_samples, base_params, Winit, init_status, tspan, feeding, priors; kwargs...)

Run Monte Carlo simulations by sampling from prior distributions.

Arguments:
- n_samples: Number of Monte Carlo samples
- base_params: Base DEBEarthwormParams (for fixed parameters)
- Winit: Initial weight (g)
- init_status: 1 = juvenile, 3 = adult
- tspan: Simulation time span
- feeding: FeedingSchedule
- priors: PriorDistributions

Keyword arguments:
- OM_soil_init: Initial soil organic matter (g)
- OM_horse_init: Initial horse manure organic matter (g)
- save_trajectories: If true, save full time series
- seed: Random seed for reproducibility
- show_progress: Show progress bar

Returns:
- MCResult
"""
function run_monte_carlo(n_samples::Int,
                          base_params::DEBEarthwormParams,
                          Winit::Float64,
                          init_status::Int,
                          tspan::Tuple{Float64, Float64},
                          feeding::FeedingSchedule,
                          priors::PriorDistributions;
                          OM_soil_init::Float64 = 10.0,
                          OM_horse_init::Float64 = 3.0,
                          save_trajectories::Bool = false,
                          seed::Union{Int, Nothing} = nothing,
                          show_progress::Bool = true)
    
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Storage
    sampled_params = Vector{SampledParameters}(undef, n_samples)
    trajectories = Vector{Union{NamedTuple, Nothing}}(undef, n_samples)
    
    # Results storage
    Weight_final = Vector{Union{Float64, Missing}}(undef, n_samples)
    Size_final = Vector{Union{Float64, Missing}}(undef, n_samples)
    Reproduction_final = Vector{Union{Float64, Missing}}(undef, n_samples)
    Maturity_final = Vector{Union{Float64, Missing}}(undef, n_samples)
    
    n_success = 0
    
    # Run simulations with progress bar
    @showprogress desc="Monte Carlo: " enabled=show_progress for i in 1:n_samples
        # Sample parameters
        params_i, sampled_i = sample_from_priors(priors, base_params)
        sampled_params[i] = sampled_i
        
        # Run simulation
        result = run_single_simulation(params_i, Winit, init_status, tspan, feeding;
                                        OM_soil_init = OM_soil_init,
                                        OM_horse_init = OM_horse_init,
                                        save_trajectory = save_trajectories)
        
        if isnothing(result)
            trajectories[i] = nothing
            Weight_final[i] = missing
            Size_final[i] = missing
            Reproduction_final[i] = missing
            Maturity_final[i] = missing
        else
            n_success += 1
            if save_trajectories
                trajectories[i] = result
                Weight_final[i] = result.Weight[end]
                Size_final[i] = result.Size[end]
                Reproduction_final[i] = result.Reproduction[end]
                Maturity_final[i] = result.Maturity[end]
            else
                trajectories[i] = nothing
                Weight_final[i] = result.Weight_final
                Size_final[i] = result.Size_final
                Reproduction_final[i] = result.Reproduction_final
                Maturity_final[i] = result.Maturity_final
            end
        end
    end
    
    # Create summary DataFrame
    summary_df = DataFrame(
        sample = 1:n_samples,
        Weight_final = Weight_final,
        Size_final = Size_final,
        Reproduction_final = Reproduction_final,
        Maturity_final = Maturity_final,
        # Sampled parameters
        pAm = [p.pAm for p in sampled_params],
        r_pAm_pM = [p.r_pAm_pM for p in sampled_params],
        kap = [p.kap for p in sampled_params],
        Ehb = [p.Ehb for p in sampled_params],
        Ehp = [p.Ehp for p in sampled_params],
        E_coc = [p.E_coc for p in sampled_params],
        rOM_ClxHorse = [p.rOM_ClxHorse for p in sampled_params],
        Sigma_W = [p.Sigma_W for p in sampled_params],
        # Derived parameters
        pM = [p.pM for p in sampled_params],
        Em = [p.Em for p in sampled_params]
    )
    
    return MCResult(n_samples, n_success, sampled_params, trajectories, summary_df)
end

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

"""
    summarize_mc(result::MCResult)

Print summary statistics of Monte Carlo results.
"""
function summarize_mc(result::MCResult)
    df = result.summary
    
    println("=" ^ 70)
    println("MONTE CARLO SUMMARY")
    println("=" ^ 70)
    println("Total samples: $(result.n_samples)")
    println("Successful:    $(result.n_success) ($(round(100*result.n_success/result.n_samples, digits=1))%)")
    println("Failed:        $(result.n_samples - result.n_success)")
    println()
    
    if result.n_success > 0
        println("OUTPUT STATISTICS (at final time):")
        println("-" ^ 50)
        
        for (col, unit) in [(:Weight_final, "g"), 
                            (:Size_final, "cm"), 
                            (:Reproduction_final, "cocoons"),
                            (:Maturity_final, "J")]
            vals = collect(skipmissing(df[!, col]))
            if col == :Weight_final
                vals_display = vals .* 1000  # Convert to mg
                unit = "mg"
            else
                vals_display = vals
            end
            
            println("$(rpad(string(col), 20))")
            println("  Mean:   $(round(mean(vals_display), digits=3)) $unit")
            println("  Std:    $(round(std(vals_display), digits=3)) $unit")
            println("  Median: $(round(median(vals_display), digits=3)) $unit")
            println("  2.5%:   $(round(quantile(vals_display, 0.025), digits=3)) $unit")
            println("  97.5%:  $(round(quantile(vals_display, 0.975), digits=3)) $unit")
            println()
        end
        
        println("SAMPLED PARAMETER STATISTICS:")
        println("-" ^ 50)
        
        for col in [:pAm, :r_pAm_pM, :kap, :Ehb, :Ehp, :E_coc, :rOM_ClxHorse]
            vals = df[!, col]
            println("$(rpad(string(col), 15)) Mean: $(round(mean(vals), digits=3)), SD: $(round(std(vals), digits=3))")
        end
    end
end

"""
    compute_quantiles(result::MCResult; probs=[0.025, 0.25, 0.5, 0.75, 0.975])

Compute quantiles of outputs across Monte Carlo samples.
"""
function compute_quantiles(result::MCResult; probs=[0.025, 0.25, 0.5, 0.75, 0.975])
    df = result.summary
    
    quantiles = Dict{Symbol, Vector{Float64}}()
    for col in [:Weight_final, :Size_final, :Reproduction_final, :Maturity_final]
        vals = collect(skipmissing(df[!, col]))
        quantiles[col] = quantile(vals, probs)
    end
    
    return quantiles, probs
end

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

"""
    plot_prior_samples(result::MCResult, priors::PriorDistributions)

Plot histograms of sampled parameter values vs prior distributions.
"""
function plot_prior_samples(result::MCResult, priors::PriorDistributions)
    df = result.summary
    
    param_info = [
        (:pAm, priors.pAm, "pAm (J/d/cm²)"),
        (:r_pAm_pM, priors.r_pAm_pM, "r_pAm_pM (-)"),
        (:kap, priors.kap, "κ (-)"),
        (:Ehb, priors.Ehb, "Ehb (J)"),
        (:Ehp, priors.Ehp, "Ehp (J)"),
        (:E_coc, priors.E_coc, "E_coc (J)"),
        (:rOM_ClxHorse, priors.rOM_ClxHorse, "rOM_ClxHorse (-)")
    ]
    
    plots = []
    for (col, prior, label) in param_info
        vals = df[!, col]
        
        p = histogram(vals, normalize=:pdf, alpha=0.6, 
                      label="Samples", xlabel=label, ylabel="Density")
        
        # Overlay prior density
        x_range = range(minimum(vals) * 0.9, maximum(vals) * 1.1, length=100)
        plot!(p, x_range, pdf.(prior, x_range), lw=2, color=:red, label="Prior")
        
        push!(plots, p)
    end
    
    # Add empty plot for layout (3x3 grid with 7 plots)
    p_empty1 = plot(legend=false, axis=false, grid=false, framestyle=:none)
    p_empty2 = plot(legend=false, axis=false, grid=false, framestyle=:none)
    push!(plots, p_empty1)
    push!(plots, p_empty2)
    
    plot(plots..., layout=(3, 3), size=(1000, 800),
         plot_title="Prior Distributions vs Samples")
end

"""
    plot_mc_histograms(result::MCResult)

Plot histograms of final outputs from Monte Carlo simulations.
"""
function plot_mc_histograms(result::MCResult)
    df = result.summary
    
    p1 = histogram(collect(skipmissing(df.Weight_final)) .* 1000, 
                   xlabel="Final Weight (mg)", ylabel="Count",
                   label=false, color=:brown, alpha=0.7, bins=30)
    
    p2 = histogram(collect(skipmissing(df.Size_final)),
                   xlabel="Final Length (cm)", ylabel="Count", 
                   label=false, color=:green, alpha=0.7, bins=30)
    
    p3 = histogram(collect(skipmissing(df.Reproduction_final)),
                   xlabel="Total Cocoons", ylabel="Count",
                   label=false, color=:purple, alpha=0.7, bins=30)
    
    p4 = histogram(collect(skipmissing(df.Maturity_final)),
                   xlabel="Final Maturity (J)", ylabel="Count",
                   label=false, color=:orange, alpha=0.7, bins=30)
    
    plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600),
         plot_title="Monte Carlo Output Distributions (n=$(result.n_success))")
end

"""
    plot_mc_trajectories(result::MCResult, derived; n_show=100, alpha=0.1)

Plot Monte Carlo trajectories (requires save_trajectories=true).
"""
function plot_mc_trajectories(result::MCResult, derived::DEBDerivedParams;
                               n_show::Int = 100, alpha::Float64 = 0.1)
    
    # Filter successful trajectories
    valid_idx = findall(x -> !isnothing(x), result.trajectories)
    
    if isempty(valid_idx)
        error("No trajectories saved. Run with save_trajectories=true")
    end
    
    n_traj = min(n_show, length(valid_idx))
    show_idx = valid_idx[1:n_traj]
    
    # Weight
    p1 = plot(ylabel="Weight (mg)", legend=false, title="Weight")
    for i in show_idx
        traj = result.trajectories[i]
        plot!(p1, traj.t, traj.Weight .* 1000, alpha=alpha, color=:brown)
    end
    
    # Size
    p2 = plot(ylabel="Length (cm)", legend=false, title="Physical Length")
    for i in show_idx
        traj = result.trajectories[i]
        plot!(p2, traj.t, traj.Size, alpha=alpha, color=:green)
    end
    
    # Reproduction
    p3 = plot(ylabel="# Cocoons", xlabel="Time (days)", legend=false, title="Reproduction")
    for i in show_idx
        traj = result.trajectories[i]
        plot!(p3, traj.t, traj.Reproduction, alpha=alpha, color=:purple)
    end
    
    # Maturity
    p4 = plot(ylabel="Maturity (J)", xlabel="Time (days)", legend=false, title="Maturity")
    for i in show_idx
        traj = result.trajectories[i]
        plot!(p4, traj.t, traj.Maturity, alpha=alpha, color=:blue)
    end
    hline!(p4, [derived.Ehp], color=:red, ls=:dash, lw=2, label="Puberty")
    
    plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700),
         plot_title="Monte Carlo Trajectories (n=$n_traj)")
end

"""
    plot_trajectory_quantiles(result::MCResult; probs=[0.025, 0.25, 0.5, 0.75, 0.975])

Plot trajectory quantiles over time (requires save_trajectories=true).
"""
function plot_trajectory_quantiles(result::MCResult; 
                                    probs::Vector{Float64}=[0.025, 0.25, 0.5, 0.75, 0.975])
    
    valid_idx = findall(x -> !isnothing(x), result.trajectories)
    
    if isempty(valid_idx)
        error("No trajectories saved. Run with save_trajectories=true")
    end
    
    # Get time vector from first valid trajectory
    t = result.trajectories[valid_idx[1]].t
    n_t = length(t)
    n_valid = length(valid_idx)
    
    # Collect all trajectories into matrices
    Weight_mat = zeros(n_valid, n_t)
    Size_mat = zeros(n_valid, n_t)
    Repro_mat = zeros(n_valid, n_t)
    
    for (j, i) in enumerate(valid_idx)
        traj = result.trajectories[i]
        Weight_mat[j, :] = traj.Weight
        Size_mat[j, :] = traj.Size
        Repro_mat[j, :] = traj.Reproduction
    end
    
    # Compute quantiles at each time point
    function time_quantiles(mat, probs)
        n_probs = length(probs)
        Q = zeros(n_probs, n_t)
        for k in 1:n_t
            Q[:, k] = quantile(mat[:, k], probs)
        end
        return Q
    end
    
    Weight_Q = time_quantiles(Weight_mat, probs)
    Size_Q = time_quantiles(Size_mat, probs)
    Repro_Q = time_quantiles(Repro_mat, probs)
    
    # Plot with ribbons
    mid_idx = div(length(probs) + 1, 2)  # Median index
    
    # Weight
    p1 = plot(t, Weight_Q[mid_idx, :] .* 1000, lw=2, color=:brown, 
              label="Median", ylabel="Weight (mg)", legend=:topleft)
    plot!(p1, t, Weight_Q[1, :] .* 1000, fillrange=Weight_Q[end, :] .* 1000,
          alpha=0.2, color=:brown, label="95% CI")
    plot!(p1, t, Weight_Q[2, :] .* 1000, fillrange=Weight_Q[end-1, :] .* 1000,
          alpha=0.3, color=:brown, label="50% CI")
    
    # Size
    p2 = plot(t, Size_Q[mid_idx, :], lw=2, color=:green,
              label="Median", ylabel="Length (cm)", legend=:topleft)
    plot!(p2, t, Size_Q[1, :], fillrange=Size_Q[end, :],
          alpha=0.2, color=:green, label="95% CI")
    plot!(p2, t, Size_Q[2, :], fillrange=Size_Q[end-1, :],
          alpha=0.3, color=:green, label="50% CI")
    
    # Reproduction
    p3 = plot(t, Repro_Q[mid_idx, :], lw=2, color=:purple,
              label="Median", ylabel="# Cocoons", xlabel="Time (days)", legend=:topleft)
    plot!(p3, t, Repro_Q[1, :], fillrange=Repro_Q[end, :],
          alpha=0.2, color=:purple, label="95% CI")
    plot!(p3, t, Repro_Q[2, :], fillrange=Repro_Q[end-1, :],
          alpha=0.3, color=:purple, label="50% CI")
    
    plot(p1, p2, p3, layout=(3, 1), size=(800, 700),
         plot_title="Trajectory Quantiles (n=$(n_valid))")
end

"""
    plot_parameter_sensitivity(result::MCResult, output_var::Symbol)

Scatter plots showing parameter sensitivity for a given output.
"""
function plot_parameter_sensitivity(result::MCResult, output_var::Symbol)
    df = result.summary
    
    # Filter to non-missing outputs
    valid = .!ismissing.(df[!, output_var])
    df_valid = df[valid, :]
    
    param_cols = [:pAm, :r_pAm_pM, :kap, :Ehb, :Ehp, :E_coc, :rOM_ClxHorse]
    
    plots = []
    for param in param_cols
        y_vals = df_valid[!, output_var]
        if output_var == :Weight_final
            y_vals = y_vals .* 1000  # Convert to mg
        end
        
        p = scatter(df_valid[!, param], y_vals,
                    xlabel=string(param), ylabel="",
                    label=false, alpha=0.3, markersize=2)
        push!(plots, p)
    end
    
    # Add empty plot for layout
    p_empty = plot(legend=false, axis=false, grid=false, framestyle=:none)
    push!(plots, p_empty)
    
    y_label = output_var == :Weight_final ? "Weight (mg)" : string(output_var)
    
    plot(plots..., layout=(2, 4), size=(1200, 500),
         plot_title="Parameter Sensitivity: $y_label")
end

"""
    plot_correlation_matrix(result::MCResult)

Plot correlation matrix between parameters and outputs.
"""
function plot_correlation_matrix(result::MCResult)
    df = result.summary
    
    # Select columns for correlation
    param_cols = [:pAm, :r_pAm_pM, :kap, :Ehb, :Ehp, :E_coc, :rOM_ClxHorse]
    output_cols = [:Weight_final, :Size_final, :Reproduction_final, :Maturity_final]
    
    # Filter to complete cases
    all_cols = vcat(param_cols, output_cols)
    valid = completecases(df[!, all_cols])
    df_valid = df[valid, all_cols]
    
    # Compute correlation matrix
    data_mat = Matrix(df_valid)
    cor_mat = cor(data_mat)
    
    # Create heatmap
    labels = string.(all_cols)
    heatmap(labels, labels, cor_mat,
            color=:RdBu, clims=(-1, 1),
            size=(800, 700),
            title="Parameter-Output Correlations",
            xrotation=45)
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# Set up priors
priors = PriorDistributions()

# Base parameters (fixed parameters come from here)
base_params = DEBEarthwormParams(Texp = 15.0, Dens = 1.0)

# Initial conditions
Winit = 0.015      # 15 mg hatchling
init_status = 1    # Juvenile

# Time span
tspan = (0.0, 365.0)

# Feeding schedule
feeding = FeedingSchedule(
    horse_interval = 14.0,
    horse_amount = 2.7,
    horse_start = 14.0,
    soil_interval = 28.0,
    soil_amount = 13.0,
    soil_start = 28.0,
    verbose = false
)

# Run Monte Carlo
println("Starting Monte Carlo simulations...")
println("Priors:")
println("  pAm ~ TruncNormal(1172, 117.2, 800, 8000)")
println("  r_pAm_pM ~ TruncNormal(1.5, 0.2, 0.01, 100)")
println("  kap ~ Uniform(0.05, 0.25)")
println("  Ehb ~ TruncNormal(0.5, 0.5, 0.1, 10)")
println("  Ehp ~ TruncNormal(100, 50, 1, 1000)")
println("  E_coc ~ TruncNormal(200, 100, 50, 5000)")
println("  rOM_ClxHorse ~ TruncNormal(0.2, 0.1, 0, 1)")
println()

mc_result = run_monte_carlo(
    1000,                      # Number of samples
    base_params,
    Winit,
    init_status,
    tspan,
    feeding,
    priors;
    OM_soil_init = 10.0,
    OM_horse_init = 3.0,
    save_trajectories = true,  # Enable trajectory saving for plots
    seed = 42,                 # Reproducibility
    show_progress = true
)

# Print summary
summarize_mc(mc_result)

# Get derived parameters for plotting
derived_base = derive_params(base_params)

# Generate plots
println("\nGenerating plots...")

p_hist = plot_mc_histograms(mc_result)
display(p_hist)
# savefig(p_hist, "mc_histograms.png")

p_traj = plot_mc_trajectories(mc_result, derived_base; n_show=200, alpha=0.05)
display(p_traj)
# savefig(p_traj, "mc_trajectories.png")

p_quant = plot_trajectory_quantiles(mc_result)
display(p_quant)
# savefig(p_quant, "mc_quantiles.png")

p_sens = plot_parameter_sensitivity(mc_result, :Weight_final)
display(p_sens)
# savefig(p_sens, "sensitivity_weight.png")

p_prior = plot_prior_samples(mc_result, priors)
display(p_prior)
# savefig(p_prior, "prior_samples.png")

println("\nMonte Carlo simulations complete!")