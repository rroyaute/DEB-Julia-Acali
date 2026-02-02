# Dynamic Energy Budget Modeling in Julia: A Tutorial for R Users

## Introduction

This tutorial explains two Julia files that implement a Dynamic Energy Budget (DEB) model for the earthworm *Apporectodea caliginosa*:

1. **`DEB_Acali_wfood.jl`** — The core DEB model with ODE dynamics and feeding events
2. **`DEB_Acali_MC_Sims.jl`** — Monte Carlo simulations sampling from prior distributions

If you're coming from R with experience in Bayesian packages like `brms`, `MCMCglmm`, or `Stan`, this guide will help you understand Julia syntax and concepts through comparisons to R.

---

## Part 1: Julia Basics for R Users

### 1.1 Key Syntax Differences

| Concept | R | Julia |
|---------|---|-------|
| Assignment | `x <- 5` or `x = 5` | `x = 5` |
| Function definition | `f <- function(x) { x^2 }` | `f(x) = x^2` or `function f(x) ... end` |
| Exponentiation | `x^2` | `x^2` (same!) |
| Array indexing | `x[1]` (1-based) | `x[1]` (also 1-based!) |
| Sequence | `1:10` | `1:10` (same!) |
| Vector creation | `c(1, 2, 3)` | `[1, 2, 3]` |
| Boolean AND/OR | `&` / `|` | `&&` / `||` (short-circuit) |
| If-else | `if (cond) { } else { }` | `if cond ... else ... end` |
| For loop | `for (i in 1:10) { }` | `for i in 1:10 ... end` |
| Comments | `# comment` | `# comment` (same!) |
| NULL equivalent | `NULL` | `nothing` |
| NA equivalent | `NA` | `missing` |
| Print | `print(x)` | `println(x)` |
| Package loading | `library(pkg)` | `using Pkg` |
| Help | `?function` | `?function` (same!) |

### 1.2 Type Annotations

Julia is optionally typed. You'll see syntax like:

```julia
x::Float64 = 3.14
```

This declares `x` as a 64-bit floating point number. The `::` means "is of type". This is similar to type hints in Python, but Julia actually uses them for performance optimization.

**R equivalent:** R doesn't have explicit types, but think of it as:
```r
x <- 3.14  # R infers this is numeric
```

### 1.3 Structs (Julia's equivalent to R's lists/S4 classes)

```julia
struct MyStruct
    field1::Float64
    field2::String
end
```

This is similar to creating an S4 class in R or a named list. To create an instance:

```julia
obj = MyStruct(3.14, "hello")
obj.field1  # Access field (like obj$field1 in R)
```

**R equivalent:**
```r
# Using a list
obj <- list(field1 = 3.14, field2 = "hello")
obj$field1

# Or S4 class
setClass("MyStruct", slots = c(field1 = "numeric", field2 = "character"))
```

### 1.4 The `@kwdef` Macro

```julia
Base.@kwdef struct DEBParams
    pAm::Float64 = 1712.41
    v::Float64 = 0.018505
end
```

The `@kwdef` macro automatically creates a constructor with keyword arguments and default values. This is like creating a function in R with default arguments:

```r
# R equivalent concept
create_params <- function(pAm = 1712.41, v = 0.018505) {
  list(pAm = pAm, v = v)
}
```

You can then create instances with:
```julia
# Use all defaults
params = DEBParams()

# Override some defaults
params = DEBParams(pAm = 2000.0)
```

### 1.5 Functions

Julia functions can be defined in multiple ways:

```julia
# Long form (like R)
function add(a, b)
    return a + b
end

# Short form (one-liner)
add(a, b) = a + b

# Anonymous function (like R's function(x) x^2)
f = x -> x^2
```

**Important:** The last expression in a Julia function is automatically returned (like R), but you can also use explicit `return`.

### 1.6 Multiple Dispatch

Julia's superpower is multiple dispatch — the same function name can have different implementations based on argument types:

```julia
process(x::Int) = "Got an integer"
process(x::Float64) = "Got a float"
process(x::String) = "Got a string"

process(5)      # "Got an integer"
process(5.0)    # "Got a float"
process("hi")   # "Got a string"
```

This is like R's S3/S4 method dispatch but more flexible.

### 1.7 Broadcasting (Vectorization)

Julia uses `.` for element-wise operations:

```julia
x = [1, 2, 3]
x .^ 2      # [1, 4, 9] — element-wise squaring
sin.(x)     # Apply sin to each element
x .+ 1      # Add 1 to each element
```

**R equivalent:**
```r
x <- c(1, 2, 3)
x^2         # R automatically vectorizes
sin(x)
x + 1
```

In R, vectorization is automatic. In Julia, you explicitly request it with `.`

---

## Part 2: Understanding `DEB_Acali_wfood.jl`

Let's walk through the DEB model file section by section.

### 2.1 Package Imports

```julia
import DifferentialEquations as DE
using Plots
```

**What it does:**
- `import ... as DE` — Loads DifferentialEquations package with alias `DE` (like Python's `import numpy as np`)
- `using Plots` — Loads Plots and brings all exported functions into scope

**R equivalent:**
```r
library(deSolve)  # For ODEs
library(ggplot2)  # For plotting
```

**Why `import` vs `using`?**
- `using Pkg` — All exported functions available directly (like `library()` in R)
- `import Pkg` — Must use `Pkg.function_name()` (more explicit)
- `import Pkg as P` — Use alias `P.function_name()`

### 2.2 Parameter Structure

```julia
Base.@kwdef struct DEBEarthwormParams
    # --- Environment ---
    muOM::Float64 = 11700.0       # Energy in 1g OM from horse dung (J/g)
    rOM_ClxHorse::Float64 = 0.3   # Ratio of energy: soil OM vs horse dung OM
    
    # --- Energy Assimilation ---
    Fm::Float64 = 0.5             # Max specific searching rate (g/d/g)
    kapX::Float64 = 0.258         # Digestion efficiency (-)
    pAm::Float64 = 1712.41        # {p_Am}, spec assimilation flux (J/d/cm²)
    # ... more parameters ...
end
```

**What it does:**
This defines a container (struct) for all DEB model parameters with default values. Think of it as a structured list in R that holds all your model constants.

**R equivalent:**
```r
# As a list
deb_params <- list(
  muOM = 11700.0,
  rOM_ClxHorse = 0.3,
  Fm = 0.5,
  kapX = 0.258,
  pAm = 1712.41
  # ...
)

# Or as a function with defaults
create_deb_params <- function(
  muOM = 11700.0,
  rOM_ClxHorse = 0.3,
  # ...
) {
  list(muOM = muOM, rOM_ClxHorse = rOM_ClxHorse, ...)
}
```

**Why use a struct?**
1. **Type safety** — Julia knows exactly what types to expect, enabling faster code
2. **Documentation** — All parameters in one place with comments
3. **Immutability** — By default, structs can't be modified after creation (prevents bugs)

### 2.3 Derived Parameters

```julia
struct DEBDerivedParams
    pM::Float64       # Derived from pAm * r_pAm_pM
    Ehb::Float64      # Derived from alpha_Ehb * E_coc
    Ehp::Float64      # Derived from beta_Ehp * Ehb
    Em::Float64       # Derived from pAm / v
    Km::Float64       # Derived from pM / Eg
end

function derive_params(p::DEBEarthwormParams)
    pM = p.pAm * p.r_pAm_pM
    Ehb = p.alpha_Ehb * p.E_coc
    Ehp = p.beta_Ehp * Ehb
    Em = p.pAm / p.v
    Km = pM / p.Eg
    return DEBDerivedParams(pM, Ehb, Ehp, Em, Km)
end
```

**What it does:**
Some DEB parameters are derived from primary parameters. This function calculates them.

**Note the syntax:**
- `p::DEBEarthwormParams` — The argument `p` must be of type `DEBEarthwormParams`
- `p.pAm` — Access field `pAm` from struct `p` (like `p$pAm` in R)

**R equivalent:**
```r
derive_params <- function(p) {
  list(
    pM = p$pAm * p$r_pAm_pM,
    Ehb = p$alpha_Ehb * p$E_coc,
    Ehp = p$beta_Ehp * p$Ehb,
    Em = p$pAm / p$v,
    Km = pM / p$Eg
  )
}
```

### 2.4 Temperature Correction (Arrhenius)

```julia
function temperature_correction(Texp_C, p::DEBEarthwormParams)
    T = Texp_C + 273.15  # Convert Celsius to Kelvin
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
```

**What it does:**
Implements the Arrhenius temperature correction with an upper boundary. This is standard DEB theory — metabolic rates depend on temperature.

**Julia syntax notes:**
- `if ... else ... end` — No braces like R, but requires `end` keyword
- Multi-line expressions can span lines without special characters

**R equivalent:**
```r
temperature_correction <- function(Texp_C, p) {
  T <- Texp_C + 273.15
  sA <- exp(p$TA / p$Tref - p$TA / T)
  srH <- (1 + exp(p$TAH / p$TH - p$TAH / p$Tref)) / 
         (1 + exp(p$TAH / p$TH - p$TAH / T))
  
  if (T >= p$Tref) {
    Tc <- sA * srH
  } else {
    Tc <- sA
  }
  
  return(Tc)
}
```

### 2.5 Functional Response

```julia
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
```

**What it does:**
Calculates the Type II functional response — how much the earthworm can actually eat given available food. This is the classic Holling Type II response: `f = X / (1 + X)`.

**Julia syntax notes:**
- `return freal, Qf_soil, Qf_horse` — Returns multiple values as a tuple
- You can unpack: `f, qs, qh = functional_response(...)`

**R equivalent:**
```r
functional_response <- function(Weight, OM_soil, OM_horse, L, pAm_t, p, d) {
  Qfmax <- pAm_t * L^2 / p$kapX
  Qf_soil <- p$Fm * Weight * (p$muOM * p$rOM_ClxHorse * OM_soil / p$Dens)
  Qf_horse <- p$Fm * Weight * (p$muOM * OM_horse / p$Dens)
  Qf <- Qf_soil + Qf_horse
  X <- Qf / (0.5 * Qfmax)
  freal <- X / (1 + X)
  
  return(list(freal = freal, Qf_soil = Qf_soil, Qf_horse = Qf_horse))
}
```

### 2.6 The ODE System (Core Model)

This is the heart of the model — the differential equations:

```julia
function deb_earthworm!(du, u, params, t)
    E, L, Eh, R, OM_soil, OM_horse = u
    p, d = params
    # ... calculations ...
    du[1] = dE
    du[2] = dL
    du[3] = dEh
    du[4] = dR
    du[5] = dOM_soil
    du[6] = dOM_horse
end
```

**What it does:**
Defines the system of ordinary differential equations (ODEs) that describe how the state variables change over time.

**Important Julia conventions:**
- `!` at the end of function name means it **modifies its arguments in place** (mutates)
- `du` is the output vector of derivatives — we fill it in rather than return it
- This is for performance (avoids allocating new memory each timestep)

**The function signature:**
- `du` — Output: derivatives (dE/dt, dL/dt, etc.)
- `u` — Input: current state [E, L, Eh, R, OM_soil, OM_horse]
- `params` — Parameters tuple (p, d)
- `t` — Current time

**R equivalent using deSolve:**
```r
deb_earthworm <- function(t, u, params) {
  # Unpack state variables
  E <- u[1]; L <- u[2]; Eh <- u[3]
  R <- u[4]; OM_soil <- u[5]; OM_horse <- u[6]
  
  # Unpack parameters
  p <- params$p; d <- params$d
  
  # ... calculations ...
  
  # Return list of derivatives (deSolve convention)
  return(list(c(dE, dL, dEh, dR, dOM_soil, dOM_horse)))
}
```

**Understanding the state equations:**

```julia
# Reserve dynamics: dE/dt = assimilation - mobilization
tmp_E = (pAm_t / L) * (freal - E / d.Em)
if E < 1e-12
    dE = -E        # Prevent negative reserves
else
    dE = tmp_E
end
```

This says: reserve changes based on feeding (freal) minus what's being used. The `if` prevents numerical issues with negative values.

```julia
# Structural growth: dL/dt
dL = (v_t / (3 * ((E / d.Em) + g))) * ((E / d.Em) - (L / Lm))
```

Structure grows when reserve density exceeds what's needed for maintenance.

```julia
# Maturity: increases until puberty
if Eh < d.Ehp
    dEh = (1 - kapreal) * pC - kJ_t * Eh
else
    dEh = 0.0
end
```

Energy not used for soma goes to maturity (until puberty) or reproduction (after puberty).

### 2.7 Initialization

```julia
function initialize_earthworm(Winit, init_status, p::DEBEarthwormParams, d::DEBDerivedParams;
                               OM_soil_init=10.0, OM_horse_init=5.0)
    E0 = d.Em
    L0 = (Winit / (1 + p.w))^(1/3)
    
    if init_status == 3
        Eh0 = d.Ehp    # Adult
    else
        Eh0 = d.Ehb    # Juvenile
    end
    
    R0 = 1e-6
    return [E0, L0, Eh0, R0, OM_soil_init, OM_horse_init]
end
```

**What it does:**
Sets up initial conditions for the ODE solver. Converts initial weight to structural length using the DEB weight equation.

**Julia syntax notes:**
- `; OM_soil_init=10.0` — The semicolon separates positional arguments from keyword arguments
- This is like R's default arguments, but explicitly marks them as "keyword-only"

**R equivalent:**
```r
initialize_earthworm <- function(Winit, init_status, p, d,
                                  OM_soil_init = 10.0, OM_horse_init = 5.0) {
  E0 <- d$Em
  L0 <- (Winit / (1 + p$w))^(1/3)
  
  if (init_status == 3) {
    Eh0 <- d$Ehp
  } else {
    Eh0 <- d$Ehb
  }
  
  R0 <- 1e-6
  return(c(E0, L0, Eh0, R0, OM_soil_init, OM_horse_init))
}
```

### 2.8 Feeding Event Callbacks

```julia
Base.@kwdef struct FeedingSchedule
    horse_interval::Float64 = 14.0
    horse_amount::Float64 = 20.0
    horse_start::Float64 = 14.0
    soil_interval::Float64 = 28.0
    soil_amount::Float64 = 100.0
    soil_start::Float64 = 28.0
    verbose::Bool = true
end
```

**What it does:**
Defines the feeding regime — when and how much food is added.

```julia
function create_feeding_callbacks(tspan, schedule::FeedingSchedule)
    t_start, t_end = tspan
    
    # Generate event times
    horse_times = collect(schedule.horse_start : schedule.horse_interval : t_end)
    soil_times = collect(schedule.soil_start : schedule.soil_interval : t_end)
    
    # Callback for adding horse manure
    function add_horse_manure!(integrator)
        integrator.u[6] += schedule.horse_amount
        if schedule.verbose
            @info "Day $(round(integrator.t, digits=1)): Added $(schedule.horse_amount)g horse manure"
        end
    end
    horse_callback = DE.PresetTimeCallback(horse_times, add_horse_manure!)
    
    # ... similar for soil ...
    
    return DE.CallbackSet(horse_callback, soil_callback)
end
```

**What it does:**
Creates "callbacks" — functions that trigger at specific times during the ODE solution to modify the state (add food).

**Key concepts:**
- `integrator` — The ODE solver's internal state object
- `integrator.u[6]` — Access and modify state variable 6 (OM_horse)
- `integrator.t` — Current time
- `PresetTimeCallback` — Trigger at predetermined times

**R equivalent (using deSolve events):**
```r
# In deSolve, you'd use the 'events' argument
eventfun <- function(t, y, parms) {
  y[6] <- y[6] + 20  # Add horse manure
  return(y)
}

# Event times
event_times <- seq(14, 365, by = 14)

# In ode() call:
ode(y0, times, func, parms, 
    events = list(func = eventfun, time = event_times))
```

### 2.9 Solving the ODE

```julia
prob = DE.ODEProblem(deb_earthworm!, u0, tspan, (params, derived))
sol = DE.solve(prob, DE.Tsit5(), callback=callbacks, saveat=1.0)
```

**What it does:**
1. `ODEProblem` — Creates the problem definition (ODE function, initial conditions, time span, parameters)
2. `solve` — Solves it using the Tsit5 algorithm (a good default for non-stiff ODEs)

**Arguments:**
- `deb_earthworm!` — The ODE function
- `u0` — Initial conditions
- `tspan` — Time span as `(t_start, t_end)`
- `(params, derived)` — Parameters passed to ODE function
- `Tsit5()` — Solver algorithm (similar to `rk45` or `dopri5`)
- `callback=callbacks` — Event handling
- `saveat=1.0` — Save solution every 1 day

**R equivalent:**
```r
library(deSolve)

sol <- ode(
  y = u0,
  times = seq(0, 365, by = 1),
  func = deb_earthworm,
  parms = list(p = params, d = derived),
  method = "ode45",
  events = list(func = eventfun, time = event_times)
)
```

### 2.10 Output Calculations

```julia
function calc_outputs(sol, p::DEBEarthwormParams, d::DEBDerivedParams)
    E = sol[1, :]       # Extract first state variable across all times
    L = sol[2, :]
    # ...
    
    Size = L ./ p.Shape  # Physical length = structural length / shape coefficient
    Weight = L.^3 .* (1 .+ E ./ d.Em .* p.w)  # DEB weight equation
    
    return (
        t = sol.t,
        Energy = E,
        Size = Size,
        Weight = Weight,
        # ...
    )
end
```

**What it does:**
Extracts solution components and calculates observable outputs (weight, length).

**Julia syntax notes:**
- `sol[1, :]` — First row, all columns (like R matrix indexing)
- `L ./ p.Shape` — Element-wise division (the `.` broadcasts)
- `return (a = x, b = y)` — Returns a NamedTuple (like R's named list)

**R equivalent:**
```r
calc_outputs <- function(sol, p, d) {
  E <- sol[, 2]  # deSolve puts time in column 1
  L <- sol[, 3]
  
  Size <- L / p$Shape
  Weight <- L^3 * (1 + E / d$Em * p$w)
  
  return(list(
    t = sol[, 1],
    Energy = E,
    Size = Size,
    Weight = Weight
  ))
}
```

---

## Part 3: Understanding `DEB_Acali_MC_Sims.jl`

This file implements Monte Carlo sampling from prior distributions — the first step toward Bayesian inference.

### 3.1 Loading Dependencies

```julia
include("DEB_Acali_wfood.jl")

using Distributions
using Random
using DataFrames
using Statistics
using ProgressMeter
```

**What it does:**
- `include(...)` — Executes another Julia file (like `source()` in R)
- `Distributions` — Probability distributions (like R's built-in `dnorm`, `rnorm`, etc.)
- `Random` — Random number generation
- `DataFrames` — Tabular data (like R's data.frame or tibble)
- `ProgressMeter` — Progress bars for loops

**R equivalent:**
```r
source("DEB_Acali_wfood.R")
library(truncnorm)  # For truncated distributions
library(dplyr)      # Data manipulation
```

### 3.2 Prior Distributions

```julia
Base.@kwdef struct PriorDistributions
    pAm::Distribution = truncated(Normal(1172.0, 117.2), 800.0, 8000.0)
    r_pAm_pM::Distribution = truncated(Normal(1.5, 0.2), 0.01, 100.0)
    kap::Distribution = Uniform(0.05, 0.25)
    Ehb::Distribution = truncated(Normal(0.5, 0.5), 0.1, 10.0)
    Ehp::Distribution = truncated(Normal(100.0, 50.0), 1.0, 1000.0)
    E_coc::Distribution = truncated(Normal(200.0, 100.0), 50.0, 5000.0)
    rOM_ClxHorse::Distribution = truncated(Normal(0.2, 0.1), 0.0, 1.0)
    Sigma_W::Distribution = truncated(Normal(5.0, 1.0), 0.01, 100.0)
end
```

**What it does:**
Defines prior distributions matching your MCSim specification.

**Julia syntax notes:**
- `::Distribution` — Type annotation saying this field holds a distribution object
- `truncated(Normal(μ, σ), lower, upper)` — Creates a truncated normal distribution
- `Uniform(a, b)` — Uniform distribution on [a, b]

**Comparison to your MCSim priors:**

| MCSim | Julia |
|-------|-------|
| `TruncNormal(1172, 117.2, 800, 8000)` | `truncated(Normal(1172.0, 117.2), 800.0, 8000.0)` |
| `Uniform(0.05, 0.25)` | `Uniform(0.05, 0.25)` |

**R equivalent:**
```r
# Using a list to store distribution parameters
priors <- list(
  pAm = list(dist = "truncnorm", mean = 1172, sd = 117.2, a = 800, b = 8000),
  kap = list(dist = "uniform", min = 0.05, max = 0.25)
)

# Sampling in R:
library(truncnorm)
pAm_sample <- rtruncnorm(1, mean = 1172, sd = 117.2, a = 800, b = 8000)
kap_sample <- runif(1, 0.05, 0.25)
```

**brms comparison:**
In brms, you'd set priors like:
```r
prior(normal(1172, 117.2), class = "b", coef = "pAm", lb = 800, ub = 8000)
prior(uniform(0.05, 0.25), class = "b", coef = "kap")
```

### 3.3 Sampling from Priors

```julia
function sample_from_priors(priors::PriorDistributions, base_params::DEBEarthwormParams)
    # Sample from each prior
    pAm = rand(priors.pAm)
    r_pAm_pM = rand(priors.r_pAm_pM)
    kap = rand(priors.kap)
    # ...
    
    # Calculate derived parameters
    pM = pAm * r_pAm_pM
    Em = pAm / base_params.v
    
    # Back-calculate ratios for model compatibility
    alpha_Ehb = Ehb / E_coc
    beta_Ehp = Ehp / Ehb
    
    # Create new parameter set
    new_params = DEBEarthwormParams(
        pAm = pAm,
        kap = kap,
        # ...
    )
    
    return new_params, sampled_values
end
```

**What it does:**
1. Draws random samples from each prior distribution
2. Calculates derived parameters
3. Creates a new parameter struct for simulation

**Key function:**
- `rand(distribution)` — Draw one random sample from the distribution

**R equivalent:**
```r
sample_from_priors <- function(priors, base_params) {
  pAm <- rtruncnorm(1, mean = 1172, sd = 117.2, a = 800, b = 8000)
  kap <- runif(1, 0.05, 0.25)
  Ehb <- rtruncnorm(1, mean = 0.5, sd = 0.5, a = 0.1, b = 10)
  # ...
  
  return(list(pAm = pAm, kap = kap, Ehb = Ehb, ...))
}
```

### 3.4 The Monte Carlo Loop

```julia
function run_monte_carlo(n_samples::Int, ...)
    # Storage
    sampled_params = Vector{SampledParameters}(undef, n_samples)
    Weight_final = Vector{Union{Float64, Missing}}(undef, n_samples)
    
    n_success = 0
    
    # Run simulations with progress bar
    @showprogress desc="Monte Carlo: " enabled=show_progress for i in 1:n_samples
        # Sample parameters
        params_i, sampled_i = sample_from_priors(priors, base_params)
        sampled_params[i] = sampled_i
        
        # Run simulation
        result = run_single_simulation(params_i, ...)
        
        if isnothing(result)
            Weight_final[i] = missing
        else
            n_success += 1
            Weight_final[i] = result.Weight[end]
        end
    end
    
    # Create summary DataFrame
    summary_df = DataFrame(
        sample = 1:n_samples,
        Weight_final = Weight_final,
        pAm = [p.pAm for p in sampled_params],
        # ...
    )
    
    return MCResult(n_samples, n_success, sampled_params, trajectories, summary_df)
end
```

**What it does:**
1. Pre-allocates storage for results
2. Loops through `n_samples` iterations
3. Each iteration: sample parameters → run ODE → store results
4. Compiles everything into a DataFrame

**Julia syntax notes:**
- `Vector{Float64}(undef, n)` — Pre-allocate vector of n Float64s (uninitialized)
- `Union{Float64, Missing}` — Can hold either a Float64 or `missing` (like R's NA)
- `@showprogress for i in 1:n` — Macro that adds a progress bar to the loop
- `[p.pAm for p in sampled_params]` — List comprehension (like `sapply(sampled_params, function(p) p$pAm)` in R)

**R equivalent:**
```r
run_monte_carlo <- function(n_samples, base_params, priors, ...) {
  # Pre-allocate
  results <- data.frame(
    sample = 1:n_samples,
    Weight_final = NA_real_,
    pAm = NA_real_
  )
  
  pb <- txtProgressBar(max = n_samples, style = 3)
  
  for (i in 1:n_samples) {
    setTxtProgressBar(pb, i)
    
    # Sample parameters
    params_i <- sample_from_priors(priors, base_params)
    results$pAm[i] <- params_i$pAm
    
    # Run simulation
    tryCatch({
      sol <- run_single_simulation(params_i, ...)
      results$Weight_final[i] <- tail(sol$Weight, 1)
    }, error = function(e) {
      results$Weight_final[i] <- NA
    })
  }
  
  close(pb)
  return(results)
}
```

### 3.5 Error Handling

```julia
try
    sol = DE.solve(prob, DE.Tsit5(), ...)
    if sol.retcode != :Success
        return nothing
    end
    # ... process solution ...
catch e
    return nothing
end
```

**What it does:**
- `try ... catch` — Exception handling (like `tryCatch` in R)
- `sol.retcode != :Success` — Check if ODE solver succeeded
- `:Success` — A Symbol (Julia's version of R's "symbol" or factor level)
- `return nothing` — Return Julia's null value to indicate failure

**R equivalent:**
```r
tryCatch({
  sol <- ode(...)
  # Process solution
}, error = function(e) {
  return(NULL)
})
```

### 3.6 DataFrame Creation

```julia
summary_df = DataFrame(
    sample = 1:n_samples,
    Weight_final = Weight_final,
    pAm = [p.pAm for p in sampled_params],
    kap = [p.kap for p in sampled_params]
)
```

**What it does:**
Creates a tabular data structure, just like R's `data.frame`.

**R equivalent:**
```r
summary_df <- data.frame(
  sample = 1:n_samples,
  Weight_final = Weight_final,
  pAm = sapply(sampled_params, function(p) p$pAm),
  kap = sapply(sampled_params, function(p) p$kap)
)
```

### 3.7 Statistical Summaries

```julia
vals = collect(skipmissing(df.Weight_final))
mean(vals)
std(vals)
median(vals)
quantile(vals, 0.025)
quantile(vals, 0.975)
```

**What it does:**
- `skipmissing(x)` — Iterator that skips `missing` values (like `na.rm = TRUE` in R)
- `collect(...)` — Materializes iterator into a vector
- `quantile(x, p)` — Same as R's `quantile(x, p)`

**R equivalent:**
```r
vals <- na.omit(df$Weight_final)
mean(vals)
sd(vals)
median(vals)
quantile(vals, 0.025)
quantile(vals, 0.975)
```

### 3.8 Visualization

```julia
p1 = histogram(collect(skipmissing(df.Weight_final)) .* 1000, 
               xlabel="Final Weight (mg)", ylabel="Count",
               label=false, color=:brown, alpha=0.7, bins=30)
```

**What it does:**
Creates a histogram using the Plots.jl package.

**Julia syntax notes:**
- `:brown` — A Symbol representing color (like `"brown"` string but more efficient)
- `label=false` — Don't show legend label
- `alpha=0.7` — Transparency

**R equivalent (ggplot2):**
```r
ggplot(df, aes(x = Weight_final * 1000)) +
  geom_histogram(fill = "brown", alpha = 0.7, bins = 30) +
  labs(x = "Final Weight (mg)", y = "Count")
```

---

## Part 4: Conceptual Comparison to brms/Stan

### 4.1 What This Code Does vs. What brms Does

| Aspect | This Code | brms/Stan |
|--------|-----------|-----------|
| **Purpose** | Forward simulation (prior predictive) | Bayesian inference (posterior) |
| **Data** | Not used (generates predictions) | Used to update priors |
| **Output** | Prior predictive distribution | Posterior distribution |
| **Algorithm** | Direct sampling + ODE solving | MCMC (HMC/NUTS) |

### 4.2 The Bayesian Workflow

```
1. PRIOR PREDICTIVE (This code!)
   Sample θ ~ Prior(θ)  →  Simulate y ~ Model(θ)  →  Check predictions make sense
   
2. POSTERIOR INFERENCE (Next step: Turing.jl)
   Observe data y_obs  →  Compute P(θ | y_obs) via MCMC  →  Get posterior samples
   
3. POSTERIOR PREDICTIVE
   Sample θ ~ Posterior(θ)  →  Simulate y_new ~ Model(θ)  →  Compare to y_obs
```

The current Monte Carlo code implements **Step 1** — checking that your priors generate reasonable predictions before fitting to data.

### 4.3 Likelihood Specification (Preview of Turing.jl)

Your MCSim file specified:
```
Likelihood(Weight, Normal, Prediction(Weight), Sigma_W)
Likelihood(Reproduction, Poisson, Prediction(Reproduction))
```

In Turing.jl (which we'll implement next), this becomes:

```julia
using Turing

@model function deb_model(observed_weight, observed_reproduction)
    # Priors
    pAm ~ truncated(Normal(1172, 117.2), 800, 8000)
    kap ~ Uniform(0.05, 0.25)
    Sigma_W ~ truncated(Normal(5, 1), 0.01, 100)
    # ...
    
    # Run ODE model
    predictions = run_deb_simulation(pAm, kap, ...)
    
    # Likelihood
    for i in 1:length(observed_weight)
        observed_weight[i] ~ Normal(predictions.Weight[i], Sigma_W)
    end
    
    for i in 1:length(observed_reproduction)
        observed_reproduction[i] ~ Poisson(predictions.Reproduction[i])
    end
end
```

**brms comparison:**
```r
brm(
  bf(Weight ~ deb_prediction, sigma ~ 1) +
  bf(Reproduction ~ deb_prediction, family = poisson()),
  prior = c(
    prior(normal(1172, 117.2), class = "b", coef = "pAm", lb = 800, ub = 8000),
    prior(uniform(0.05, 0.25), class = "b", coef = "kap")
  )
)
```

The key difference is that brms can't easily handle ODE models, while Turing.jl integrates seamlessly with DifferentialEquations.jl.

---

## Part 5: Quick Reference

### 5.1 Common Operations Cheat Sheet

| Task | Julia | R |
|------|-------|---|
| Create vector | `[1, 2, 3]` | `c(1, 2, 3)` |
| Vector of zeros | `zeros(10)` | `rep(0, 10)` |
| Sequence | `1:10` or `collect(1:10)` | `1:10` |
| Element-wise ops | `x .+ y`, `x .* y` | `x + y`, `x * y` |
| Apply function | `map(f, x)` or `f.(x)` | `sapply(x, f)` |
| Filter | `filter(f, x)` | `Filter(f, x)` |
| Sample from dist | `rand(dist)` | `rdist(1, ...)` |
| PDF | `pdf(dist, x)` | `ddist(x, ...)` |
| CDF | `cdf(dist, x)` | `pdist(x, ...)` |
| Set seed | `Random.seed!(42)` | `set.seed(42)` |
| Read CSV | `CSV.read("f.csv", DataFrame)` | `read.csv("f.csv")` |
| Write CSV | `CSV.write("f.csv", df)` | `write.csv(df, "f.csv")` |

### 5.2 Distribution Comparison

| Distribution | Julia (Distributions.jl) | R |
|--------------|--------------------------|---|
| Normal | `Normal(μ, σ)` | `rnorm(n, μ, σ)` |
| Uniform | `Uniform(a, b)` | `runif(n, a, b)` |
| Truncated Normal | `truncated(Normal(μ, σ), a, b)` | `truncnorm::rtruncnorm(n, a, b, μ, σ)` |
| Beta | `Beta(α, β)` | `rbeta(n, α, β)` |
| Gamma | `Gamma(α, θ)` | `rgamma(n, α, scale=θ)` |
| Poisson | `Poisson(λ)` | `rpois(n, λ)` |
| LogNormal | `LogNormal(μ, σ)` | `rlnorm(n, μ, σ)` |

### 5.3 Running the Code

```julia
# Start Julia REPL
# Navigate to your directory

# First time: Install packages
using Pkg
Pkg.add(["DifferentialEquations", "Plots", "Distributions", 
         "DataFrames", "ProgressMeter", "Statistics"])

# Run the model
include("DEB_Acali_wfood.jl")

# Run Monte Carlo
include("DEB_Acali_MC_Sims.jl")
```

---

## Summary

You now have a complete DEB model implementation in Julia that:

1. **Defines parameters** using typed structs with defaults
2. **Implements the ODE system** following DEB theory
3. **Handles discrete events** (feeding) via callbacks
4. **Runs Monte Carlo sampling** from specified priors
5. **Produces diagnostic plots** of prior predictive distributions

The next step would be implementing Bayesian inference with Turing.jl to fit the model to observed data — similar to what you'd do with brms or Stan, but with full support for ODE-based models.