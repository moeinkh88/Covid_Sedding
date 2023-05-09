using Statistics
using CSV, DataFrames
using Interpolations,LinearAlgebra
using Optim, FdeSolver
using SpecialFunctions, StatsBase, Random, DifferentialEquations
using Plots, StatsPlots

# Dataset
dataset_CC = CSV.read("Covid_Shedding/time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",64:164] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

dataset_R = CSV.read("Covid_Shedding/time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",64:164]
TrueR=diff(Float64.(Vector(RData[1,:])))

dataset_D = CSV.read("Covid_Shedding/time_series_covid19_deaths_global.csv", DataFrame) # all data of Recover
DData=dataset_D[dataset_D[!,2].=="South Africa",64:164]
TrueD=(Float64.(Vector(DData[1,:])))
#initial conditons and parameters


τ0=25

function β(t)
    if  0 < t < τ0
        return β0
    else
        return β1+ (β0-β1)/(1+ω*(t-τ0))
    end
end

# β(t)=β0

β0=0.492
β1=0.166
ω=.005
η1=.75
η2=.5
η3=2e-6
γ1=.85
γ2=.2
σ=1
r=.6
τ1,τ2, τ3=[1/6, 1/10, 1/14]
δ1=.035
δ2=.018
 ζ1,ζ2, ζ3=[0.002, 0.002, 0.001]
ν=.85

S0,E0,A0,I0,J0,R0,P0=[5.9e7, 0,0, 65,0,0,0]

N0=S0+E0+A0+I0+J0+R0
X0=[S0,E0,A0,I0,J0,R0,P0,N0,0]

par=σ, r, γ1, τ1,γ2, τ2,δ1,τ3, δ2, ζ1,ζ2, ζ3, ν

function  Ff(t, x, par)


    σ, r, γ1, τ1,γ2, τ2,δ1,τ3, δ2, ζ1,ζ2, ζ3, ν = par
    S,E,A,I,J,R,P,N,D = x
    λ=(β(t)*(η1*A + I + η2*J))/N + β(t)*η3*P


    dS= -λ*S
    dE= λ*S - σ*E
    dA= r* σ*E - (γ1 + τ1)*A
    dI= (1-r)*σ*E - (γ2+ τ2 +δ1)*I
    dJ= γ1*A + γ2*I - (τ3+δ2)*J
    dR= τ1*A + τ2*I + τ3*J
    dP= ζ1*A + ζ2*I + ζ3*J - ν*P
    dN= -δ1*I-δ2*J
    dD= δ1*I+δ2*J

    return [dS,dE,dA,dI,dJ,dR,dP,dN,dD]

end


_, x = FDEsolver(Ff, [1,length(C)], X0, ones(9), par, h=.05, nc=4)

scatter(TrueD, legend=false)
plot!((x[1:20:20*23,9]))

scatter(C)
plot!(sum(x[1:20:end,[5]],dims=2))

plot(x[1:20:end,5])


K1=γ1 + τ1
K2=γ2 + τ2 + δ1
K3=τ3 + δ2

Rhh=(β1*(r*K2*(K3*σ + γ1)+(1-r)*K1*(K3*η2+γ2)))/(K1*K2*K3)
Renv=(β1*η3*S0*(r*K2*(K3*ζ1 + γ1*ζ3)+(1-r)*K1*(K3*ζ2+γ2*ζ3)))/(K1*K2*K3*ν)

Rhh+Renv
