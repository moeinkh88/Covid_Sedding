## initial fit (unbounded) with new R and D, fit CRF
# same mortality of I_A and I_S
# consider tested I_A, proportion of considering I_A in data
# later we can consider scaled paramters

using Statistics
using CSV, DataFrames
using Interpolations,LinearAlgebra
using Optim, FdeSolver
using SpecialFunctions, StatsBase, Random, DifferentialEquations, Turing

# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",70:250] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",70:250]
TrueR=diff(Float64.(Vector(RData[1,:])))

dataset_D = CSV.read("time_series_covid19_deaths_global.csv", DataFrame) # all data of Recover
DData=dataset_D[dataset_D[!,2].=="South Africa",70:250]
TrueD=diff(Float64.(Vector(DData[1,:])))

#initial conditons and parameters
#
# i=parse(Int32,ARGS[1])
# X0par=CSV.read("par400X0.csv", DataFrame, header=1)
# IA=X0par[i,1]
# P=X0par[i,2]
S0=665188;E0=0;IA0=100;IS0=17;R0=0;RT0=0;P0=100;D0=0;DT0=0;N0=S0+E0+IA0+IS0+R0+D0
x0=[S0,E0,IA0,IS0,R0,RT0,P0,D0,DT0] # initial conditons S0,E0,IA0,IS0,R0,P0,D0

Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate
μp=0.172499999999 # natural death rate of pathogens virusess
ϕ1=2e-6 # proportion of interaction with an infectious environment
ϕ2=0.3 #proportion of interaction with an infectious I
β1=0.00414 # infection rate from S to E due to contact with P
β2=0.0115 # infection rate from S to E due to contact with IA and/or IS
δ=0.7 #proportion of symptomatic infectious people
ψ=0.0051  #progression rate from E back to S due to robust immune system
ω=0.5 #progression rate from E to either IA or IS
σ2=0.025 #disease induced death rate from IS or IA
γS=0.1 #rate of recovery of the symptomatic individuals
γA=1/6 #rate of recovery of the asymptomatic individuals
ηS=0.002 #rate of virus spread to environment by IS
ηA=0.002 #rate of virus spread to environment by IA
ξ=0.002 #rate of transission from IA to IS
T=.1 # proportion of IA cases that did the covid test


par=[Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]

Ndays=length(C)
tSpan=(1,Ndays)
# Define the equation


function  F(dx, x, par, t)

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T=par
    S,E,IA,IS,R,R1,P,D,D1=x

    dx[1]= Λ - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dx[2]= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dx[3]= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dx[4]= δ*ω*E - (μ + σ)*IS - γS*IS
    dx[5]=γS*IS + γA*IA - μ*R
    dx[6]=γS*IS + T*γA*IA - μ*R1
    dx[7]=ηA*IA + ηS*IS - μp*P
    dx[8]=σ*(IA+IS) - μ*D
    dx[9]=σ*(T*IA+IS) - μ*D1
    return nothing

end


prob = ODEProblem(F, x0, tSpan, par)

#optimization
ϵ=1e-7

@model function fitprob(data,prob)
    # Prior distributions.

    σ ~ InverseGamma(2, 3)
	S0 ~ truncated(Normal(500,100000); lower=1000, upper=1500000)
	μp ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ϕ1 ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ϕ2 ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	β1 ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	β2 ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	δ ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ψ ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ω ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	σ2 ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	γS ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	γA ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ηS ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ηA ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	T ~ truncated(Normal(0, 1); lower=ϵ, upper=.8)
	E0 ~ truncated(Normal(0,300); lower=0, upper=300)
	IA0 ~ truncated(Normal(0,300); lower=0, upper=300)
	P0 ~ truncated(Normal(0,300); lower=0, upper=300)

    # Simulate model.
	p=[Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]

	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0,DT0]
	prob = remake(prob; p = p, u0 = X0)
    x = solve(prob,alg_hints=[:stiff], abstol = 1e-12, reltol = 1e-12; saveat=1)
	II=x[4,:] .+ T.*x[3,:]
	RR=x[6,:]
	DD=x[9,:]


	pred=[II RR DD]
	# Observations.
    for i in 1:length(pred[1,:])
        data[:,i] ~ MvNormal(pred[:,i], σ^2 * I)
    end

    return nothing
end

model = fitprob([C TrueR TrueD],prob)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
Nch=3
chain = sample(model, NUTS(0.65), MCMCSerial(), Nch, 4; progress=false)


display(chain)
##
posterior_samples = sample(chain[[:S0,:μp,:ϕ1,:ϕ2,:β1,:β2,:δ,:ψ,:ω,:σ2,:γS,:γA,:ηS,:ηA,:T, :E0, :IA0, :P0]], Nch; replace=false)

function myshowall(io, x, limit = false)
  println(io, summary(x), ":")
  Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, Array(posterior_samples.value[:,:,1]), false)


Err=zeros(Nch)
for i in 1:Nch
	pp=Array(posterior_samples.value[:,:,1])[i,:]
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = pp[2:15]
	p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=pp[1]
	E0=pp[16]
	IA0=pp[17]
	P0=pp[18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0,DT0]

	prob = ODEProblem(F, X0, tSpan, p)
	sol = solve(prob,alg_hints=[:stiff], abstol = 1e-12, reltol = 1e-12; saveat=1)
	II=sol[4,:] .+ T.*sol[3,:]
	RR=sol[6,:]
	DD=sol[9,:]
	Err[i]=rmsd([C TrueR TrueD], [II RR DD])

end

valErr,indErr=findmin(Err)

display(["MinErr",valErr])

myshowall(stdout, Array(posterior_samples.value[:,:,1])[indErr,:], false)
