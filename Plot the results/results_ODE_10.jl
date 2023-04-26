

using Statistics
using CSV, DataFrames
using Interpolations,LinearAlgebra
using Optim, FdeSolver
using SpecialFunctions, StatsBase, Random, DifferentialEquations

# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",70:250] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",70:250]
TrueR=diff(Float64.(Vector(RData[1,:])))


dataset_D = CSV.read("time_series_covid19_deaths_global.csv", DataFrame) # all data of Recover
DData=dataset_D[dataset_D[!,2].=="South Africa",70:249]
TrueD=(Float64.(Vector(DData[1,:])))
#initial conditons and parameters

E0=0;IA0=100;IS0=17;R0=0;P0=100;D0=0;

tSpan=(1,length(C))

# Define the equation

function  F(dx, x, par, t)

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=par
    S,E,IA,IS,R,P,D,N=x

    dx[1]= Λ - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dx[2]= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dx[3]= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dx[4]= δ*ω*E - (μ + σ)*IS - γS*IS
    dx[5]=γS*IS + γA*IA - μ*R
    dx[6]=ηA*IA + ηS*IS - μp*P
    dx[7]=σ*(IA+IS)
    dx[8]=Λ - σ*(IA+IS) - μ*N
    return nothing

end


  pp=[1.702103879720292e6
	  0.03153364356746533
	 13.853882423490122
	  0.0009844701621384458
	  1.1371874387714e-7
	  0.978334818708447
	  1.0465921676230195e-6
	  1.3397253658684369e-9
	  0.5482753791316302
	  0.006921501734044292
	  0.11093874661602011
	  0.025339327326045272
	  0.0018748129801086598
	  0.11691211740148623
	  1.6181747789292504e-7
	  0.9206392003870149
	  9.91573754393608e-5
	  0.0004317721661172488]
	 μ ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = pp[2:16]
 	p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=pp[1]
	IA0=pp[17]
 	P0=pp[18]
 	N0=S0+E0+IA0+IS0+R0
 	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]

prob = ODEProblem(F, X0, tSpan, p)
sol = solve(prob, alg_hints=[:stiff]; p=p, saveat=1)
# plot(reduce(vcat,sol.u')[:,8])
Pred=reduce(vcat,sol.u')[:,[4,5,7]]
rmsd([C TrueR TrueD], Pred)


using Plots
plot(reduce(vcat,sol.u')[:,4])
scatter!(C)

plot(reduce(vcat,sol.u')[:,5])
scatter!(TrueR)

plot(reduce(vcat,sol.u')[:,7])
scatter!(TrueD)

#population
plot(reduce(vcat,sol.u')[:,8])
