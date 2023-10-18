# result from turing when pathogens effects are excluded

using Statistics
using CSV, DataFrames
using Interpolations,LinearAlgebra
using Optim, FdeSolver
using SpecialFunctions, StatsBase, Random, DifferentialEquations
using Plots

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

E0=0;IA0=100;IS0=17;R0=0;R10=0;P0=100;D0=0;
Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate
tSpan=(1,length(C))

# Define the equation


function  F(dx, x, par, t)

	Λ,μ,ϕ2,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=par
    S,E,IA,IS,R,D,N=x

	dx[1]= Λ*N - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dx[2]= β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dx[3]= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dx[4]= δ*ω*E - (μ + σ)*IS - γS*IS
    dx[5]=γS*IS + γA*IA - μ*R
    dx[6]=σ*(IA+IS)
    dx[7]=Λ*N - σ*(IA+IS) - μ*N
    return nothing

end


# Open the file
AA=readlines("Output_CSC/2023_10_18 ODE/slurm-19014192.txt")

BB=map(x -> parse.(Float64, split(x)), AA)

plot(; legend=false)
Err=zeros(length(BB))
for ii in 1:length(BB)
	ϕ2,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[ii][2:11]
	p1= [Λ,μ,ϕ2,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=BB[ii][1]
	IA0=BB[ii][12]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,D0,N0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	if reduce(vcat,sol.u')[50,4] < 840
	Pred1=reduce(vcat,sol.u')[:,[4,5]]
	plot!(Pred1[:,1]; alpha=0.1, color="#BBBBBB")
	Err[ii]=rmsd([C TrueR], Pred1)
	end
end

# Plot real
scatter!(C)

##
# plot(; legend=false)
# Err=zeros(length(BB))
# for ii in 1:length(BB)
# 	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[ii][2:14]
# 	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
# 	S0=BB[ii][1]
# 	IA0=BB[ii][15]
# 	P0=BB[ii][16]
# 	N0=S0+E0+IA0+IS0+R0
# 	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
#
# 	prob = ODEProblem(F, X0, tSpan, p1)
# 	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
# 	if reduce(vcat,sol.u')[50,5] < 400
# 	Pred1=reduce(vcat,sol.u')[:,5]
# 	plot!(Pred1; alpha=0.1, color="#BBBBBB")
# 	Err[ii]=rmsd(TrueR, Pred1)
# 	end
# end
#
# # Plot real
# scatter!(TrueR)

#plot the best
Err=replace!(Err, 0=>Inf)
# Err=filter(!iszero, Err)
valErr,indErr=findmin(Err)
display(["MinErr",valErr])
function myshowall(io, x, limit = false)
  println(io, summary(x), ":")
  Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, BB[indErr,:], false)


ϕ2,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[indErr][2:11]
p1= [Λ,μ,ϕ2,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
S0=BB[indErr][1]
IA0=BB[indErr][12]
N0=S0+E0+IA0+IS0+R0
X0=[S0,E0,IA0,IS0,R0,D0,N0]
prob = ODEProblem(F, X0, tSpan, p1)
sol = solve(prob, alg_hints=[:stiff]; saveat=1)
plot!(reduce(vcat,sol.u')[:,4])

# rmsd(TrueR, reduce(vcat,sol.u')[:,5])
rmsd(C, reduce(vcat,sol.u')[:,4])
