# initial fit (unbounded) with birth rate and new R, fit CRF
# different mortality of I_A and I_S
# consider tested I_A, proportion of considering I_A in data
# transission I_A to I_S

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

IS0=17;R0=0;RT0=0;P0=100;D0=0;DT0=0;
Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate
tSpan=(1,length(C))

# Define the equation

function  F(dx, x, par, t)

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σS,σA,γS,γA,ηS,ηA,ξ,T=par
    S,E,IA,IS,R,RT,P,D,DT,N=x

    dx[1]= Λ*N - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dx[2]= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - (ψ + μ + ω)*E
    dx[3]= (1-δ)*ω*E - (μ+σA)*IA - γA*IA - ξ*IA
    dx[4]= δ*ω*E - (μ + σS)*IS - γS*IS + ξ*IA
    dx[5]=γS*IS + γA*IA - μ*R
    dx[6]=γS*IS + T*γA*IA - μ*RT
    dx[7]=ηA*IA + ηS*IS - μp*P
    dx[8]=σA*IA + σS*IS - μ*D
	dx[9]=T*σA*IA + σS*IS - μ*DT
    dx[10]=Λ*N - σA*IA - σS*IS - μ*N
    return nothing

end


# Open the file
AA=readlines("Output_CSC/2023_10_18 ODE/complete.txt")

BB=map(x -> parse.(Float64, split(x)), AA)

plot(; legend=false)
Err=zeros(length(BB))
for ii in 1:length(BB)
	μ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σS,σA,γS,γA,ηS,ηA,ξ,T = BB[ii][2:19]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σS,σA,γS,γA,ηS,ηA,ξ,T]
	S0=BB[ii][1]
	E0=BB[ii][20]
	IA0=BB[ii][21]
	P0=BB[ii][22]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0,DT0,N0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	II=sol[4,:] .+ T.*sol[3,:]
	RR=sol[6,:]
	DD=sol[9,:]
	plot!(II; alpha=0.1, color="#BBBBBB")
	Err[ii]=rmsd([C TrueR TrueD], [II RR DD])
end

# Plot real
scatter!(C)

##
plot(; legend=false)
Err=zeros(length(BB))
for ii in 1:length(BB)
	μ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σS,σA,γS,γA,ηS,ηA,ξ,T = BB[ii][2:19]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σS,σA,γS,γA,ηS,ηA,ξ,T]
	S0=BB[ii][1]
	E0=BB[ii][20]
	IA0=BB[ii][21]
	P0=BB[ii][22]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0,DT0,N0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	II=sol[4,:] .+ T.*sol[3,:]
	RR=sol[6,:]
	DD=sol[9,:]
	plot!(RR; alpha=0.1, color="#BBBBBB")
end

# Plot real
scatter!(TrueR)

plot(; legend=false)
Err=zeros(length(BB))
for ii in 1:length(BB)
	μ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σS,σA,γS,γA,ηS,ηA,ξ,T = BB[ii][2:19]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σS,σA,γS,γA,ηS,ηA,ξ,T]
	S0=BB[ii][1]
	E0=BB[ii][20]
	IA0=BB[ii][21]
	P0=BB[ii][22]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0,DT0,N0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	II=sol[4,:] .+ T.*sol[3,:]
	RR=sol[6,:]
	DD=sol[9,:]
	plot!(DD; alpha=0.1, color="#BBBBBB")
end

# Plot real
scatter!(TrueD)

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
