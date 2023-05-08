

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
DData=dataset_D[dataset_D[!,2].=="South Africa",70:250]
TrueD=diff(Float64.(Vector(DData[1,:])))
#initial conditons and parameters

E0=0;IA0=100;IS0=17;R0=0;R10=0;P0=100;D0=0;

tSpan=(1,length(C))

# Define the equation

function  F(dx, x, par, t)

	Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=par
    S,E,IA,IS,R,R1,P,D,N=x

    dx[1]= Λ*N - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dx[2]= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dx[3]= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dx[4]= δ*ω*E - (μ + σ)*IS - γS*IS
    dx[5]=γS*IS + γA*IA - μ*R
    dx[6]=γS*IS - μ*R1
    dx[7]=ηA*IA + ηS*IS - μp*P
    dx[8]=σ*(IA+IS) - μ*D
    dx[9]=Λ*N - σ*(IA+IS) - μ*N
    return nothing

end

function  Ff(t, x, par)

	Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=par
    S,E,IA,IS,R,R1,P,D,N=x

    dS= Λ*N - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dE= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dIA= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dIS= δ*ω*E - (μ + σ)*IS - γS*IS
    dR=γS*IS + γA*IA - μ*R
	dR1=γS*IS - μ*R1
    dP=ηA*IA + ηS*IS - μp*P
    dD=σ*(IA+IS) - μ*D
    dN=Λ*N - σ*(IA+IS) - μ*N
    return [dS,dE,dIA,dIS,dR,dR1,dP,dD,dN]

end

  pp=[ 260276.0124643658
      0.15710660446485994
      3.2368572156584134e-5
      0.9806327153319467
      6.604793841780076e-5
      9.617697564292186e-5
      0.9988101372380245
      0.001655255062422364
      0.07638883095519367
      0.9999911321929686
      0.025687039235518315
      0.9500307616083895
      0.017925463276958195
      0.3452665875137063
    292.3228126168938
     22.66704909995665
	 13.239275804383805]
	Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
	μ=9.468e-3 # natural human death rate
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = pp[2:14]
 	p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=pp[1]
	E0=pp[15]
	IA0=pp[16]
 	P0=pp[17]
 	N0=S0+E0+IA0+IS0+R0
 	X0=[S0,E0,IA0,IS0,R0,R10,P0,D0,N0]
p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]

prob = ODEProblem(F, X0, tSpan, p)
sol = solve(prob, alg_hints=[:stiff]; p=p, saveat=1)
# plot(reduce(vcat,sol.u')[:,8])
Pred=reduce(vcat,sol.u')[:,[4,6]]
rmsd([C TrueR], Pred)

Order=ones(9)

# args =[0.9999999944414343, 0.9999999994175779, 0.9999999806117581, 0.9999999520246089, 0.7961929904503826, 0.8269792658520363, 0.8684684546888737, 0.9999999889738381]
args=[0.9999999999999997, 0.9999999999999998, 0.5000000000000014, 0.9999999999999988, 0.7500067885054235, 0.9520000468530682, 0.9999999999999997, 0.9999999999999997]
Order[1:7]=args[1:7];
Order[9]=args[8]
# E0, IA0, P0, ϕ2, δ, ω, γA = args[8:14]
# E0, IA0, P0 = args[8:10]
# N0=S0+E0+IA0+IS0+R0
# X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
par1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par1, h=.05, nc=4)
Pred1=x1[1:20:end,[4,6]]
rmsd([C TrueR], Pred1)

using Plots
plot(reduce(vcat,sol.u')[:,4])
plot!(x1[1:20:end,4])
scatter!(C)

plot(reduce(vcat,sol.u')[:,5])
plot!(reduce(vcat,sol.u')[:,6])
plot!(x1[1:20:end,5])
plot!(x1[1:20:end,6])
scatter!(TrueR)

plot(reduce(vcat,sol.u')[:,8])
plot!(x1[1:20:end,8])
scatter!(TrueD)

#population
plot(reduce(vcat,sol.u')[:,9])
plot!(x1[1:20:end,9])

##
# Open the file
AA=readlines("Output_CSC/26huhti ODE/Output_CSC_ODE20.txt")

BB=map(x -> parse.(Float64, split(x)), AA)

plot(; legend=false)
Err=zeros(length(BB))
for ii in 1:length(BB)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[ii][2:14]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=BB[ii][1]
	E0=BB[ii][15]
	IA0=BB[ii][16]
	P0=BB[ii][17]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,R10,P0,D0,N0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	if reduce(vcat,sol.u')[45,6] < 2.5e3
	Pred1=reduce(vcat,sol.u')[:,5]
	plot!(Pred1; alpha=0.1, color="#BBBBBB")
	# Err[ii]=rmsd(C, Pred1)
	Err[ii]=rmsd(TrueR, Pred1)
	end
end

# Plot real
# scatter!(C)
scatter!(TrueR)

##
Err=zeros(length(BB))
plot(; legend=false)
for ii in 1:length(BB)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[ii][2:14]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=BB[ii][1]
	IA0=BB[ii][15]
	P0=BB[ii][16]
	E0=BB[ii][17]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,R10,P0,D0,N0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	# if reduce(vcat,sol.u')[50,5] < 400
	Pred1=reduce(vcat,sol.u')[:,6]
	plot!(Pred1; alpha=0.1, color="#BBBBBB")
	Err[ii]=rmsd(TrueR, Pred1)
	# end
end

# Plot real
scatter!(TrueR)

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


μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[indErr][2:14]
p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
S0=BB[indErr][1]
IA0=BB[indErr][15]
P0=BB[indErr][16]
N0=S0+E0+IA0+IS0+R0
X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
prob = ODEProblem(F, X0, tSpan, p1)
sol = solve(prob, alg_hints=[:stiff]; saveat=1)
plot!(reduce(vcat,sol.u')[:,4])

# rmsd(TrueR, reduce(vcat,sol.u')[:,5])
rmsd(C, reduce(vcat,sol.u')[:,4])
