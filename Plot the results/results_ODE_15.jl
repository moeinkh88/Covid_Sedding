

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


  pp=[ 692267.3510933297
      0.3166954676936247
      0.5730771571643412
      0.23288682334315053
      0.687760205638112
      0.49285736947320974
      0.1281953909726841
      0.3393015688374467
      0.7510695405377631
      0.49263832600102525
      0.4717190316664661
      0.46107237686820673
      0.19352509376760066
      0.24886789762862677
    263.96385812493844
    108.84058939146061
     40.700062261124714]
	Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
	μ=9.468e-3 # natural human death rate
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = pp[2:14]
 	p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=pp[1]
	IA0=pp[15]
 	P0=pp[16]
 	N0=S0+E0+IA0+IS0+R0
 	X0=[S0,E0,IA0,IS0,R0,R10,P0,D0,N0]
p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]

prob = ODEProblem(F, X0, tSpan, p)
sol = solve(prob, alg_hints=[:stiff]; p=p, saveat=1)
# plot(reduce(vcat,sol.u')[:,8])
Pred=reduce(vcat,sol.u')[:,[4,6, 8]]
rmsd([C TrueR TrueD], Pred)


using Plots
plot(reduce(vcat,sol.u')[:,4])
scatter!(C)

plot(reduce(vcat,sol.u')[:,5])
scatter!(TrueR)
plot!(reduce(vcat,sol.u')[:,6])

plot(reduce(vcat,sol.u')[:,8])
scatter!(TrueD)

#population
plot(reduce(vcat,sol.u')[:,9])



##
# Open the file
AA=readlines("Output_CSC/26huhti ODE/Output_CSC_ODE15.txt")

BB=map(x -> parse.(Float64, split(x)), AA)

plot(; legend=false)
Err=zeros(length(BB))
for ii in 1:length(BB)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[ii][2:14]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=BB[ii][1]
	IA0=BB[ii][15]
	P0=BB[ii][16]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,R10,P0,D0,N0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	# if reduce(vcat,sol.u')[50,4] < 840
	Pred1=reduce(vcat,sol.u')[:,[4,5]]
	plot!(Pred1[:,1]; alpha=0.1, color="#BBBBBB")
	Err[ii]=rmsd([C TrueR], Pred1)
	# end
end

# Plot real
scatter(C)

##
plot(; legend=false)
Err=zeros(length(BB))
for ii in 1:length(BB)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[ii][2:14]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=BB[ii][1]
	IA0=BB[ii][15]
	P0=BB[ii][16]
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
