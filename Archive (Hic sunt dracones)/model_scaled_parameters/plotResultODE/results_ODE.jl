# new R and D, fit CRF
# same mortality of I_A and I_S
# consider tested I_A, proportion of considering I_A in data
# later we can consider scaled paramters

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

E0=0;IA0=100;IS0=17;R0=0;RT0=0;P0=100;D0=0;DT0=0;

tSpan=(1,length(C))

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


  pp=[ 157318.78828818825
        0.0022059202987591317
        2.3448911642017564e-7
        0.3625933709684469
        6.057533128618978e-7
        1.9283759794330223e-7
        0.13416286289661633
        0.09998766782492906
        0.9929089042420249
        0.0018995802105997222
        0.0001398060576871766
        0.0712215480882499
        0.010706419935342127
        0.5343330884381733
        0.22214014334932028
        0.09769883534491229
        1.3464832468146044
        0.43194058487679143]
	Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
	μ=9.468e-3 # natural human death rate
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


using Plots
plot(reduce(vcat,sol.u')[:,4])
scatter!(C)

plot(reduce(vcat,sol.u')[:,6])
scatter!(TrueR)

plot(reduce(vcat,sol.u')[:,9])
scatter!(TrueD)

#population
plot(reduce(vcat,sol.u')[:,8])


plot(reduce(vcat,sol.u')[:,6])
plot(reduce(vcat,sol.u')[:,9])
##
# Open the file
AA=readlines("./output.txt")

BB=map(x -> parse.(Float64, split(x)), AA)

plot(; legend=false)
Err=zeros(length(BB))
for ii in 1:length(BB)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[ii][2:15]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[ii][1]
	E0=BB[ii][16]
	IA0=BB[ii][17]
	P0=BB[ii][18]
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0,DT0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	if reduce(vcat,sol.u')[45,5] < 2.5e3
	Pred1=reduce(vcat,sol.u')[:,4]
	plot!(Pred1; alpha=0.1, color="#BBBBBB")
	Err[ii]=rmsd(C, Pred1)
	# Err[ii]=rmsd(TrueR, Pred1)
	end
end

# Plot real
scatter!(C)
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


μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[indErr][2:15]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[indErr][1]
	E0=BB[indErr][16]
	IA0=BB[indErr][17]
	P0=BB[indErr][18]
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0,DT0]
prob = ODEProblem(F, X0, tSpan, p1)
sol = solve(prob, alg_hints=[:stiff]; saveat=1)
plot!(reduce(vcat,sol.u')[:,4])

# rmsd(TrueR, reduce(vcat,sol.u')[:,5])
rmsd(C, reduce(vcat,sol.u')[:,4])
