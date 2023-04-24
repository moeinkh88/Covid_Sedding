

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

#initial conditons and parameters
S0=1.765239357851899e6;E0=0;IA0=0;IS0=17;R0=0;P0=0;D0=0;N0=S0+E0+IA0+IS0+R0
X0=[S0,E0,IA0,IS0,R0,P0,D0,N0] # initial conditons S0,E0,IA0,IS0,R0,P0,D0,N0


pp=[ 0.03255008480745857
    14.858164922911593
    0.0019534731862089643
    1.9701078182517523e-7
    0.8167732148061541
    3.3303562586264358e-6
    4.915958509895935e-10
    0.5971825544021288
    0.0014978053497659918
    0.1199799323762963
    0.02020996148740671
    0.0017309900596145895
    0.2935998931476053
    2.000315974316195e-7
    0.48987945846046593]
μ ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA = pp[1:15]
ϕ1=1e-5
ηS=1e-5
β1=1e-5
β2=1e-5
par = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]

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

prob = ODEProblem(F, X0, tSpan, par)


# Open the file
AA=readlines("Res22huhti.txt")

BB=map(x -> parse.(Float64, split(x)), AA)

Err=zeros(20000)
for i in 1:20000
	pp=BB[i]
	μ ,Λ,μp,ϕ2,δ,ψ,ω,σ2,γS,γA,ηA = pp[2:12]
	p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=pp[1]
	IA0=pp[13]
	P0=pp[14]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,μ,μp,ϕ2,δ,ψ,ω,σ,γS,γA,ηA,P0,D0,N0]

	prob = remake(prob; p = p, u0 = X0)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	Pred=reduce(vcat,sol.u')[:,[4,5]]
	Err[i]=rmsd([C TrueR], Pred)

end

# sizBB=zeros(7000)
# for i=1:7000
# 	sizBB[i]=length(BB[i])
# end
#
# valBB,indBB=findmin(sizBB)

valErr,indErr=findmin(Err)

display(["MinErr",valErr])
function myshowall(io, x, limit = false)
  println(io, summary(x), ":")
  Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, BB[indErr,:], false)


μ ,Λ,μp,ϕ2,δ,ψ,ω,σ2,γS,γA,ηA = BB[indErr][2:12]
p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
S0=BB[indErr][1]
IA0=BB[indErr][13]
P0=BB[indErr][14]
N0=S0+E0+IA0+IS0+R0
X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]

# show(BB[indErr])
# [267850.2685925563, 0.016818172053174516, 1.2411757238810646, 0.00031540723990122294, 0.9833485009621881, 0.6030817365954939, 0.003420841776850588, 0.11689562364029779, 0.028714260367701618, 0.00018720515157755936, 0.1001155457698303, 0.09563763100681981, 0.040360142041651545, 0.05669642467242789]

prob = remake(prob; p = p1, u0 = X0)
sol = solve(prob, alg_hints=[:stiff]; saveat=1)
Pred=reduce(vcat,sol.u')[:,[4,5]]

using Plots
plot(Pred[:,1])
scatter!(C)

plot!(Pred[:,2])
scatter!(TrueR)
