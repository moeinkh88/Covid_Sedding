

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

function  Ff(t, x, par)

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=par
    S,E,IA,IS,R,P,D,N=x

    dS= Λ - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dE= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dIA= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dIS= δ*ω*E - (μ + σ)*IS - γS*IS
    dR=γS*IS + γA*IA - μ*R
    dP=ηA*IA + ηS*IS - μp*P
    dD=σ*(IA+IS)
    dN=Λ - σ*(IA+IS) - μ*N
    return [dS,dE,dIA,dIS,dR,dP,dD,dN]

end


  pp=[1.765239357851899e6
  	 0.03255008480745857
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
	 μ ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = pp[2:16]
 	p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=pp[1]
	IA0=0
 	P0=0
 	N0=S0+E0+IA0+IS0+R0
 	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]

prob = ODEProblem(F, X0, tSpan, p)
sol = solve(prob, alg_hints=[:stiff]; p=p, saveat=1)
# plot(reduce(vcat,sol.u')[:,8])
Pred=reduce(vcat,sol.u')[:,[4,5,7]]
rmsd([C TrueR TrueD], Pred)

# args= [0.9999999999999999, 0.9999999999999996, 0.9434424646516903, 0.9999999999999998, 0.9999999999999999, 0.8668878536002538, 0.9999999999999988, 0.5683746010638825, 0.1779528700316752]
# Order=ones(8)
# Order[1:6]=args[1:6]
# ϕ2,δ,γA  = args[7:9]

args = [0.9999999998004621, 0.9999999964668235, 0.9265172839266816, 0.9835508220131483, 0.999999999046695, 0.9999999907091582, 1.221807552538849e-5, 3.768795628177853, 0.592964835141523, 0.5983303276384108]
Order=ones(8)
Order[1:6]=args[1:6]
Λ,ϕ2,δ,ηA  = args[7:10]
# obtained for fitting CRF when P-Up=1 [0.999663461604957, 0.9999999999999964, 0.9408486963376657, 0.9999999999999716, 0.9999999999999992, 0.9999999999999962, 1.0000003690572517e-5, 0.9999999999999672, 0.5891565550722695, 0.5150955698380737]

# args= [0.9999999999999999, 0.9999999999999999, 0.9026022510689862, 0.9999999999999999, 0.9999999999999998, 0.9403095539891934, 1.0000000062648699e-5, 0.5770180436649379, 0.21956522105746495]
# Order=ones(8)
# Order[1:6]=args[1:6]
# Λ,δ,γA = args[7:9]

# β2,γS = args[7:8]150
# μ,μp,ϕ2,δ,ψ,ω,σ,γS,γA,ηA =args[7:16]

# args= [0.9645748457348268, 0.40000000018431675, 0.4000000001758517, 0.634822328006963, 0.8570365968827433, 0.4000000000000001, 1.0000000000000003e-5, 1.0014636271752807e-5, 1.0000000000000004e-5, 1.0002798481743851e-5]
# Order=ones(8)
# Order[1:6]=args[1:6]
# β1,β2,ηS,ϕ1 = args[7:10]

par1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par1, h=.05, nc=4)
Pred1=x1[1:20:end,[4,5,7]]
rmsd([C TrueR TrueD], Pred1)

using Plots
plot(reduce(vcat,sol.u')[:,4])
plot!(Pred1[:,1])
scatter!(C)

plot(reduce(vcat,sol.u')[:,5])
plot!(Pred1[:,2])
scatter!(TrueR)

plot(reduce(vcat,sol.u')[:,7])
plot!(Pred1[:,3])
scatter!(TrueD)

#population
# plot(reduce(vcat,sol.u')[:,8])
# plot!(x1[1:20:end,8])
