# Plots bifurcation diagram for R0 and N and R0 and I_S
# we use 300 results

using Statistics
using CSV, DataFrames
using Interpolations,LinearAlgebra
using Optim, FdeSolver
using SpecialFunctions, StatsBase, Random, DifferentialEquations
using Plots, StatsPlots

# Dataset
dataset_CC = CSV.read("Covid_Shedding/time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",70:250] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

#initial conditons and parameters

IS0=17;R0=0;D0=0;
Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate
tSpan=(1,length(C))

# Define the equation


function  F(dx, x, par, t)

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=par
    S,E,IA,IS,R,P,D,N=x

    dx[1]= Λ*N - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dx[2]= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dx[3]= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dx[4]= δ*ω*E - (μ + σ)*IS - γS*IS
    dx[5]=γS*IS + γA*IA - μ*R
    dx[6]=ηA*IA + ηS*IS - μp*P
    dx[7]=σ*(IA+IS) - μ*D
    dx[8]=Λ*N - σ*(IA+IS) - μ*N
    return nothing

end

function  Ff(t, x, par)

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=par
    S,E,IA,IS,R,P,D,N=x

    dS= Λ*N - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dE= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dIA= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dIS= δ*ω*E - (μ + σ)*IS - γS*IS
    dR=γS*IS + γA*IA - μ*R
    dP=ηA*IA + ηS*IS - μp*P
    dD=σ*(IA+IS) - μ*D
    dN=Λ*N - σ*(IA+IS) - μ*N
    return [dS,dE,dIA,dIS,dR,dP,dD,dN]

end

# Open the file
AA=readlines("Covid_Shedding/Output_CSC/output14CSC.txt")

BB=map(x -> parse.(Float64, split(x)), AA)

using Dates
DateTick=Date(2020,3,27):Day(1):Date(2020,9,22)
DateTick2= Dates.format.(DateTick, "d-m")

#find all errors for ODE
Err=zeros(length(BB))
NODE=zeros(length(BB),180)
for ii in 1:length(BB)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[ii][2:14]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=BB[ii][1]
	IA0=BB[ii][15]
	P0=BB[ii][16]
	E0=BB[ii][17]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
		Pred1=reduce(vcat,sol.u')[:,4]
		NODE[ii,:]=reduce(vcat,sol.u')[:,8]
		Err[ii]=rmsd(C, Pred1)
end

Ind=sortperm(Err)
Candidate=BB[Ind[1:300]] # 300 best fitted paramters

##
AAf=readlines("Covid_Shedding/Output_CSC/output_final300plot.txt")
BBf=map(x -> parse.(Float64, split(x)), AAf)

Order=ones(8)
NFDE=zeros(length(BBf), 180)
for ii in 1:length(BBf)
	i=Int(BBf[ii][1])
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = Candidate[i][2:14]
	par= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=Candidate[i][1]
	IA0=Candidate[i][15]
	P0=Candidate[i][16]
	E0=BBf[ii][9]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]

	Order[1:6]=BBf[ii][2:7]
	Order[8]=BBf[ii][8]

	_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par, h=.05, nc=4)
	NFDE[ii,:]=x1[1:20:end,8]
end

## bifurcation plot

RODE=zeros(300)
RFDE=zeros(300)
for i=1:300
	N1=mean(NODE[Ind[i],:])
	N2=mean(NFDE[i,:])
	PP=[9.468e-3; 19.995e-3; Candidate[i][2:14]]
    μ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=PP
    ωe=ψ+μ+ω
    ωia=μ+σ+γA
    ωis=μ+σ+γS
 RODE[i] = Λ*N1*ω/μ*((β1*δ*ηS)/(μp*ωe*ωis)+(β2*δ)/(ωe*ωis)+(β1*(1-δ)*ηA)/(μp*ωe*ωia)+(β2*(1-δ))/(ωe*ωia))
 RFDE[i] = Λ*N2*ω/μ*((β1*δ*ηS)/(μp*ωe*ωis)+(β2*δ)/(ωe*ωis)+(β1*(1-δ)*ηA)/(μp*ωe*ωia)+(β2*(1-δ))/(ωe*ωia))
end

scatter(RODE, mean(NODE[Ind[1:300],:]./length(C),dims=2))
 scatter!(RFDE, mean(NFDE./length(C),dims=2), xaxis=:log, yaxis=:log)

##

scatter(RODE, var(NODE[Ind[1:300],:]./length(C),dims=2))
 scatter!(RFDE, var(NFDE./length(C),dims=2), xaxis=:log, yaxis=:log)



## dynamic R0

dynN=reduce(vcat,sol.u')[:,8]
Rdyn=zeros(180)
function rep_numDyn(PP,i)
    μ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=PP
    ωe=ψ+μ+ω
    ωia=μ+σ+γA
    ωis=μ+σ+γS
 Rdyn[i] = Λ*dynN[i] .*ω/μ*((β1*δ*ηS)/(μp*ωe*ωis)+(β2*δ)/(ωe*ωis)+(β1*(1-δ)*ηA)/(μp*ωe*ωia)+(β2*(1-δ))/(ωe*ωia))
 return Rdyn
end


par1=[9.468e-3; 19.995e-3; BB[22][2:14]]

Rn=rep_numDyn(par1,1:180)

plot(Rn)
plot!(dynN, xaxis=:log, yaxis=:log)

## check feasibility
Bound=zeros(300)
for i=1:300
 Bound[i]= 19.995e-3- Candidate[i][10]- 9.468e-3
end

plot(Bound)

σ_candids = [vec[10] for vec in Candidate]

plot(σ_candids)
