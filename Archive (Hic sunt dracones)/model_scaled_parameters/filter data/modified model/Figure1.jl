# This code give the plots for I_S+T*I_A, R (concerning testing people), and Death by disease
# comparing ODE with FDE results and real data

using CSV, DataFrames
using Interpolations,LinearAlgebra
using FdeSolver
using SpecialFunctions, StatsBase, Random, DifferentialEquations
using CSV, DataFrames, Statistics
using Optim, FdeSolver, StatsBase
using Plots, StatsPlots


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

IS0=17;R0=0;RT0=0;D0=0;
Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate
tSpan=(1,length(C))

# Define the equations


function  F(dx, x, par, t)

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T=par
    S,E,IA,IS,R,R1,P,D=x

    dx[1]= Λ - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dx[2]= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dx[3]= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dx[4]= δ*ω*E - (μ + σ)*IS - γS*IS
    dx[5]=γS*IS + γA*IA - μ*R
    dx[6]=γS*IS + T*γA*IA - μ*R1
    dx[7]=ηA*IA + ηS*IS - μp*P
    dx[8]=σ*(IA+IS) - μ*D
    return nothing

end

function  Ff(t, x, par) # for FDE model

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T=par[1:16]
    S,E,IA,IS,R,R1,P,D=x
    α=par[17][:]

    dS= Λ^(α[1]) - β1^(α[1])*S*P/(1+ϕ1*P) - β2^(α[1])*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ^(α[1])*E - µ^(α[1])*S
    dE= β1^(α[2])*S*P/(1+ϕ1*P)+β2^(α[2])*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ^(α[2])*E - μ^(α[2])*E - ω^(α[2])*E
    dIA= (1-δ)*ω*E - (μ^(α[3])+σ^(α[3]))*IA - γA^(α[3])*IA
    dIS= δ*ω^(α[4])*E - (μ^(α[4]) + σ^(α[4]))*IS - γS^(α[4])*IS
    dR=γS^(α[5])*IS + γA^(α[5])*IA - μ^(α[5])*R
    dR1=γS^(α[5])*IS + T*γA^(α[5])*IA - μ^(α[5])*R1
    dP=ηA^(α[6])*IA + ηS^(α[6])*IS - μp^(α[6])*P
    dD=σ^(α[7])*(IA+IS) - μ^(α[7])*D
    
    return [dS,dE,dIA,dIS,dR,dR1,dP,dD]
    
end

# Open the file
AA=readlines("output2.txt")

BB=map(x -> parse.(Float64, split(x)), AA)

using Dates
DateTick=Date(2020,3,27):Day(1):Date(2020,9,22)
DateTick2= Dates.format.(DateTick, "d-m")

Err=zeros(length(BB))
for ii in 1:length(BB)
	
    μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[ii][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[ii][1]
	E0=BB[ii][16]
	IA0=BB[ii][17]
	P0=BB[ii][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob,alg_hints=[:stiff], abstol = 1e-12, reltol = 1e-12; saveat=1)
	II=sol[4,:] .+ T.*sol[3,:]
	RR=sol[6,:]
	DD=sol[8,:]
	Err[ii]=rmsd([C TrueR TrueD], [II RR DD])
end

# Plot Confirm cases
Ind=sortperm(Err)
Candidate=BB[Ind[1:300]]
plot(; legend = false)
for ii in Ind[301:5:end]
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[ii][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[ii][1]
	E0=BB[ii][16]
	IA0=BB[ii][17]
	P0=BB[ii][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
		Pred1=sol[4,:] .+ T.*sol[3,:]
	if reduce(vcat,sol.u')[50,4] < 2500
		plot!(DateTick2, Pred1[:,1]; alpha=0.7, color="gray72")
	end
end
for ii in 1:length(Candidate)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[ii][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[ii][1]
	E0=BB[ii][16]
	IA0=BB[ii][17]
	P0=BB[ii][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
		Pred1=sol[4,:] .+ T.*sol[3,:]
		plot!(Pred1[:,1]; color="gray45")
end
scatter!(C, color=:white,markerstrokewidth=1,xlabel="Date (days)",
	title = "(a) " , titleloc = :left,titlefont = font(9),ylabel="Daily new confirmed cases (South Africa)", xrotation=20)
#plot the best

valErr,indErr=findmin(Err)
display(["MinErr",valErr])
function myshowall(io, x, limit = false)
	println(io, summary(x), ":")
	Base.print_matrix(IOContext(io, :limit => limit), x)
end
non300=290

myshowall(stdout, BB[indErr,:], false)

μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[indErr][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[indErr][1]
	E0=BB[ii][16]
	IA0=BB[ii][17]
	P0=BB[ii][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	PlODE_I=plot!(sol[4,:] .+ T.*sol[3,:], lw=1.5, color=:black,formatter = :plain)

Err1best=rmsd(C, sol[4,:] .+ T.*sol[3,:])

# Plot R
plot(; legend = false)
for ii in Ind[301:5:end]
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[ii][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[ii][1]
	E0=BB[ii][16]
	IA0=BB[ii][17]
	P0=BB[ii][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
		Pred1=sol[6,:]
	if reduce(vcat,sol.u')[50,4] < 2500
		plot!(DateTick2, Pred1[:,1]; alpha=0.7, color="gray72")
	end
end
for ii in 1:length(Candidate)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[ii][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[ii][1]
	E0=BB[ii][16]
	IA0=BB[ii][17]
	P0=BB[ii][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
		Pred1=sol[6,:]
		plot!(Pred1[:,1]; color="gray45")
end
scatter!(TrueR, color=:white,markerstrokewidth=1,xlabel="Date (days)",
	title = "(b) " , titleloc = :left,titlefont = font(9),ylabel="Recovered individuals", xrotation=20)
#plot the best

valErr,indErr=findmin(Err)
display(["MinErr",valErr])
function myshowall(io, x, limit = false)
	println(io, summary(x), ":")
	Base.print_matrix(IOContext(io, :limit => limit), x)
end
non300=290

myshowall(stdout, BB[indErr,:], false)

μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[indErr][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[indErr][1]
	E0=BB[indErr][16]
	IA0=BB[indErr][17]
	P0=BB[indErr][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	PlODE_R=plot!(sol[6,:], lw=1.5, color=:black,formatter = :plain)

Err1best=rmsd(C, sol[6,:])


# Plot D
plot(; legend = false)
for ii in Ind[301:5:end]
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[ii][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[ii][1]
	E0=BB[ii][16]
	IA0=BB[ii][17]
	P0=BB[ii][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
		Pred1=sol[8,:]
	if reduce(vcat,sol.u')[50,4] < 2500
		plot!(DateTick2, Pred1[:,1]; alpha=0.7, color="gray72")
	end
end
for ii in 1:length(Candidate)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[ii][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[ii][1]
	E0=BB[ii][16]
	IA0=BB[ii][17]
	P0=BB[ii][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
		Pred1=sol[8,:]
		plot!(Pred1[:,1]; color="gray45")
end
scatter!(TrueD, color=:white,markerstrokewidth=1,xlabel="Date (days)",
	title = "(c) " , titleloc = :left,titlefont = font(9),ylabel="Deceased Individuals", xrotation=20)
#plot the best

valErr,indErr=findmin(Err)
display(["MinErr",valErr])
function myshowall(io, x, limit = false)
	println(io, summary(x), ":")
	Base.print_matrix(IOContext(io, :limit => limit), x)
end
non300=290

myshowall(stdout, BB[indErr,:], false)

μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[indErr][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[indErr][1]
	E0=BB[indErr][16]
	IA0=BB[indErr][17]
	P0=BB[indErr][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	PlODE_D=plot!(sol[8,    :], lw=1.5, color=:black,formatter = :plain)

Err1best=rmsd(TrueD, sol[8,:])

# plots for FDEs

AAf=readlines("Covid_Shedding/Output_CSC/output_final300plot.txt")
BBf=map(x -> parse.(Float64, split(x)), AAf)


# Plot c

plot(; legend=false)
Order=ones(8)
Errf=zeros(length(BBf))
for ii in 1:length(BBf)
	i=Int(BBf[ii][1])
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = Candidate[i][2:15]
	par= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=Candidate[i][1]
	IA0=Candidate[i][15]
	P0=Candidate[i][16]
	E0=BBf[ii][9]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]

	Order[1:6]=BBf[ii][2:7]
	Order[8]=BBf[ii][8]

	_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par, h=.05, nc=4)
	Pred1=x1[1:20:end,4]
		plot!(DateTick2, Pred1; alpha=0.5, color="gray45")
		Errf[ii]=rmsd(C, Pred1)
end