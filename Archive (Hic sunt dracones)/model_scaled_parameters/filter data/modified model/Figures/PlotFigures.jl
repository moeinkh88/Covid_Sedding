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
AA=readlines("output1.txt")

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
for ii in Ind[301:50:end]
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
		plot!(DateTick2, Pred1[:,1]; alpha=0.7, color="gray92")
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
	title = "(a) " , titleloc = :left,titlefont = font(9),ylabel="Daily New Confirmed Cases", xrotation=20)
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
	E0, IA0, P0=BB[indErr][16:18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	PlODE_I=plot!(sol[4,:] .+ T.*sol[3,:], lw=1.5, color=:black,formatter = :plain)

Err1best=rmsd(C, sol[4,:] .+ T.*sol[3,:])

# Plot R
plot(; legend = false)
for ii in Ind[301:50:end]
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
		plot!(DateTick2, Pred1[:,1]; alpha=.7, color="gray92")
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
	title = "(b) " , titleloc = :left,titlefont = font(9),ylabel="Recovered Individuals", xrotation=20)
#plot the best

μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[indErr][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[indErr][1]
	E0,IA0,P0=BB[indErr][16:18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	PlODE_R=plot!(sol[6,:], lw=1.5, color=:black,formatter = :plain)

Err1best=rmsd(TrueR, sol[6,:])


# Plot D
plot(; legend = false)
for ii in Ind[301:50:end]
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
		plot!(DateTick2, Pred1[:,1]; alpha=0.7, color="gray92")
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

μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[indErr][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[indErr][1]
	E0=BB[indErr][16]
	IA0=BB[indErr][17]
	P0=BB[indErr][18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	PlODE_D=plot!(sol[8,:], lw=1.5, color=:black,formatter = :plain)

Err1best=rmsd(TrueD, sol[8,:])

####################################### plots for FDEs

AAf=readlines("OrderMatrix.txt")
BBf=map(x -> parse.(Float64, split(x)), AAf)
Candid=CSV.read("outputODE.csv", DataFrame, header=0)

# Plot c

plot(; legend=false)
Order=ones(8)
Errf=zeros(length(BBf))
for i in 1:length(BBf)
	
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = Candid[i,2:15]
	
	S0=Candid[i,1]
	E0=Candid[i,16]
	IA0=Candid[i,17]
	P0=Candid[i,18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	Order[1:5]=BBf[i][1:5]
	Order[6]=copy(Order[5])
	Order[7:8]=BBf[i][6:7]

	par= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T,BBf[i]]

	_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par, h=.05, nc=4)
	PredI=x1[1:20:end,4] .+ T.*x1[1:20:end,3]
	PredR=x1[1:20:end,6]
	PredD=x1[1:20:end,8]
	plot!(DateTick2, PredI; alpha=0.5, color="gray45")
	Errf[i]=rmsd([C TrueR TrueD], [PredI PredR PredD])
end

scatter!(C, color=:white, markerstrokewidth=1,xlabel="Date (days)",
	title = "(d)" , titleloc = :left,titlefont = font(9),ylabel="Daily New Confirmed Cases" , xrotation=20)
#plot the best
valErrf,indErrf=findmin(Errf)
display(["MinErrf",valErrf])

myshowall(stdout, BBf[indErrf,:], false)

μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = Candid[indErrf,2:15]
	S0=Candid[indErrf,1]
	E0=Candid[indErrf,16]
	IA0=Candid[indErrf,17]
	P0=Candid[indErrf,18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]
	Order[1:5]=BBf[indErrf][1:5]
	Order[6]=copy(Order[5])
	Order[7:8]=BBf[indErrf][6:7]

	par= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T,BBf[indErrf]]
	_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par, h=.05, nc=4)
	PredI=x1[1:20:end,4] .+ T.*x1[1:20:end,3]

	plot!(DateTick2, PredI; alpha=0.5, color="gray45")
	plFDE_I=plot!(PredI, lw=1.5, color=:black,formatter=:plain)

Errfbest=rmsd(C, PredI)
 
### plot R
plot(; legend=false)
Order=ones(8)

for i in 1:length(BBf)
	
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = Candid[i,2:15]
	
	S0=Candid[i,1]
	E0=Candid[i,16]
	IA0=Candid[i,17]
	P0=Candid[i,18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	Order[1:5]=BBf[i][1:5]
	Order[6]=copy(Order[5])
	Order[7:8]=BBf[i][6:7]

	par= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T,BBf[i]]

	_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par, h=.05, nc=4)
	PredR=x1[1:20:end,6]
	plot!(DateTick2, PredR; alpha=0.5, color="gray45")
end

scatter!(TrueR, color=:white, markerstrokewidth=1,xlabel="Date (days)",
	title = "(e)" , titleloc = :left,titlefont = font(9),ylabel="Recovered Individuals" , xrotation=20)
#plot the best


μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = Candid[indErrf,2:15]
	S0=Candid[indErrf,1]
	E0=Candid[indErrf,16]
	IA0=Candid[indErrf,17]
	P0=Candid[indErrf,18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]
	Order[1:5]=BBf[indErrf][1:5]
	Order[6]=copy(Order[5])
	Order[7:8]=BBf[indErrf][6:7]

	par= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T,BBf[indErrf]]
	_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par, h=.05, nc=4)
	PredR=x1[1:20:end,6]

	plot!(DateTick2, PredR; alpha=0.5, color="gray45")
	plFDE_R=plot!(PredR, lw=1.5, color=:black,formatter=:plain)

Errfbest=rmsd(TrueR, PredR)


####plot D

plot(; legend=false)
Order=ones(8)

for i in 1:length(BBf)
	
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = Candid[i,2:15]
	
	S0=Candid[i,1]
	E0=Candid[i,16]
	IA0=Candid[i,17]
	P0=Candid[i,18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	Order[1:5]=BBf[i][1:5]
	Order[6]=copy(Order[5])
	Order[7:8]=BBf[i][6:7]

	par= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T,BBf[i]]

	_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par, h=.05, nc=4)
	PredD=x1[1:20:end,8]
	plot!(DateTick2, PredD; alpha=0.5, color="gray45")
end

scatter!(TrueD, color=:white, markerstrokewidth=1,xlabel="Date (days)",
	title = "(f)" , titleloc = :left,titlefont = font(9),ylabel="Deceased Individuals" , xrotation=20)
#plot the best


μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = Candid[indErrf,2:15]
	S0=Candid[indErrf,1]
	E0=Candid[indErrf,16]
	IA0=Candid[indErrf,17]
	P0=Candid[indErrf,18]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]
	Order[1:5]=BBf[indErrf][1:5]
	Order[6]=copy(Order[5])
	Order[7:8]=BBf[indErrf][6:7]

	par= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T,BBf[indErrf]]
	_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, par, h=.05, nc=4)
	PredD=x1[1:20:end,8]

	plot!(DateTick2, PredD; alpha=0.5, color="gray45")
	plFDE_D=plot!(PredD, lw=1.5, color=:black,formatter=:plain)

Errfbest=rmsd(TrueD, PredD)

L=@layout[grid(3,2)]

Plotall=plot(PlODE_I,plFDE_I,PlODE_R,plFDE_R,PlODE_D,plFDE_D, layout =L,size=(800,600), guidefont=font(8), legendfont=font(8))

################## plot errors
mean(Err[Ind][1:300])
std(Err[Ind][1:300])
var(Err[Ind][1:300])
median(Err[Ind][1:300])
mean(Errf[sortperm(Errf)][1:300])
std(Errf[sortperm(Errf)][1:300])
var(Errf[sortperm(Errf)][1:300])
median(Errf[sortperm(Errf)][1:300])

violin(repeat([""],outer=300), Err[Ind][1:300], side = :left,
	c=:white, label="Model with integer orders", #ylabel="Distribution of RMSD values",
	title = "Density of RMSD values" ,titlefont = font(13), titleloc = :left,ytickfontsize=11)

	violin!(repeat([""],outer=300), Errf[sortperm(Errf)][1:300], side = :right, c=:white, label="Model with modifyed derivatives", legendposition=(.62,.9))
	annotate!(.3, 1752, text("Integer Order model", :black,:top, 11),legend=false)
	annotate!(.7, 1752, text("Fractional Order model", :black,:top, 11),legend=false)
	plot!([0.3;.5], [valErr; valErr], legend=false, c=:black, linestyle=:dash)
	plot!([0.2;.5], [mean(Err[Ind][1:300]); mean(Err[Ind][1:300])], legend=false, c=:black, linestyle=:dash)
	annotate!(.16, mean(Err[Ind][1:300]), text("Mean:\n$(round(mean(Err[Ind][1:300]),digits=3))", :black,:top, 11),legend=false)
	annotate!(.8, mean(Errf[sortperm(Errf)][1:300]), text("Mean:\n$(round(mean(Errf[sortperm(Errf)][1:300]),digits=3))", :black,:top, 11),legend=false)
		plot!([0.5;.88], [mean(Errf[sortperm(Errf)][1:300]); mean(Errf[sortperm(Errf)][1:300])], legend=false, c=:black, linestyle=:dash)
		annotate!(.36, valErr-5, text("Minimum:\n$(round(valErr,digits=3))", :black,:top, 11),legend=false)
		plot!([.5;.7], [valErrf; valErrf], legend=false,color=:black,linestyle=:dash)
		plRMSD=annotate!(.8, valErrf+28, text("Minimum:\n$(round(valErrf,digits=4))", :black,:top, 11))

#######################plot values
plbox1=boxplot(repeat(["ϕ2" "δ" "ψ" "ω" "ηA"],outer=300),reduce(vcat,BB[Ind][1:300]')[:,[4,7,8,9,14]], legend=:false,
	title = "(c) parameter values for top 300 fits", titlefont = font(9) , titleloc = :left, color=:white, bar_width = 0.9,marker=(0.2, :black, stroke(0)))
plbox2=boxplot(repeat(["μp" "ϕ1" "β1" "β2" "σ" "γS" "γA" "ηS"],outer=300),reduce(vcat,BB[Ind][1:300]')[:,[2,3,5,6,10,11,12,13]],legend=:false, yaxis=:log,color=:white,bar_width = .9, marker=(0.2, :black, stroke(0)))
boxplot([reduce(vcat,BB[Ind][1:300]')[:,[1,16,17,18]]],legend=:false, yaxis=:log)

mS0IA0P0=vec(mean(reduce(vcat,BB[Ind][1:300]')[:,[1,16,17,18]],dims=1))
stdS0IA0P0=vec(std(reduce(vcat,BB[Ind][1:300]')[:,[1,16,17,18]],dims=1))
mdS0IA0P0=vec(median(reduce(vcat,BB[Ind][1:300]')[:,[1,16,17,18]],dims=1))
OptIC=[BB[indErr][1], BB[indErr][16] , BB[indErr][17] , BB[indErr][18]]

MEAN=mean(reduce(vcat,BB[Ind][1:300]')[:,:],dims=1)
STD=std(reduce(vcat,BB[Ind][1:300]')[:,:],dims=1)
VAR=var(reduce(vcat,BB[Ind][1:300]')[:,:],dims=1)
MEDIAN=median(reduce(vcat,BB[Ind][1:300]')[:,:],dims=1)
MEANf=mean(reduce(vcat,BBf[1:300]')[:,:],dims=1)
STDf=std(reduce(vcat,BBf[1:300]')[:,:],dims=1)
VARf=var(reduce(vcat,BBf[1:300]')[:,:],dims=1)
MEDIANf=median(reduce(vcat,BBf[1:300]')[:,:],dims=1)

################### sensitivity
function rep_num1(PP)
    μ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=PP[1:15]

	αS, αE, αIA, αIS, αR, αD, αW=PP[16:22]
    ωe=ψ^αE+ μ^αE + ω^αE
    ωia=μ ^ αIA + σ ^ αIA + γA ^ αIA
    ωis=μ ^ αIS + σ ^ αIS + γS ^ αIS
 
 R = (Λ / μ) ^ αS * (
    β2 ^ αE * (
        (δ * ω ^ αIS) / (ωis * ωe) + ((1 - δ) * ω ^ αIA) / (ωia * ωe)
    ) + β1 ^ αE * (
        (ηS ^ αW * δ * ω ^ αIS) / (μp ^ αW * ωe * ωis) + 
        (ηA ^ αW * (1 - δ) * ω ^ αIA) / (μp ^ αW * ωe * ωia)
    )
	)

 return R
end

Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate

par1=[9.468e-3
	19.995e-3
	0.003635875138058216
	8.151561722127852e-7
	0.43942169371267265
	2.804848603101783e-6
	8.863963267287063e-7
	0.4947151645777698
	0.0002071891280094621
	0.5396364150788414
	0.0012347012610145762
	0.0701231213714429
	0.00015520509627699186
	0.006363097612794978
	0.35851366456952516
	ones(7)]

R01=zeros(15)

for i in 1:15
    Arg=par1
            Arg1=copy(par1)
            Arg1[i]=Arg[i]+0.01
            R01[i]=(rep_num1(Arg1)-rep_num1(Arg))*Arg[i]/(rep_num1(Arg)*0.01)
end

function myshowall(io, x, limit = false)
  println(io, summary(x), ":")
  Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, R01, false)

using DataFrames
parameters=["μ","Λ","μp","ϕ1","ϕ2","β1","β2","δ","ψ","ω","σ","γS","γA","ηS","ηA"]

df=DataFrame(Parameters=parameters, sensitivity=R01, value=par1[1:15])

show(IOContext(stdout, :limit=>false), df)

## sensitivity density

R0=zeros(length(Candidate))
Sens=zeros(length(Candidate),15)
for j in 1:length(Candidate)
		μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[j][2:14]

	par1=[9.468e-3,19.995e-3,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	par1 = vcat(par1, ones(7))
	R0[j]= rep_num1(par1)


	for i in 1:15
	    Arg=par1
	            Arg1=copy(par1)
	            Arg1[i]=Arg[i]+0.01
	            Sens[j,i]=(rep_num1(Arg1)-rep_num1(Arg))*Arg[i]/(rep_num1(Arg)*0.01)
	end
end
plboxSen=boxplot(repeat(["μ" "Λ" "μp" "ϕ1" "ϕ2" "β1" "β2" "δ" "ψ" "ω" "σ" "γS" "γA" "ηS" "ηA"],outer=300),Sens, legend=:false, outliers=false,
	title = "(a) Parameter Sensitivity",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)),ylims=(-1, 1))
plboxR0=boxplot(repeat(["R0"],outer=300),R0, legend=:false, outliers=false,
	 title = "(b) Density of R0",titlefont = font(10) , titleloc = :left, color=:white, marker=(0.2, :black, stroke(0)),bar_width = .8, xaxis = ((0, 1), 0:1), yaxis=:log,ylims=(1e-3, 10))

## R0 without environment pathogens

R0=zeros(length(Candidate))
Sens=zeros(length(Candidate),15)
for j in 1:length(Candidate)
		μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[j][2:14]
	ϕ1,β1,ηS,ηA=zeros(4)
	par1=[9.468e-3,19.995e-3,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	par1 = vcat(par1, ones(7))
	R0[j]=rep_num1(par1)


	for i in 1:15
	    Arg=par1
	            Arg1=copy(par1)
	            Arg1[i]=Arg[i]+0.01
	            Sens[j,i]=(rep_num1(Arg1)-rep_num1(Arg))*Arg[i]/(rep_num1(Arg)*0.01)
	end
end
plboxSen0W=boxplot(repeat(["μ" "Λ" "μp" "ϕ1" "ϕ2" "β1" "β2" "δ" "ψ" "ω" "σ" "γS" "γA" "ηS" "ηA"],outer=300),Sens, legend=:false, outliers=false,
	title = "(c) Parameter Sensitivity",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)),ylims=(-1, 1))
plboxR00W=boxplot(repeat(["R0"],outer=300),R0, legend=:false, outliers=false,
	 title = "(d) Density of R0",titlefont = font(10) , titleloc = :left, color=:white, marker=(0.2, :black, stroke(0)),bar_width = .8, xaxis = ((0, 1), 0:1), yaxis=:log, ylims=(1e-3, 10))


L=@layout[b b{0.2w}; c c{.2w}]
PlotSenR0=plot(plboxSen,plboxR0,plboxSen0W,plboxR00W, layout =L, size=(750,600))#, guidefont=font(8), legendfont=font(8))


########## R0 for fractional
# check for optimized

par1f=copy([9.468e-3
	19.995e-3
	0.003635875138058216
	8.151561722127852e-7
	0.43942169371267265
	2.804848603101783e-6
	8.863963267287063e-7
	0.4947151645777698
	0.0002071891280094621
	0.5396364150788414
	0.0012347012610145762
	0.0701231213714429
	0.00015520509627699186
	0.006363097612794978
	0.35851366456952516])
par1f = vcat(par1f, [0.9999999999953245, 0.9611344150875429, 0.9999999998709871, 0.8024106717752314, 0.7000000000400238, 0.7144772230212804, 0.9999999998909126])

R01f=zeros(22)

for i in 1:22
    Arg=par1f
            Arg1=copy(par1f)
            Arg1[i]=Arg[i]+0.01
            R01f[i]=(rep_num1(Arg1)-rep_num1(Arg))*Arg[i]/(rep_num1(Arg)*0.01)
end

parametersf=["μ","Λ","μp","ϕ1","ϕ2","β1","β2","δ","ψ","ω","σ","γS","γA","ηS","ηA","αS", "αE", "αIA", "αIS", "αR" ,"αD", "αW"]

dff=DataFrame(Parameters=parametersf, sensitivity=R01f, value=par1[1:22])

show(IOContext(stdout, :limit=>false), dff)

## density


R0f=zeros(length(Candidate))
Sensf=zeros(length(Candidate),22)
for j in 1:length(Candidate)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[j][2:14]

par1f=[9.468e-3,19.995e-3,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
par1f = vcat(par1f, [0.9999999999953245, 0.9611344150875429, 0.9999999998709871, 0.8024106717752314, 0.7000000000400238, 0.7144772230212804, 0.9999999998909126])
R0f[j]=rep_num1(par1f)


for i in 1:22
	Arg=par1f
			Arg1=copy(par1f)
			Arg1[i]=Arg[i]+0.01
			Sensf[j,i]=(rep_num1(Arg1)-rep_num1(Arg))*Arg[i]/(rep_num1(Arg)*0.01)
end
end

plboxSenf=boxplot(repeat(["μ" "Λ" "μp" "ϕ1" "ϕ2" "β1" "β2" "δ" "ψ" "ω" "σ" "γS" "γA" "ηS" "ηA"],outer=300),Sensf[:,1:15], legend=:false, outliers=false,
	title = "(a) Parameter Sensitivity",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)),ylims=(-1, 1))
plboxSenforder=boxplot(repeat(["αS" "αE" "αIA" "αIS" "αR" "αD" "αW"],outer=300),Sensf[:,16:22], legend=:false, outliers=false,ylims=(-15, 10),
	title = "(b) Order Sensitivity",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)))
plboxR0f=boxplot(repeat(["R0"],outer=300),R0f, legend=:false, outliers=false,
	 title = "(c) Density of R0",titlefont = font(10) , titleloc = :left, color=:white, marker=(0.2, :black, stroke(0)),bar_width = .8, xaxis = ((0, 1), 0:1), yaxis=:log, ylims=(1e-3, 10))

# without pathogens
R0f=zeros(length(Candidate))
Sensf=zeros(length(Candidate),22)
for j in 1:length(Candidate)
		μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[j][2:14]
	ϕ1,β1,ηS,ηA=zeros(4)
	par1f=[9.468e-3,19.995e-3,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	par1f = vcat(par1f, [0.9999999999953245, 0.9611344150875429, 0.9999999998709871, 0.8024106717752314, 0.7000000000400238, 0.7144772230212804, 0.9999999998909126])
	R0f[j]=rep_num1(par1f)


	for i in 1:22
		Arg=par1f
				Arg1=copy(par1f)
				Arg1[i]=Arg[i]+0.01
				Sensf[j,i]=(rep_num1(Arg1)-rep_num1(Arg))*Arg[i]/(rep_num1(Arg)*0.01)
	end
end
plboxSen0Wf=boxplot(repeat(["μ" "Λ" "μp" "ϕ1" "ϕ2" "β1" "β2" "δ" "ψ" "ω" "σ" "γS" "γA" "ηS" "ηA"],outer=300),Sensf[:,1:15], legend=:false, outliers=false,
	title = "(d) Parameter Sensitivity",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)),ylims=(-1, 1))
plboxSen0Wforder=boxplot(repeat(["αS" "αE" "αIA" "αIS" "αR" "αD" "αW"],outer=300),Sensf[:,16:22], legend=:false, outliers=false,
	title = "(e) Order Sensitivity",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)),ylims=(-15, 10))
plboxR00Wf=boxplot(repeat(["R0"],outer=300),R0f, legend=:false, outliers=false,
	 title = "(f) Density of R0",titlefont = font(10) , titleloc = :left, color=:white, marker=(0.2, :black, stroke(0)),bar_width = .8, xaxis = ((0, 1), 0:1), yaxis=:log,  ylims=(1e-3, 10))


L=@layout[b b{0.3w} b{0.2w}; c c{0.3w} c{.2w}]
PlotSenR0=plot(plboxSenf,plboxSenforder,plboxR0f,plboxSen0Wf,plboxSen0Wforder,plboxR00Wf, layout =L, size=(770,550))#, guidefont=font(8), legendfont=font(8))
	
