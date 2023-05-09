# plots for paramters obtained from Turing ODE 14, when Λ=Birth rate and fitting only to C and R.
# comparing the results with those modified by optimized fractional orders

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
DateTick2= Dates.format.(DateTick, "d u")

Err=zeros(length(BB))
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
		Err[ii]=rmsd(C, Pred1)
end

# Err=replace!(Err, 0=>Inf) # filter inacceptable results
pyplot()
Ind=sortperm(Err)
Candidate=BB[Ind[1:300]]
plot(; legend = false)
for ii in Ind[301:5:end]
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
	if reduce(vcat,sol.u')[50,4] < 2500
		plot!(DateTick2, Pred1[:,1]; alpha=0.7, color="#BBBBBB")
	end
end
for ii in 1:length(Candidate)
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = Candidate[ii][2:14]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=Candidate[ii][1]
	IA0=Candidate[ii][15]
	P0=Candidate[ii][16]
	E0=Candidate[ii][17]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
		Pred1=reduce(vcat,sol.u')[:,4]
		plot!(Pred1[:,1]; color="yellow1")
end
scatter!(C, color=:gray25,markerstrokewidth=0,
	title = "(a) Fitting paramters of the ODE model" , titleloc = :left,titlefont = font(9),ylabel="Daily new confirmed cases (South Africa)", xrotation=20)
#plot the best

valErr,indErr=findmin(Err)
display(["MinErr",valErr])
function myshowall(io, x, limit = false)
	println(io, summary(x), ":")
	Base.print_matrix(IOContext(io, :limit => limit), x)
end
non300=290

myshowall(stdout, BB[indErr,:], false)

μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[indErr][2:14]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=BB[indErr][1]
	IA0=BB[indErr][15]
	P0=BB[indErr][16]
	E0=BB[indErr][17]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	PlODE=plot!(reduce(vcat,sol.u')[:,4], lw=3, color=:darkorchid1,formatter = :plain)

Err1best=rmsd(C, reduce(vcat,sol.u')[:,4])

##
AAf=readlines("Covid_Shedding/Output_CSC/output_final300plot.txt")
BBf=map(x -> parse.(Float64, split(x)), AAf)

plot(; legend=false)
Order=ones(8)
Errf=zeros(length(BBf))
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
	Pred1=x1[1:20:end,4]
		plot!(DateTick2, Pred1; alpha=0.5, color="darkorange1")
		Errf[ii]=rmsd(C, Pred1)
end
scatter!(C, color=:gray25, markerstrokewidth=0,
	title = "(b) Fitting fractional order derivatives" , titleloc = :left,titlefont = font(9),ylabel="Daily new confirmed cases (South Africa)" , xrotation=20)
#plot the best
valErrf,indErrf=findmin(Errf)
display(["MinErrf",valErrf])

myshowall(stdout, BBf[indErrf,:], false)
i=Int(BBf[indErrf][1])
μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = Candidate[i][2:14]
	p1= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=Candidate[i][1]
	IA0=Candidate[i][15]
	P0=Candidate[i][16]
	E0=BBf[indErrf][9]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
	Order[1:6]=BBf[indErrf][2:7]
	Order[8]=BBf[indErrf][8]
	_, x1 = FDEsolver(Ff, [1,length(C)], X0, Order, p1, h=.05, nc=4)
	Pred1=x1[1:20:end,4]
	plFDE=plot!(Pred1, lw=3, color=:dodgerblue1,formatter=:plain)

Errfbest=rmsd(C, x1[1:20:end,4])
##
mean(Err[Ind][1:300])
std(Err[Ind][1:300])
var(Err[Ind][1:300])
median(Err[Ind][1:300])
mean(Errf[sortperm(Errf)][1:300])
std(Errf[sortperm(Errf)][1:300])
var(Errf[sortperm(Errf)][1:300])
median(Errf[sortperm(Errf)][1:300])

violin(repeat([""],outer=non300), Err[Ind][1:non300], side = :left,
	c=:yellow1, label="Model with integer orders",
	title = "(d) Density of RMSD values " ,titlefont = font(9), titleloc = :left)
# boxplot!(ones(non300), Err[Ind][1:non300], side = :left, fillalpha=0.75, linewidth=.02)
# dotplot!(ones(non300), Err[Ind][1:non300], side = :left, marker=(:black, stroke(0)))
	violin!(repeat([""],outer=non300), Errf[sortperm(Errf)][1:non300], side = :right, c=:darkorange1, label="Model with modifyed derivatives", legendposition=(.62,.9))
	# scatter!([1.], [mean(Err[Ind][1:non300])])
	# scatter!([1.], [mean(Errf[sortperm(Errf)][1:non300])])
	plot!([0;.5], [Err1best; Err1best], lw=3, legend=false,color=:darkorchid1)
		annotate!(.25, Err1best+24, text("Min RMSD=$(round(Err1best,digits=4))", :darkorchid1,:top, 7))
		plot!([.5;1], [Errfbest; Errfbest], lw=3, legend=false,color=:dodgerblue1)
		plRMSD=annotate!(.8, Errfbest+24, text("Min RMSD=$(round(Errfbest,digits=4))", :dodgerblue1,:top, 7))
# boxplot(reduce(vcat,BB[Ind][1:300]')[:,2:14],legend=:false)
plbox1=boxplot(repeat(["μp" "ϕ2" "δ" "ψ" "ω" "σ" "γA" "ηS" "ηA"],outer=300),reduce(vcat,BB[Ind][1:300]')[:,[2,4,7,8,9,10,12,13,14]], legend=:false,
	title = "(c) Paramter values for top 300 fits", titlefont = font(9) , titleloc = :left, color=:white, bar_width = 0.9,marker=(0.2, :black, stroke(0)))
plbox2=boxplot(repeat(["ϕ1" "β1" "β2" "γS"],outer=300),reduce(vcat,BB[Ind][1:300]')[:,[3,5,6,11]],legend=:false, yaxis=:log,color=:white,bar_width = .9, marker=(0.2, :black, stroke(0)))
boxplot([reduce(vcat,BB[Ind][1:300]')[:,[1,15,16]] reduce(vcat,BBf')[:,9]],legend=:false, yaxis=:log)


mE0=mean(reduce(vcat,BBf')[:,9])
mdE0=median(reduce(vcat,BBf')[:,9]*1e10)
stdE0=std(reduce(vcat,BBf')[:,9])
mS0IA0P0=vec(mean(reduce(vcat,BB[Ind][1:300]')[:,[1,15,16]],dims=1));mIC=[mS0IA0P0[1], mE0, mS0IA0P0[2], mS0IA0P0[3]]
stdS0IA0P0=vec(std(reduce(vcat,BB[Ind][1:300]')[:,[1,15,16]],dims=1));stdIC=[stdS0IA0P0[1], stdE0, stdS0IA0P0[2], stdS0IA0P0[3]]
mdS0IA0P0=vec(median(reduce(vcat,BB[Ind][1:300]')[:,[1,15,16]],dims=1));mdIC=[mdS0IA0P0[1], mdE0, mdS0IA0P0[2], mdS0IA0P0[3]]
OptIC=[Candidate[i][1], BBf[indErrf][9], Candidate[i][15], Candidate[i][16]]

##
function rep_num1(PP)
    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=PP
    ωe=ψ+μ+ω
    ωia=μ+σ+γA
    ωis=μ+σ+γS
 R = Λ*ω/μ*((β1*δ*ηS)/(μp*ωe*ωis)+(β2*δ)/(ωe*ωis)+(β1*(1-δ)*ηA)/(μp*ωe*ωia)+(β2*(1-δ))/(ωe*ωia))
 return R
end

parameters=["μp","ϕ1","ϕ2","β1","β2",
			"δ","ψ","ω","σ","γS","γA","ηS","ηA"]

OptPar=[0.6641890543353863, 9.088630587992902e-7, 0.06344949244030884, 7.694078495487392e-6, 1.6779006579837752e-5, 0.9337865215267538, 0.00814228008865711, 0.9677337168567939, 0.3194429121145711, 0.00023617022276914285, 0.8667486940960136, 0.21882043013719396, 0.9048429015159145]
mPar=vec(mean(reduce(vcat,BB[Ind][1:300]')[:,2:14],dims=1))
stdPar=vec(std(reduce(vcat,BB[Ind][1:300]')[:,2:14],dims=1))
mdPar=vec(median(reduce(vcat,BB[Ind][1:300]')[:,2:14],dims=1))

dfPar=DataFrame(Parameters=parameters, mean=mPar, std=stdPar, median=mdPar, optimized_value=OptPar)

show(IOContext(stdout, :limit=>false), dfPar)

IC=["S0", "E0", "IA0", "P"]

df1=DataFrame(Initial_conditions=IC, mean=mIC, std=stdIC, median=mdIC, optimized_value=OptIC)

show(IOContext(stdout, :limit=>false), df1)

mO=vec(mean(reduce(vcat,BBf')[:,2:8],dims=1))
stdO=vec(std(reduce(vcat,BBf')[:,2:8],dims=1));stdIC=[stdS0IA0P0[1], stdE0, stdS0IA0P0[2], stdS0IA0P0[3]]
mdO=vec(median(reduce(vcat,BBf')[:,2:8],dims=1));mdIC=[mdS0IA0P0[1], mdE0, mdS0IA0P0[2], mdS0IA0P0[3]]
OptO=round.(BBf[i][2:8],digits=6)

Orders=["αS", "αE", "αIA", "αIS", "αR","αP","αN"]

df2=DataFrame(Derivative_Orders=Orders, mean=mO, std=stdO, median=mdO, optimized_value=OptO)

show(IOContext(stdout, :limit=>false), df2)


##

plot([reduce(vcat,sol.u')[:,1] x1[1:20:end,1]], label="S")

plot([reduce(vcat,sol.u')[:,2] x1[1:20:end,2]], label="E")

plot([reduce(vcat,sol.u')[:,3] x1[1:20:end,3]], label="IA")

plot([reduce(vcat,sol.u')[:,4] x1[1:20:end,4]], label="IS")
scatter!(C)

plot([reduce(vcat,sol.u')[:,5] x1[1:20:end,5]],label="R")

plot([reduce(vcat,sol.u')[:,6] x1[1:20:end,6]], label="P")

plot([reduce(vcat,sol.u')[:,7] x1[1:20:end,7]], label="D")

#population
plot([reduce(vcat,sol.u')[:,8] x1[1:20:end,8]], label="N")

L=@layout[grid(1,2) ; [b{0.33w, .7h}  b{0.15w, .7h} b{0.52w, .7h}]]
Plotall=plot(PlODE,plFDE, plbox1,plbox2,plRMSD, layout =L,size=(700,600), guidefont=font(8), legendfont=font(8))

savefig(PlODE,"plODE.png")
savefig(plFDE,"plFDE.png")
savefig(plRMSD,"plRMSD.png")
savefig(Plotall,"plAll.png")

savefig(PlODE,"plODE.svg")
savefig(plFDE,"plFDE.svg")
savefig(plRMSD,"plRMSD.svg")
savefig(Plotall,"plAll.svg")

## sensitivity
function rep_num1(PP)
    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=PP
    ωe=ψ+μ+ω
    ωia=μ+σ+γA
    ωis=μ+σ+γS
 R = Λ*ω/μ*((β1*δ*ηS)/(μp*ωe*ωis)+(β2*δ)/(ωe*ωis)+(β1*(1-δ)*ηA)/(μp*ωe*ωia)+(β2*(1-δ))/(ωe*ωia))
 return R
end

par1=[9.468e-3,19.995e-3,0.6641890543353863, 9.088630587992902e-7, 0.06344949244030884, 7.694078495487392e-6, 1.6779006579837752e-5, 0.9337865215267538, 0.00814228008865711, 0.9677337168567939, 0.3194429121145711, 0.00023617022276914285, 0.8667486940960136, 0.21882043013719396, 0.9048429015159145]


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

df=DataFrame(Parameters=parameters, sensitivity=R01, value=par1)

show(IOContext(stdout, :limit=>false), df)

show("R0=$(rep_num1(par1))")


## sensitivity density

R0=zeros(length(Candidate))
Sens=zeros(length(Candidate),15)
for j in 1:length(Candidate)
		μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[j][2:14]

	par1=[9.468e-3,19.995e-3,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	R0[j]=rep_num1(par1)


	for i in 1:15
	    Arg=par1
	            Arg1=copy(par1)
	            Arg1[i]=Arg[i]+0.01
	            Sens[j,i]=(rep_num1(Arg1)-rep_num1(Arg))*Arg[i]/(rep_num1(Arg)*0.01)
	end
end


# boxplot(repeat(["μ" "Λ" "μp" "ϕ1" "ϕ2" "β1" "β2" "δ" "ψ" "ω" "σ" "γS" "γA" "ηS" "ηA" "R0"],outer=300),sign.(hcat(Sens,R0)).*log10.(abs.(hcat(Sens,R0)).+1), legend=:false, #outliers=false,
# 	title = "(c) Paramter values for top 300 fits", titlefont = font(9) , titleloc = :left, color=:white, bar_width = 0.9,marker=(0.2, :black, stroke(0)), ylabel="sign(x) * log(|x| + 1)")
# plboxSen=boxplot(repeat(["μ" "Λ" "μp" "ϕ1" "ϕ2" "β1" "β2" "δ" "ψ" "ω" "σ" "γS" "γA" "ηS" "ηA"],outer=300),sign.(Sens).*log10.(abs.(Sens).+1), legend=:false, outliers=false,
# 		title = "(a) Paramter sensitivity for top 300 fits",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)), ylabel="sign(x) * log10(|x| + 1)")
# plboxR0=violin(repeat(["R0"],outer=300),sign.(R0).*log10.(abs.(R0).+1), legend=:false,# outliers=false,
# 			 title = "(b) Density of R0",titlefont = font(10) , titleloc = :left, color=:black)
plboxSen=boxplot(repeat(["μ" "Λ" "μp" "ϕ1" "ϕ2" "β1" "β2" "δ" "ψ" "ω" "σ" "γS" "γA" "ηS" "ηA"],outer=300),Sens, legend=:false, outliers=false,
	title = "(a) Paramter sensitivity for top 300 fits",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)))
plboxR0=boxplot(repeat(["R0"],outer=300),R0, legend=:false,# outliers=false,
	 title = "(b) Density of R0",titlefont = font(10) , titleloc = :left, color=:white, marker=(0.2, :black, stroke(0)),bar_width = .8, xaxis = ((0, 1), 0:1), yaxis=:log)

L=@layout[b b{0.2w}]
PlotSenR0=plot(plboxSen,plboxR0, layout =L, size=(650,300))#, guidefont=font(8), legendfont=font(8))

savefig(PlotSenR0,"PlotSenR0.svg")
