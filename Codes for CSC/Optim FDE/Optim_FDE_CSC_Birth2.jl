# This code is for fitting parameters XXXX when others parameters are obtained from unbounded initial fitting and Λ is birth rate

using CSV, DataFrames
using Optim, FdeSolver, StatsBase

# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",70:250] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

dataset_D = CSV.read("time_series_covid19_deaths_global.csv", DataFrame) # all data of Death
DeathData=dataset_D[dataset_D[!,2].=="South Africa",70:249]
TrueD=(Float64.(Vector(DeathData[1,:])))

dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",70:250]
TrueR=diff(Float64.(Vector(RData[1,:])))

#initial conditons and parameters

S0=7548.045231531718;E0=0;IA0=0.16156107176510295;IS0=17;R0=0;P0= 0.2591316689996923;D0=0;N0=S0+E0+IA0+IS0+R0
x0=[S0,E0,IA0,IS0,R0,P0,D0,N0] # initial conditons S0,E0,IA0,IS0,R0,P0,D0,N0


pp=[ 0.15782338058089418
	1.0743043176090302e-6
	0.4546546658049586
	0.0001120776533507358
	1.2758470580730523e-6
	0.7248572763638965
	0.1502191907331448
	0.8841167641517534
	0.030273913785749975
	0.0024970026839355617
	0.4343498498777916
	2.3307901908341236e-5
	0.49870372293270493]
	Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
 	μ=9.468e-3 # natural human death rate
μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA = pp[1:13]

par = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]


tSpan=[1,length(C)]

# Define the equation

function  F(t, x, par)

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=par
    S,E,IA,IS,R,P,D,N=x

    dS= Λ*N - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dE= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dIA= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dIS= δ*ω*E - (μ + σ)*IS - γS*IS
    dR=γS*IS + γA*IA - μ*R
    dP=ηA*IA + ηS*IS - μp*P
    dD=σ*(IA+IS)
    dN=Λ*N - σ*(IA+IS) - μ*N
    return [dS,dE,dIA,dIS,dR,dP,dD,dN]

end

#optimization
function loss(args)

	Order=args[1:8]

	if size(x0,2) != Int64(ceil(maximum(Order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, Order)
		Order[indx]=ones(length(indx))
	end
	_, x = FDEsolver(F, tSpan, x0, Order, par, h=.05, nc=4)
	Pred=x[1:20:end,[4,5,7]]
	rmsd([C TrueR TrueD], Pred)

end
p_lo_1=vcat(.5*ones(8)) #lower bound
p_up_1=vcat(ones(8)) # upper bound
p_vec_1=vcat(.7*ones(8))

show("we are fitting orders only, in this code: Optim_FDE_CSC_Birth2.jl")

Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.999), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(#outer_iterations = 10,
						  iterations=20,
						  show_trace=true,
						  show_every=1)
			)



display(Res)

Result=vcat(Optim.minimizer(Res))
# display(Result)

function myshowall(io, x, limit = false)
  println(io, summary(x), ":")
  Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, Array(Result), false)
