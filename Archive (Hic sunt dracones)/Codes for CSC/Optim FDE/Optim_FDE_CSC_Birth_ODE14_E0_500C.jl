# This code is for fitting parameters XXXX when others parameters are obtained from unbounded initial fitting and Λ is birth rate

using CSV, DataFrames
using Optim, FdeSolver, StatsBase

# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",70:250] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",70:250]
TrueR=diff(Float64.(Vector(RData[1,:])))

#initial conditons and parameters
i=parse(Int32,ARGS[1])

BB=CSV.read("Candidate500C.csv", DataFrame, header=0)
μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = BB[i,2:14]
Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate
par= [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]

S0=BB[i,1]
IA0=BB[i,15]
P0=BB[i,16]

IS0=17;R0=0;D0=0;

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
    dD=σ*(IA+IS) - μ*D
    dN=Λ*N - σ*(IA+IS) - μ*N
    return [dS,dE,dIA,dIS,dR,dP,dD,dN]

end

Order=ones(8)
#optimization
function loss(args)

	Order[1:6]=args[1:6]
	Order[8]=args[7]
	E0 =args[8]
	# E0, IA0, P0, ϕ2, δ, ω, γA =args[8:14]
	# S0, E0, IA0, P0, γS, ω, δ, γA =args[8:15]
	# ϕ2, δ, ω, γA =args[8:11]
	# # E0, ϕ2, ω, δ, γA, ηA =args[8:12]

	if 1 != Int64(ceil(maximum(Order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, Order)
		Order[indx]=ones(length(indx))
	end

	N0=S0+E0+IA0+IS0+R0
	x0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
	_, x = FDEsolver(F, tSpan, x0, Order, par, h=.05, nc=4)
	Pred=x[1:20:end,4]
	rmsd(C, Pred)

end
p_lo_1=vcat(.5*ones(7),0) #lower bound
p_up_1=vcat(ones(7),300) # upper bound
p_vec_1=vcat(.99999*ones(7),35)


show("We are fitting orders and E0 for candidate $i")

Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.999), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(#outer_iterations = 1,
						  iterations=100,
						  show_trace=true,
						  show_every=1)
			)



display(Res)
display("RMSD=$(Res.minimum), candidate=$i")
Result=vcat(Optim.minimizer(Res))
# display(Result)

function myshowall(io, x, limit = false)
  println(io, summary(x), ":")
  Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, Array(Result), false)
