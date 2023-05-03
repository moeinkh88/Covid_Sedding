# This code is for fitting parameters XXXX when others parameters are obtained from unbounded initial fitting and Λ is birth rate

using CSV, DataFrames
using Optim, FdeSolver, StatsBase

# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",70:250] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

dataset_D = CSV.read("time_series_covid19_deaths_global.csv", DataFrame) # all data of Death
DeathData=dataset_D[dataset_D[!,2].=="South Africa",70:250]
TrueD=diff(Float64.(Vector(DeathData[1,:])))

dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",70:250]
TrueR=diff(Float64.(Vector(RData[1,:])))

#initial conditons and parameters

S0= 25060.216457227714;E0=0;IA0=167.56655046390654;IS0=17;R0=0;P0=10.718478095775652;D0=0;N0=S0+E0+IA0+IS0+R0
x0=[S0,E0,IA0,IS0,R0,P0,D0,N0] # initial conditons S0,E0,IA0,IS0,R0,P0,D0,N0


pp=[ 0.17016670078596785
     1.781847356255663e-7
     0.6553727445190615
     1.279308675438442e-5
     4.305136647875811e-5
     0.8580730111477723
     0.08585144261559632
     0.6837796665562463
     0.11184180526657209
     6.752788944397819e-5
     0.9748087783772565
     0.11779422117231593
     0.061658367897578344]
	Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
 	μ=9.468e-3 # natural human death rate

# [2.6974157505826046e-5, 299.9999981510134, 25.125975869263506, 1.999999653354374, 0.8723562698919407, 0.7066478097570968, 1.9999999811254363]
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
    dD=σ*(IA+IS) - μ*D
    dN=Λ*N - σ*(IA+IS) - μ*N
    return [dS,dE,dIA,dIS,dR,dP,dD,dN]

end

Order=ones(8)
#optimization
function loss(args)

	E0=args[1]

	p=[Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]
	if 1 != Int64(ceil(maximum(Order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, Order)
		Order[indx]=ones(length(indx))
	end

	N0=S0+E0+IA0+IS0+R0
	x0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
	_, x = FDEsolver(F, tSpan, x0, Order, p, h=.05, nc=4)
	Pred=x[1:20:end,4]
	rmsd(C, Pred)

end

p_lo_1=vcat(0) #lower bound
p_up_1=vcat(300) # upper bound
p_vec_1=vcat(35)

Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.999), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(#outer_iterations = 10,
						  iterations=40,
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
