# This code is for fitting parameters XXXX when others parameters are obtained from bounded initial fitting

using CSV, DataFrames
using Optim, FdeSolver, StatsBase

# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",70:250] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

# dataset_D = CSV.read("time_series_covid19_deaths_global.csv", DataFrame) # all data of Death
# DeathData=dataset_D[dataset_D[!,2].=="South Africa",70:250]
# TrueDeath=diff(Float64.(Vector(DeathData[1,:])))

dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",70:250]
TrueR=diff(Float64.(Vector(RData[1,:])))

#initial conditons and parameters

S0=384621.16731317673;E0=0;IA0=1.8704631687194186;IS0=17;R0=0;P0= 0.019698093787410116;D0=0;N0=S0+E0+IA0+IS0+R0
x0=[S0,E0,IA0,IS0,R0,P0,D0,N0] # initial conditons S0,E0,IA0,IS0,R0,P0,D0,N0


pp=[ 0.018236598326358076
	 3.0041028711221203
	 0.0005177431479234676
	 1.6777926187911554e-5
	 0.9538804775846411
	 3.937544708359634e-5
	 1.0126849893464974e-5
	 0.7268370872120027
	 0.0001273494602870871
	 0.093281514596918
	 0.0457507212548359
	 0.00013086213246148867
	 0.7467736441632898
	 0.004755841284461715
	 0.003686925296597469]
μ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA = pp[1:15]

par = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]


tSpan=[1,length(C)]

# Define the equation

function  F(t, x, par)

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

Order=ones(8)
#optimization
function loss(args)

	Order[1:6]=args[1:6]
	Λ,ϕ2,δ,γA = args[7:10]

	p=[Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]
	if size(x0,2) != Int64(ceil(maximum(Order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, Order)
		Order[indx]=ones(length(indx))
	end
	_, x = FDEsolver(F, tSpan, x0, Order, p, h=.05, nc=4)
	Pred=x[1:20:end,[4,5]]
	rmsd([C TrueR], Pred)

end
p_lo_1=vcat(.5*ones(6),1,1e-5*ones(3)) #lower bound
p_up_1=vcat(ones(6),300,3*ones(3)) # upper bound
p_vec_1=vcat(.999999*ones(6),Λ,ϕ2,δ,γA)


Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.999), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(#outer_iterations = 10,
						  iterations=100,
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
