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

S0= 260276.0124643658;E0=292.3228126168938;IA0=22.66704909995665;IS0=17;R0=0;R10=0;P0= 13.239275804383805;D0=0;N0=S0+E0+IA0+IS0+R0
x0=[S0,E0,IA0,IS0,R0,R10,P0,D0,N0] # initial conditons S0,E0,IA0,IS0,R0,P0,D0,N0


pp=[ 0.15710660446485994
	3.2368572156584134e-5
	0.9806327153319467
	6.604793841780076e-5
	9.617697564292186e-5
	0.9988101372380245
	0.001655255062422364
	0.07638883095519367
	0.9999911321929686
	0.025687039235518315
	0.9500307616083895
	0.017925463276958195
	0.3452665875137063]
	Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
 	μ=9.468e-3 # natural human death rate

# [2.6974157505826046e-5, 299.9999981510134, 25.125975869263506, 1.999999653354374, 0.8723562698919407, 0.7066478097570968, 1.9999999811254363]
μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA = pp[1:13]

par = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]

Ndays=length(C)
tSpan=[1,Ndays]

# Define the equation

function  F(t, x, par)

	Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=par
    S,E,IA,IS,R,R1,P,D,N=x

    dS= Λ*N - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    dE= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    dIA= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    dIS= δ*ω*E - (μ + σ)*IS - γS*IS
    dR=γS*IS + γA*IA - μ*R
	dR1=γS*IS - μ*R1
    dP=ηA*IA + ηS*IS - μp*P
    dD=σ*(IA+IS) - μ*D
    dN=Λ*N - σ*(IA+IS) - μ*N
    return [dS,dE,dIA,dIS,dR,dR1,dP,dD,dN]

end

Order=ones(9)
#optimization
function loss(args)

	Order[1:7]=args[1:7]
	Order[9]=args[8]
	# E0, IA0, P0 =args[8:10]
	# E0, IA0, P0, ϕ2, δ, ω, γA =args[8:14]
	# S0, E0, IA0, P0, γS, ω, δ, γA =args[8:15]
	# ϕ2, δ, ω, γA =args[1:4]
	# E0, ϕ2, ω, δ, γA, ηA =args[8:12]

	p=[Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]
	if 1 != Int64(ceil(maximum(Order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, Order)
		Order[indx]=ones(length(indx))
	end

	# N0=S0+E0+IA0+IS0+R0
	# x0=[S0,E0,IA0,IS0,R0,R10,P0,D0,N0]
	_, x = FDEsolver(F, tSpan, x0, Order, p, h=.05, nc=4)
	II=x[1:20:end,4];RR=x[1:20:end,6];
	add1=RR[116]*ones(Ndays)
	add2=RR[120]*ones(Ndays)
	add3=RR[125]*ones(Ndays)
	add4=RR[170]*ones(Ndays)
	add5=RR[175]*ones(Ndays)
	add6=RR[180]*ones(Ndays)

	Pred=[II RR add1 add2 add3 add4 add5 add6]

	rmsd([C TrueR TrueR[116]*ones(Ndays) TrueR[120]*ones(Ndays) TrueR[125]*ones(Ndays)	TrueR[170]*ones(Ndays) TrueR[175]*ones(Ndays) TrueR[180]*ones(Ndays)], Pred)

end
p_lo_1=vcat(.5*ones(8)) #lower bound
p_up_1=vcat(ones(8)) # upper bound
p_vec_1=vcat(.9999*ones(8))
# p_lo_1=vcat(.5*ones(7),zeros(3)) #lower bound
# p_vec_1=vcat(.9999*ones(7),E0, IA0, P0)
# p_up_1=vcat(ones(7),300*ones(3)) # upper bound
# p_lo_1=vcat(.5*ones(9),1e-3*ones(4)) #lower bound
# p_up_1=vcat(ones(9),ones(4)) # upper bound
# p_vec_1=vcat(.99999*ones(9), ϕ2,δ, ω, γA)


show("we are fitting orders and E0, ϕ2, ω, δ, γA")

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
