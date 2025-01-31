# parameters have the same time scale with derivatives

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

pp=[ 157318.78828818825
0.0022059202987591317
2.3448911642017564e-7
0.3625933709684469
6.057533128618978e-7
1.9283759794330223e-7
0.13416286289661633
0.09998766782492906
0.9929089042420249
0.0018995802105997222
0.0001398060576871766
0.0712215480882499
0.010706419935342127
0.5343330884381733
0.22214014334932028
0.09769883534491229
1.3464832468146044
0.43194058487679143]

Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate

μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T = pp[2:15]
S0=pp[1]
	E0=pp[16]
	IA0=pp[17]
 	P0=pp[18]
IS0=17;R0=0;RT0=0;D0=0;DT0=0;
x0=[S0,E0,IA0,IS0,R0,RT0,P0,D0,DT0]

Order=ones(9)
par = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T, Order[1:7]]


tSpan=[1,length(C)]

# Define the equation

function  F(t, x, par)

    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T=par[1:16]
    S,E,IA,IS,R,R1,P,D,D1=x
    α=par[17][:]

    dS= Λ^(α[1]) - β1^(α[1])*S*P/(1+ϕ1*P) - β2^(α[1])*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ^(α[1])*E - µ^(α[1])*S
    dE= β1^(α[2])*S*P/(1+ϕ1*P)+β2^(α[2])*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ^(α[2])*E - μ^(α[2])*E - ω^(α[2])*E
    dIA= (1-δ)*ω*E - (μ^(α[3])+σ^(α[3]))*IA - γA^(α[3])*IA
    dIS= δ*ω^(α[4])*E - (μ^(α[4]) + σ^(α[4]))*IS - γS^(α[4])*IS
    dR=γS^(α[5])*IS + γA^(α[5])*IA - μ^(α[5])*R
    dR1=γS^(α[5])*IS + T*γA^(α[5])*IA - μ^(α[5])*R1
    dP=ηA^(α[6])*IA + ηS^(α[6])*IS - μp^(α[6])*P
    dD=σ^(α[7])*(IA+IS) - μ^(α[7])*D
    dD1=σ^(α[7])*(T*IA+IS) - μ^(α[7])*D1
    # dS= Λ - β1*S*P/(1+ϕ1*P) - β2*S*(IA + IS)/(1+ϕ2*(IA+IS)) + ψ*E - µ*S
    # dE= β1*S*P/(1+ϕ1*P)+β2*S*(IA+IS)/(1+ϕ2*(IA+IS)) - ψ*E - μ*E - ω*E
    # dIA= (1-δ)*ω*E - (μ+σ)*IA - γA*IA
    # dIS= δ*ω*E - (μ + σ)*IS - γS*IS
    # dR=γS*IS + γA*IA - μ*R
    # dR1=γS*IS + T*γA*IA - μ*R1
    # dP=ηA*IA + ηS*IS - μp*P
    # dD=σ*(IA+IS) - μ*D
    # dD1=σ*(T*IA+IS) - μ*D1
    
    return [dS,dE,dIA,dIS,dR,dR1,dP,dD,dD1]

end


#optimization
function loss(args)

	Order[1:5]=args[1:5]
    Order[6]=copy(Order[5])
    Order[7]=args[6]
    Order[8]=args[7]
    Order[9]=copy(Order[8])

	p=copy([Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T,args])
	if size(x0,2) != Int64(ceil(maximum(Order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, Order)
		Order[indx]=ones(length(indx))
	end
    
	_, x = FDEsolver(F, tSpan, x0, Order, p, h=.05, nc=4)
  PredI=x[1:20:end,4] .+ T.*x[1:20:end,3]
  PredR=x[1:20:end,6]
  PredD=x[1:20:end,9]
	rmsd([C TrueR TrueD], [PredI PredR PredD])

end
p_lo_1=vcat(.7*ones(7)) #lower bound
p_up_1=vcat(ones(7)) # upper bound
p_vec_1=vcat(1*ones(7))


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


##test
Result= [0.9999999999999989, 0.9697877072176395, 0.9999999999999825, 0.8771992786610435, 0.7546766615014835, .7546766615014835, 0.7000000000000313, 0.9999999999999134, .9999999999999134]
    
	_, x = FDEsolver(F, tSpan, x0, Result, [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T,Result[[1:5; 7; 8]]], h=.05, nc=4)
  PredI=x[1:20:end,4] .+ T.*x[1:20:end,3]
  PredR=x[1:20:end,6]
  PredD=x[1:20:end,9]
	rmsd([C TrueR TrueD], [PredI PredR PredD])



  using Plots
  plot(reduce(vcat,sol.u')[:,4])
  scatter!(C)
  
  plot(reduce(vcat,sol.u')[:,6])
  scatter!(TrueR)
  
  plot(reduce(vcat,sol.u')[:,9])
  scatter!(TrueD)