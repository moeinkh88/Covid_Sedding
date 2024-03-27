# parameters have the same time scale with derivatives
# This code optimize only the order of derivatives
# The optimized parameters are obtained from slurm-20962481

using CSV, DataFrames, Statistics
using Optim, FdeSolver, StatsBase

# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",70:250] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",70:250]
TrueR0=diff(Float64.(Vector(RData[1,:])))

# Calculate Q1, Q3, and IQR
Q1 = quantile(TrueR0, 0.25)
Q3 = quantile(TrueR0, 0.75)
IQR = Q3 - Q1

# Calculate the bounds
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

TrueR=copy(TrueR0)
# Filter the dataset to remove outliers
indR=findall(x -> x < lower_bound || x > upper_bound, TrueR0)
TrueR[indR]=(TrueR0[indR .- 1] + TrueR0[indR .+ 1])/2


dataset_D = CSV.read("time_series_covid19_deaths_global.csv", DataFrame) # all data of Recover
DData=dataset_D[dataset_D[!,2].=="South Africa",70:250]
TrueD0=diff(Float64.(Vector(DData[1,:])))

# Calculate Q1, Q3, and IQR
Q1 = quantile(TrueD0, 0.25)
Q3 = quantile(TrueD0, 0.75)
IQR = Q3 - Q1

# Calculate the bounds
lower_bound = Q1 - 2 * IQR
upper_bound = Q3 + 2 * IQR

TrueD=copy(TrueD0)
# Filter the dataset to remove outliers
indD=findall(x -> x < lower_bound || x > upper_bound, TrueD0)
TrueD[indD]=(TrueD0[indD .- 1] + TrueD0[indD .+ 1])/2

#initial conditons and parameters
i=parse(Int32,ARGS[1])

BB=CSV.read("output1.csv", DataFrame, header=0)
Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate
Order=ones(8)
μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T = BB[i,2:15]
par = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T, Order[1:7]]

S0=BB[i,1]
E0=BB[i,16]
IA0=BB[i,17]
P0=BB[i,18]

IS0=17;R0=0;RT0=0;D0=0;

x0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

tSpan=[1,length(C)]

# Define the equation

function  F(t, x, par)

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


#optimization
function loss(args)

	Order[1:5]=args[1:5]
    Order[6]=copy(Order[5])
    Order[7]=args[6]
    Order[8]=args[7]

	p=copy([Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T,args])
	if size(x0,2) != Int64(ceil(maximum(Order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, Order)
		Order[indx]=ones(length(indx))
	end
    
	_, x = FDEsolver(F, tSpan, x0, Order, p, h=.05, nc=4)
  PredI=x[1:20:end,4] .+ T.*x[1:20:end,3]
  PredR=x[1:20:end,6]
  PredD=x[1:20:end,8]
	rmsd([C TrueR TrueD], [PredI PredR PredD])

end
p_lo_1=vcat(.7*ones(7)) #lower bound
p_up_1=vcat(ones(7)) # upper bound
p_vec_1=vcat(.99999*ones(7))

show("We are fitting 7 orders for candidate $i")

Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.999), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(#outer_iterations = 10,
						  iterations=50,
						  show_trace=true,
						  show_every=5)
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