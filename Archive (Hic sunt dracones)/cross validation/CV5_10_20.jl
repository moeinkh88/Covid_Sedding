# This code is for fitting parameters XXXX when others parameters are obtained from unbounded initial fitting and Λ is birth rate

using CSV, DataFrames
using Optim, FdeSolver
using Statistics
using Interpolations,LinearAlgebra
using SpecialFunctions, StatsBase, Random, DifferentialEquations, Turing


# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="South Africa",70:250] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
CAll=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",70:250]
TrueRAll=diff(Float64.(Vector(RData[1,:])))

#For each tr and test
i=parse(Int32,ARGS[1])

Tr=60;
ts=10*(i-1)
C=CAll[1:Tr+ts]
Cp5=CAll[Tr+ts+1:Tr+ts+5]
Cp10=CAll[Tr+ts+1:Tr+ts+10]
Cp20=CAll[Tr+ts+1:Tr+ts+20]
TrueR=TrueRAll[1:Tr+ts]
R5=TrueRAll[Tr+ts+1:Tr+ts+5]
R10=TrueRAll[Tr+ts+1:Tr+ts+10]
R20=TrueRAll[Tr+ts+1:Tr+ts+20]


show("Prediction errors: $i")
#### ODE
S0=665188;E0=0;IA0=100;IS0=17;R0=0;P0=100;D0=0;N0=S0+E0+IA0+IS0+R0
x0=[S0,E0,IA0,IS0,R0,P0,D0,N0] # initial conditons S0,E0,IA0,IS0,R0,P0,D0,N0

Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate
μp=0.172499999999 # natural death rate of pathogens virusess
ϕ1=2e-6 # proportion of interaction with an infectious environment
ϕ2=0.3 #proportion of interaction with an infectious I
β1=0.00414 # infection rate from S to E due to contact with P
β2=0.0115 # infection rate from S to E due to contact with IA and/or IS
δ=0.7 #proportion of symptomatic infectious people
ψ=0.0051  #progression rate from E back to S due to robust immune system
ω=0.5 #progression rate from E to either IA or IS
σ=0.025 #disease induced death rate
γS=0.1 #rate of recovery of the symptomatic individuals
γA=1/6 #rate of recovery of the asymptomatic individuals
ηS=0.002 #rate of virus spread to environment by IS
ηA=0.002 #rate of virus spread to environment by IA

par=[Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]

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
    dx[7]=σ*(IA+IS)
    dx[8]=Λ*N - σ*(IA+IS) - μ*N
    return nothing

end

prob = ODEProblem(F, x0, tSpan, par)

#optimization
ϵ=1e-10

@model function fitprob(data,prob)
    # Prior distributions.

    σ ~ InverseGamma(2, 3)
	S0 ~ truncated(Normal(500,20000); lower=1000, upper=20000)
	# μ ~ truncated(Normal(0, 1); lower=0, upper=1)
    # Λ ~ truncated(Normal(1, 3000); lower=S0*μ, upper=S0)
	μp ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ϕ1 ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ϕ2 ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	β1 ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	β2 ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	δ ~ truncated(Normal(0, 1); lower=.1, upper=.2)
	ψ ~ truncated(Normal(0, 1); lower=0.01, upper=0.07)
	ω ~ truncated(Normal(0, 1); lower=1/21, upper=1/4)
	σ2 ~ truncated(Normal(0, 1); lower=0.001, upper=0.03)
	γS ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	γA ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ηS ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	ηA ~ truncated(Normal(0, 1); lower=ϵ, upper=1)
	E0 ~ truncated(Normal(0,300); lower=0, upper=300)
	IA0 ~ truncated(Normal(0,300); lower=0, upper=300)
	P0 ~ truncated(Normal(0,300); lower=0, upper=300)

    # Simulate model.
	p=[Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
	prob = remake(prob; p = p, u0 = X0)
    x = solve(prob,alg_hints=[:stiff]; saveat=1)
	II=x[4,:]
	RR=x[5,:]
	pred=[II RR]
    # Observations.
    for i in 1:length(pred[1,:])
        data[:,i] ~ MvNormal(pred[:,i], σ^2 * I)
    end

    return nothing
end

model = fitprob([C TrueR],prob)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
Nch=5000
chain = sample(model, NUTS(0.65), MCMCSerial(), Nch, 5; progress=false)


display(chain)
##
posterior_samples = sample(chain[[:S0,:μp,:ϕ1,:ϕ2,:β1,:β2,:δ,:ψ,:ω,:σ2,:γS,:γA,:ηS,:ηA,:E0, :IA0, :P0]], Nch; replace=false)

function myshowall(io, x, limit = false)
  println(io, summary(x), ":")
  Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, Array(posterior_samples.value[:,:,1]), false)

Err=zeros(Nch)
for i in 1:Nch
	pp=Array(posterior_samples.value[:,:,1])[i,:]
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = pp[2:14]
	p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
	S0=pp[1]
	IA0=pp[15]
	P0=pp[16]
	E0=pp[17]
	N0=S0+E0+IA0+IS0+R0
	X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]

	prob = ODEProblem(F, X0, tSpan, p)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	Pred=reduce(vcat,sol.u')[:,[4,5]]
	Err[i]=rmsd([C TrueR], Pred)

end

valErr,indErr=findmin(Err)

display(["MinErr",valErr])

myshowall(stdout, Array(posterior_samples.value[:,:,1])[indErr,:], false)

#let's check the prediction
pODE=Array(posterior_samples.value[:,:,1])[indErr,:]
μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA = pODE[2:14]
p = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA]
S0=pODE[1]
IA0, P0, E0=pODE[15:17]
N0=S0+E0+IA0+IS0+R0
X0=[S0,E0,IA0,IS0,R0,P0,D0,N0]

prob = ODEProblem(F, X0, (1,length(CAll)), p)
sol = solve(prob, alg_hints=[:stiff]; saveat=1)
PredODE5=reduce(vcat,sol.u')[Tr+ts+1:Tr+ts+5,[4,5]]
PredODE10=reduce(vcat,sol.u')[Tr+ts+1:Tr+ts+10,[4,5]]
PredODE20=reduce(vcat,sol.u')[Tr+ts+1:Tr+ts+20,[4,5]]
ErrODE5=rmsd([Cp5 R5], PredODE5)
ErrODE10=rmsd([Cp10 R10], PredODE10)
ErrODE20=rmsd([Cp20 R20], PredODE20)

display(["ErrODE5",ErrODE5])
display(["ErrODE10",ErrODE10])
display(["ErrODE20",ErrODE20])

#### FDE

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

Order=ones(8)
#optimization
function loss(args)

	Order[1:6]=args[1:6]
	Order[8]=args[7]
	μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA =args[8:20]
	S0,IA0,P0,E0=args[21:24]

	if 1 != Int64(ceil(maximum(Order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, Order)
		Order[indx]=ones(length(indx))
	end

	par1=[Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]
	N0=S0+E0+IA0+IS0+R0
	x0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
	_, x = FDEsolver(F, tSpan, x0, Order, par1, h=.05, nc=4)
	Pred=x[1:20:end,4:5]
	rmsd([C TrueR], Pred)

end

p_lo_1=vcat(.7*ones(7), p[3:end]./10, S0./10, IA0./10, P0./10, E0./10) #lower bound
p_up_1=vcat(ones(7),p[3:end].*2,S0.*2, IA0.*2, P0.*2, E0.*2) # upper bound
p_vec_1=vcat(.99*ones(7),p[3:end],S0, IA0, P0, E0)

Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res=optimize(loss,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.999), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(#outer_iterations = 1,
						  iterations=100,
						  show_trace=true,
						  show_every=10)
			)


display(Res)
display("RMSD=$(Res.minimum)")
Result=vcat(Optim.minimizer(Res))
# display(Result)

myshowall(stdout, Array(Result), false)

Order[1:6]=Result[1:6]
Order[8]=Result[7]
μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA =Result[8:20]
S0,IA0,P0,E0=Result[21:24]

N0=S0+E0+IA0+IS0+R0
x0=[S0,E0,IA0,IS0,R0,P0,D0,N0]
par2=[Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA]
_, x = FDEsolver(F, [1,length(CAll)], x0, Order, par2, h=.05, nc=4)
PredFDE5=x[20*(Tr+ts+1):20:20*(Tr+ts+5),4:5]
PredFDE10=x[20*(Tr+ts+1):20:20*(Tr+ts+10),4:5]
PredFDE20=x[20*(Tr+ts+1):20:20*(Tr+ts+20),4:5]

ErrFDE5=rmsd([Cp5 R5], PredFDE5)
ErrFDE10=rmsd([Cp10 R10], PredFDE10)
ErrFDE20=rmsd([Cp20 R20], PredFDE20)

display(["ErrFDE5",ErrFDE5])
display(["ErrFDE10",ErrFDE10])
display(["ErrFDE20",ErrFDE20])
