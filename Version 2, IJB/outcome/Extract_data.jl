# extract 300 best parameters from output1.txt

using Statistics
using CSV, DataFrames
using Interpolations,LinearAlgebra
using Optim, FdeSolver
using SpecialFunctions, StatsBase, Random, DifferentialEquations
using Plots

# Dataset

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

IS0=17;R0=0;RT0=0;D0=0;
Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate
tSpan=(1,length(C))

# Define the equation


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


# Open the file
AA=readlines("output1.txt")

BB=map(x -> parse.(Float64, split(x)), AA)

Err=zeros(length(BB))
for ii in 1:length(BB)
	
    μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T = BB[ii][2:15]
	p1 = [Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ2,γS,γA,ηS,ηA,T]
	S0=BB[ii][1]
	IA0=BB[ii][15]
	P0=BB[ii][16]
	E0=BB[ii][17]
	
	X0=[S0,E0,IA0,IS0,R0,RT0,P0,D0]

	prob = ODEProblem(F, X0, tSpan, p1)
	sol = solve(prob,alg_hints=[:stiff], abstol = 1e-12, reltol = 1e-12; saveat=1)
	II=sol[4,:] .+ T.*sol[3,:]
	RR=sol[6,:]
	DD=sol[8,:]
	Err[ii]=rmsd([C TrueR TrueD], [II RR DD])
end
Err=replace!(Err, 0=>Inf) # filter inacceptable results

#plot the best

valErr,indErr=findmin(Err)
display(["MinErr",valErr])
function myshowall(io, x, limit = false)
	println(io, summary(x), ":")
	Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, Err[sortperm(Err)], false)


Ind=sortperm(Err)

Candidate=BB[Ind[1:300]]

# using CSV, Tables
#
# CSV.write("Candidate500.csv",  Tables.table(Candidate), writeheader=false)

using DelimitedFiles

writedlm( "output1.csv",  Candidate, ',')
