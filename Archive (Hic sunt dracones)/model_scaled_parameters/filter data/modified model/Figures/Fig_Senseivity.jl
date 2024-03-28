# Do not use this code for figure. try to have all plots in one code
## sensitivity
function rep_num1(PP)
    μ,Λ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA,T=PP[1:16]

	αS, αE, αIA, αIS, αR, αD, αW=PP[17:23]
    ωe=ψ+μ+ω
    ωia=μ+σ+γA
    ωis=μ+σ+γS
 
 R = (Λ / μ) ^ αS * (
    β2 ^ αE * (
        (δ * ω ^ αIS) / (ωis * ωe) + ((1 - δ) * ω ^ αIA) / (ωia * ωe)
    ) + β1 ^ αE * (
        (ηS ^ αW * δ * ω ^ αIS) / (μp ^ αW * ωe * ωis) + 
        (ηA ^ αW * (1 - δ) * ω ^ αIA) / (μp ^ αW * ωe * ωia)
    )
)

 return R
end

Λ=19.995e-3 # birth rate (19.995 births per 1000 people)
μ=9.468e-3 # natural human death rate

par1=[9.468e-3
	19.995e-3
	0.003635875138058216
	8.151561722127852e-7
	0.43942169371267265
	2.804848603101783e-6
	8.863963267287063e-7
	0.4947151645777698
	0.0002071891280094621
	0.5396364150788414
	0.0012347012610145762
	0.0701231213714429
	0.00015520509627699186
	0.006363097612794978
	0.35851366456952516
	0.7621311692485067
	ones(7)]


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

df=DataFrame(Parameters=parameters, sensitivity=R01, value=par1[1:15])

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
# 	title = "(c) parameter values for top 300 fits", titlefont = font(9) , titleloc = :left, color=:white, bar_width = 0.9,marker=(0.2, :black, stroke(0)), ylabel="sign(x) * log(|x| + 1)")
# plboxSen=boxplot(repeat(["μ" "Λ" "μp" "ϕ1" "ϕ2" "β1" "β2" "δ" "ψ" "ω" "σ" "γS" "γA" "ηS" "ηA"],outer=300),sign.(Sens).*log10.(abs.(Sens).+1), legend=:false, outliers=false,
# 		title = "(a) parameter sensitivity for top 300 fits",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)), ylabel="sign(x) * log10(|x| + 1)")
# plboxR0=violin(repeat(["R0"],outer=300),sign.(R0).*log10.(abs.(R0).+1), legend=:false,# outliers=false,
# 			 title = "(b) Density of R0",titlefont = font(10) , titleloc = :left, color=:black)
plboxSen=boxplot(repeat(["μ" "Λ" "μp" "ϕ1" "ϕ2" "β1" "β2" "δ" "ψ" "ω" "σ" "γS" "γA" "ηS" "ηA"],outer=300),Sens, legend=:false, outliers=false,
	title = "(a) parameter sensitivity for top 300 fits",titlefont = font(10) , titleloc = :left, color=:white,marker=(0.2, :black, stroke(0)))
plboxR0=boxplot(repeat(["R0"],outer=300),R0, legend=:false,# outliers=false,
	 title = "(b) Density of R0",titlefont = font(10) , titleloc = :left, color=:white, marker=(0.2, :black, stroke(0)),bar_width = .8, xaxis = ((0, 1), 0:1), yaxis=:log)

L=@layout[b b{0.2w}]
PlotSenR0=plot(plboxSen,plboxR0, layout =L, size=(650,300))#, guidefont=font(8), legendfont=font(8))

savefig(PlotSenR0,"PlotSenR0.svg")
