# sensivity for paramters obtained from bounded initial fitting
#M1

function rep_num1(PP)
    Λ,μ,μp,ϕ1,ϕ2,β1,β2,δ,ψ,ω,σ,γS,γA,ηS,ηA=PP
    ωe=ψ+μ+ω
    ωia=μ+σ+γA
    ωis=μ+σ+γS
 R = Λ*ω/μ*((β1*δ*ηS)/(μp*ωe*ωis)+(β2*δ)/(ωe*ωis)+(β1*(1-δ)*ηA)/(μp*ωe*ωia)+(β2*(1-δ))/(ωe*ωia))
 return R
end

par1=[9.468e-3
	19.995e-3
	0.15782338058089418
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

df=DataFrame(Parameters=parameters, sensivity=R01, value=par1)

show(IOContext(stdout, :limit=>false), df)

show("R0=$(rep_num1(par1))")
