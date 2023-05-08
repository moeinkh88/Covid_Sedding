# sensivity for paramters obtained from bounded initial fitting, Birth and CR
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
	0.17016670078596785
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
