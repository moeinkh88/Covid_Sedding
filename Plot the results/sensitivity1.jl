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
par1= [0.018236598326358076
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
