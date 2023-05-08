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

par1=[9.468e-3,19.995e-3,0.6641890543353863, 9.088630587992902e-7, 0.06344949244030884, 7.694078495487392e-6, 1.6779006579837752e-5, 0.9337865215267538, 0.00814228008865711, 0.9677337168567939, 0.3194429121145711, 0.00023617022276914285, 0.8667486940960136, 0.21882043013719396, 0.9048429015159145]


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
