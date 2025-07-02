call_trino <- function(r,S0,K,T,dt,sigma)
  {
    N <- trunc(T/dt)
    tmp <- 0
    # upward and downward width
    u <- exp(sigma*sqrt(3*dt))-1
    d <- 1/(1+u) - 1
    # risk neutral measure
    q_u <- -sqrt(dt/12)*(r-0.5*sigma*sigma)/sigma+1/6
    q_n <- 2/3
    q_d = 1-q_u-q_n
    for(i in 0:N) # loop from 0 to N
      {
      for(j in 0:N-i){
         # add the pay-off of call option
          tmp<-tmp+max(S0*(1+u)^i*(1+d)^(N-i-j)-K,0)*choose(N,i)*
          choose(N-i,j)*q_u^i*q_n^j*q_d^(N-i-j)
      }
# for avoiding double loop, use the following code
# tmp_seq <- 0:N-i
# tmp_tmp <- (S0*(1+u)^i*(1+d)^(N-i-tmp_seq)-K)*choose(N,i)*
# choose(N-i,tmp_seq)*q_u^i*q_n^tmp_seq*q_d^(N-i-tmp_seq)
# tmp <- tmp + sum(tmp_tmp[tmp_tmp>0])
  }
# discount by the pay-off of safe asset
tmp <- tmp/(1+r*dt)^N
return(c("risk neutral measure q_u"=q_u,"call option price"=tmp))
}



call_trino(0.02,100,80,1,0.02,0.4)


