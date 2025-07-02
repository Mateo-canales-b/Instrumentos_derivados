black_scholes_1 <- function(S,K,r,sigma,T){
  d1 <- (log(S/K) + (r+sigma^2/2)*T)/(sigma*sqrt(T))
  d2 <- d1 - sigma*sqrt(T)
  C0 <- S*pnorm(d1) - exp(-r*T)*K*pnorm(d2)
  return(c("Call option price"=C0))
  }

black_scholes_2 <- function(S,K,r,sigma,T){
  d1 <- (log(S/K) + (r+sigma^2/2)*T)/(sigma*sqrt(T))
  d2 <- d1 - sigma*sqrt(T)
  C0 <- S*pnorm(d1) - exp(-r*T)*K*pnorm(d2)
  P0 <- C0 - S + exp(-r*T)*K
  return(c("call option price"=C0, "put option price"=P0))
}



# The function calculating the error, i.e.
# difference, between the given market price
# and the price of Black-Scholes formula
err <- function(S,K,r,sigma,T,MktPrice){
   tmp <- abs(MktPrice - black_scholes_1(S,K,r,sigma,T))
   return(tmp)
}
# minimization of err() by using optimize()
results <- optimize(err,interval=c(0,5),maximum=FALSE,MktPrice=8.43,S=100,K=100,r=0.01,T=1)

# Implied volatility
results$minimum

