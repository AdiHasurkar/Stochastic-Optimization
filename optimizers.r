### Gradient descent (GD) given a cost function f and its gradient grad

gd <- function(f, grad, y, X, theta0, npars, ndata, a, niters) {
  theta <- matrix(data=NA, nrow=niters, ncol=npars)
  cost <- vector(mode="numeric", length=niters)
  
  theta[1, ] <- theta0
  cost[1] <- f(y, X, theta0, ndata)
  
  for (i in 2:niters) {
    theta[i, ] <- theta[i-1, ]-a*grad(y, X, theta[i-1, ], ndata)
    cost[i] <- f(y, X, theta[i, ], ndata)
  }
  
  return(list(theta=theta, cost=cost))
}

### Stochastic gradient descent (SGD) given a cost function f and its gradient grad
sgd <- function(f, grad, y, X, theta0, npars, ndata, a, niters, nsubsamples) {
  
  theta <- matrix(data=NA, nrow=niters, ncol=npars)
  cost <- vector(mode="numeric", length=niters)
  
  theta[1, ] <- theta0
  sample_obs <- sample(x = 1:ndata,size = nsubsamples,replace = F)
  cost[1] <- f(y[sample_obs], X[sample_obs,], theta0, nsubsamples)
  
  for (i in 2:niters) {
    
    sample_obs <- sample(x = 1:ndata,size = nsubsamples,replace = F)
    theta[i, ] <- theta[i-1, ]-a*grad(y[sample_obs], X[sample_obs,], theta[i-1, ], nsubsamples)
    cost[i] <- f(y[sample_obs], X[sample_obs,], theta[i, ], nsubsamples)
    
  }
  
  return(list(theta=theta, cost=cost))
}

### Stochastic gradient descent with momentum (MSGD) given a cost function f and its gradient grad


msgd <- function(f, grad, y, X, theta0, npars, ndata, a, niters, nsubsamples, b, m0) {
  
  theta <- matrix(data=NA, nrow=niters, ncol=npars)
  cost <- vector(mode="numeric", length=niters)
  m <- matrix(data=NA, nrow=niters, ncol=npars)
  m[1,] <- m0
  
  theta[1, ] <- theta0
  samp <- sample(x = 1:ndata,size = nsubsamples,replace = F)
  cost[1] <- f(y[samp], X[samp,], theta0, nsubsamples)
  
  for (i in 2:niters) {
    samp <- sample(x = 1:ndata,size = nsubsamples,replace = F)
    
    m[i,] <- b*m[i-1,] + (1-b)*grad(y[samp], X[samp,], theta[i-1, ], nsubsamples)
    
    theta[i, ] <- theta[i-1, ]-a*m[i,]
    
    cost[i] <- f(y[samp], X[samp,], theta[i, ], nsubsamples)
    
  }
  
  return(list(theta=theta, cost=cost))
  
}

## Stochastic gradient descent with Nesterov accelerated gradient (NAGSGD)
nagsgd <- function(f, grad, y, X, theta0, npars, ndata, a, niters, nsubsamples, b, m0) {
  
  theta <- matrix(data=NA, nrow=niters, ncol=npars)
  cost <- vector(mode="numeric", length=niters)
  m <- matrix(data=NA, nrow=niters, ncol=npars)
  m[1,] <- m0
  
  theta[1, ] <- theta0
  samp <- sample(x = 1:ndata,size = nsubsamples,replace = F)
  cost[1] <- f(y[samp], X[samp,], theta0, nsubsamples)
  
  for (i in 2:niters) {
    # cat(i,"\n")
    samp <- sample(x = 1:ndata,size = nsubsamples,replace = F)
    
    m[i,] <- b*m[i-1,] + (1-b)*grad(y[samp], X[samp,], theta[i-1, ]-a*b*m[i-1,], nsubsamples)
    
    theta[i, ] <- theta[i-1, ]-a*m[i,]
    
    cost[i] <- f(y[samp], X[samp,], theta[i, ], nsubsamples)
    
  }
  return(list(theta=theta, cost=cost))
}

### AdaGrad given a cost function f and its gradient grad
adagrad <- function(f, grad, y, X, theta0, npars, ndata, a, niters, nsubsamples, epsilon, G0) {
  
  theta <- matrix(data=NA, nrow=niters, ncol=npars)
  cost <- vector(mode="numeric", length=niters)
  G <- matrix(data=NA, nrow=npars, ncol=npars)
  g <- matrix(data=NA, nrow=npars, ncol=niters)
  sample_obs <-  matrix(data=NA, nrow=nsubsamples, ncol=niters)
  
  theta[1, ] <- theta0
  diag(G) <- G0
  sample_obs[,1] <- sample(x = 1:ndata,size = nsubsamples,replace = F)
  cost[1] <- f(y[sample_obs[,1]], X[sample_obs[,1],], theta0, nsubsamples)
  g[,1] <- t(0)
  
  for (i in 2:niters) {
    
    for (p in 1:npars){
      theta[i,p] <- theta[i-1,p]- (a/sqrt(G[p,p]+epsilon))*g[p,i-1]
    }
    
    sample_obs[,i] <- sample(x = 1:ndata,size = nsubsamples,replace = F)
    g[,i] <- grad(y[sample_obs[,i]], X[sample_obs[,i],], theta[i, ], nsubsamples)
    
    for (j in 1:i){
      for (p in 1:npars){
        G[p,p] <- sum(g[p,j]**2)
      }
    }  
    
    cost[i] <- f(y[sample_obs[,i]], X[sample_obs[,i],], theta[i, ], nsubsamples)
    
  }
  
  return(list(theta = theta,cost = cost))
}

### RMSProp given a cost function f and its gradient grad
rmsprop <- function(f, grad, y, X, theta0, npars, ndata, a, niters, nsubsamples, c, epsilon, v0) 
{
  theta <- matrix(data=NA, nrow=niters, ncol=npars)
  cost <- vector(mode="numeric", length=niters)
  #G <- matrix(data=NA, nrow=npars, ncol=npars)
  #g <- matrix(data=NA, nrow=npars, ncol=niters)
  samp <-  matrix(data=NA, nrow=nsubsamples, ncol=niters)
  g <- vector(mode="numeric", length=npars)
  v <- vector(mode="numeric", length=npars)
  
  theta[1, ] <- theta0
  samp[,1] <- sample(x = 1:ndata,size = nsubsamples,replace = F)
  cost[1] <- f(y[samp[,1]], X[samp[,1],], theta0, nsubsamples)
  g <- grad(y[samp[,1]], X[samp[,1],], theta[1, ], nsubsamples)
  
  v <- v0
  
  for (i in 2:niters) {
    
    for (p in 1:npars){
      v[p] <- c*v[p] + (1-c)*(g[p]**2)
      theta[i,p] <- theta[i-1,p]- (a/sqrt(v[p]+epsilon))*g[p]
    }
    
    samp[,i] <- sample(x = 1:ndata,size = nsubsamples,replace = F)
    
    g <- grad(y[samp[,i]], X[samp[,i],], theta[i, ], nsubsamples)
    
    cost[i] <- f(y[samp[,i]], X[samp[,i],], theta[i, ], nsubsamples)
    
  }
  
  return(list(theta = theta,cost = cost))
  
}

### Adam given a cost function f and its gradient grad
adam <- function(f, grad, y, X, theta0, npars, ndata, a, niters, nsubsamples, b, c, epsilon, m0, v0) 
{
  theta <- matrix(data=NA, nrow=niters, ncol=npars)
  cost <- vector(mode="numeric", length=niters)
  samp <-  matrix(data=NA, nrow=nsubsamples, ncol=niters)
  g <- vector(mode="numeric", length=npars)
  v <- vector(mode="numeric", length=npars)
  v1 <- vector(mode="numeric", length=npars)
  m <- vector(mode="numeric", length=npars)
  m1 <- vector(mode="numeric", length=npars)
  
  theta[1, ] <- theta0
  
  samp[,1] <- sample(x = 1:ndata,size = nsubsamples,replace = F)
  cost[1] <- f(y[samp[,1]], X[samp[,1],], theta0, nsubsamples)
  g <- grad(y[samp[,1]], X[samp[,1],], theta[1, ], nsubsamples)
 
  v <- v0
  m <- m0
  
  for (i in 2:niters) {
    
    for (p in 1:npars){
      m[p] <- b*m[p] + (1-b)*g[p]
      v[p] <- c*v[p] + (1-c)*(g[p]**2)
      m1[p] <- m[p]/(1-b^i)
      v1[p] <- v[p]/(1-c^i)
      theta[i,p] <- theta[i-1,p]- (a/sqrt(v1[p]+epsilon))*m1[p]
    }
    
    samp[,i] <- sample(x = 1:ndata,size = nsubsamples,replace = F)
    
    g <- grad(y[samp[,i]], X[samp[,i],], theta[i, ], nsubsamples)
    
    cost[i] <- f(y[samp[,i]], X[samp[,i],], theta[i, ], nsubsamples)
    
  }
  
  return(list(theta = theta,cost = cost))
  
}
