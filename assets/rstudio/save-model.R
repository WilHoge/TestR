install.packages("pmml")
install.packages("nnet")

library(nnet)
library(pmml)

ird <- data.frame(rbind(iris3[,,1], iris3[,,2], iris3[,,3]),
                  species = factor(c(rep("s",50), rep("c", 50), rep("v", 50))))
samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))
ir.nn2 <- nnet(species ~ ., data = ird, subset = samp, size = 2, rang = 0.1,
               decay = 5e-4, maxit = 200)

pmmlmodel <- pmml(ir.nn2, model.name = "IrisNet_Model", namespace = "4_3")
saveXML(pmmlmodel,file = "IrisNet.xml")

