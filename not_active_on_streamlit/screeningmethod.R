# https://www.who.int/docs/default-source/coronaviruse/act-accelerator/covax/screening-method-r-script.zip?Status=Master&sfvrsn=d7d1f3e8_5
# cited in file:///C:/Users/rcxsm/Downloads/WHO-2019-nCoV-vaccine-effectiveness-variants-2021.1-eng.pdf

######################################################################################################
#### Screening Method: Calculating a Crude Vaccine Effectiveness (VE) and 95% Confidence Limits   ####
#### References: Orenstein et al. Field evaluation of vaccine efficacy.                           ####
####             https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2536484/                            ####
####             Farrington, C. Estimation of vaccine effectiveness using the screening method.   ####
####             https://doi.org/10.1093/ije/22.4.742                                             ####
######################################################################################################

######################################################################################################
#### Data needed:                                                                                 ####
####             Number of total cases (vaccinated and unvaccinated)                             #### 
####             Number of vaccinated cases                                                       ####
####             Percent of the population vaccinated (as a decimal, example 80%=0.8)             #### 
######################################################################################################

getwd()

######################################################################################################
#### Data Entry NL okt without kids
cases<- 2061 # integer ((47498/123.65333) + (56063/33.42667)) # 104 #103561 #replace XXX with total number of cases
vaxcases<- 384 #integer(47498/123.65333) #47498 #replace XXX with number of vaccinated cases
perpopvax<- 12365333 / (12365333+ 3342667)  #0.7872 #replace XXX with percent of population vaccinated (as a decimal)
######################################################################################################
cases
vaxcases
perpopvax
#install packages: tidyverse#
#install.packages("tidyverse")
library(tidyverse)


# Read in input data containing four columns: cohort case vac ppv
one <- data.frame(cohort=c("group"),
                  case=c(cases), 
			vac =c(vaxcases),
			ppv =c(perpopvax))

# Restructure dataset to fit model
two <- one %>%
  group_by(cohort,case,vac,ppv) %>%
  expand(count=1:case) %>%
  mutate(y=if_else(count<=vac, 1, 0)) %>%
  mutate(pcv=vac/case, logit_ppv=log(ppv/(1-ppv))) %>%
  select(-count)

# Fit logistic regression 
mylogit <- glm(y ~ 1+offset(logit_ppv), data=two, family="binomial")
mylogit

# calculate VE estimates and confidence interval
results <- (1-c(exp(coef(mylogit)), exp(confint(mylogit))))[c(1,3,2)]
names(results)<-c("VE estimate", "Lower CI", "Higher CI")
results