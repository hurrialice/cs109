---
title: Lending Club
---



## Contents


{:.no_toc}
*  
{: toc}




## Background  


Lending Club is the lending network to register with the Securities and Exchange Commission (SEC). It's mission is to transform the banking system to make credit more affordable and investing more rewarding [1]. The Lending Club platform connects borrowers to investors and facilitates the payment of loans up to \$40,000 [2]. The basic idea is that borrowers apply for a loan on the Lending Club Marketplace. Investors can review the loan applications, along with risk analysis provided by LendingClub, to determine how much of the loan request they are willing to fund. When the loan is funded, the borrower will receive the loan.



Based on the setting of this platform, investors can make wrong decisions in some situations (the borrower received the loan even they don't have ability to return the money). In this case, we want to provide a better model that can predict the probability of fully return.    



Moreover, similar to other loan and investment platforms, Lending Club claims it is an "Equal Housing Lender" which means "the bank makes loans without regard to race, color, religion, national origin, sex, handicap, or familial status." [4] With this in mind, analyzing discrimination cross states is also necessary.



## Statement  


Our goal in this project is:   


- to predict the probability that a loan will be fully returned plus interest rate by the end of the due date when the loan just be proved by lending company


- use the probilities from (1) to calculate ROI


- compare predicted ROI with true ROI to see the benefit of our model (profit)


- use rejected loan data and accepted loan data to find discrimination among states



## Data Sourcing

The data in this project came from 2016 Lending Club web as row represents each loan and column represents features of each loan. We decided to use this dataset since we believed most of the borrowers can return the full loan within two years (2018), and the most recent data set should give us a better model that can reflecte the LendingClub's current lending loan. 


In accepted datasets, they contain information about the loans as well as personal description. For example, loan information contains loan amount, funded amount, interest rate, etc. Personal information contains employ title, length, annual income, purpose, state, etc. The other one is the rejected datasets. Since they only contain a few variables, we will not use it to fit the model. We probably will draw some plots based on rejected and accepted datasets to explore discrimination later. 


## Reference 

[1] Lending Club  
https://www.lendingclub.com/  

[2] Lending Club Personal Loan  
https://www.lendingclub.com/loans/personal-loans    

[3] FDIC Equal Housing Lender  
https://www.fdic.gov/regulations/laws/rules/2000-6000.html










