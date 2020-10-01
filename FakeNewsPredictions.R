##
## Run Models for Cleaned Fake News Data
##

## Libraries
library(tidyverse)
library(caret)

## Read in the data
fakeNews <- read_csv("./CleanFakeNews.csv")

## Split the data back into training and test sets
fn.train <- fakeNews %>% filter(!is.na(isFake))
fn.test <- fakeNews %>% filter(is.na(isFake))

###################
## Preprocessing ##
###################

## None needed because tfidf is already between 0 and 1


#######################
## Predict using RFs ##
#######################

gbmod <- train(form=as.factor(isFake)~.,
               data=fn.train %>% select(-Id, -Set),
               method="xgbTree",
               trControl=trainControl(method="cv",
                                      number=5),
               tuneGrid = expand.grid(nrounds=100, # Boosting Iterations
                 max_depth=3, #Max Tree Depth
                 eta=0.3, #(Shrinkage)
                 gamma=1,
                 colsample_bytree=1,# (Subsample Ratio of Columns)
                 min_child_weight=1,# (Minimum Sum of Instance Weight)
                 subsample=1)# (Subsample Percentage)0)
)

preds <- predict(gbmod, newdata=fn.test)

predframe <- data.frame(id=fn.test$Id, label=preds)

write.csv(x=predframe,file="./fakeNewsPreds.csv", row.names=FALSE)




