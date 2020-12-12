

# Downloading the different zip files from my GitHub. The data set is stored in 6 zip files, Flowers1.zip to Flowers6.zip.

for(n in c(1:6)) {

dl <- tempfile()
download.file(paste("https://github.com/cordierpc/FlowersRecognition/raw/main/Flowers", n, ".zip", sep = ""), dl)
unzip(dl)

}

rm(dl, n)




# Loading required libraries.

if(!require(jpeg)) install.packages("jpeg", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(party)) install.packages("party", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")


library("tidyverse")
library("jpeg")
library("stringr")
library("caret")
library("randomForest")
library("glmnet")
library("rpart")
library("dplyr")
library("e1071")
library("party")
library("rpart.plot")



# Creation of local folders to store processed and temporary pictures.

dir.create(paste(getwd(), "/TempPictures", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/TempPictures/daisy", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/TempPictures/dandelion", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/TempPictures/rose", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/TempPictures/sunflower", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/TempPictures/tulip", sep = ""), showWarnings = FALSE)

dir.create(paste(getwd(), "/ProcessedPictures", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/ProcessedPictures/daisy", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/ProcessedPictures/dandelion", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/ProcessedPictures/rose", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/ProcessedPictures/sunflower", sep = ""), showWarnings = FALSE)
dir.create(paste(getwd(), "/ProcessedPictures/tulip", sep = ""), showWarnings = FALSE)

dir.create(paste(getwd(), "/ExamplePictures", sep = ""), showWarnings = FALSE)



# Get all flower pictures from the 5 folders of the dataset, store them in a single dataframe and assign a type for each of them depending of the folder they are located.

flowersFullList <- data.frame(name = list.files(paste(getwd(), "/OriginalPictures/daisy", sep = "")),
                              address = paste(getwd(), "/OriginalPictures/daisy/", sep = ""), flowerType = "daisy")

flowersFullList <- rbind(flowersFullList, 
                         data.frame(name = list.files(paste(getwd(), "/OriginalPictures/dandelion", sep = "")), 
                                    address = paste(getwd(), "/OriginalPictures/dandelion/", sep = ""), flowerType = "dandelion"))

flowersFullList <- rbind(flowersFullList, 
                         data.frame(name = list.files(paste(getwd(), "/OriginalPictures/rose/", sep = "")), 
                                    address = paste(getwd(), "/OriginalPictures/rose/", sep = ""), flowerType = "rose"))

flowersFullList <- rbind(flowersFullList, 
                         data.frame(name = list.files(paste(getwd(), "/OriginalPictures/tulip/", sep = "")), 
                                    address = paste(getwd(), "/OriginalPictures/tulip/", sep = ""), flowerType = "tulip"))

flowersFullList <- rbind(flowersFullList, 
                         data.frame(name = list.files(paste(getwd(), "/OriginalPictures/sunflower/", sep = "")), 
                                    address = paste(getwd(), "/OriginalPictures/sunflower/", sep = ""), flowerType = "sunflower"))


# The original pictures are stored in the folder OriginalPictures, pictures after processing will be stored in folder ProcessedPictures, temporary pictures will be stored in folder TempPictures created before.

flowersFullList <- flowersFullList %>% mutate(source = paste(address, name, sep = "")) %>%
  mutate(original = source,
         temp = str_replace(source, "Original", "Temp"),
         processed = str_replace(source, "Original", "Processed")) %>%
  select(-address, -name)



picProperties <- data.frame(ratio = numeric(), resolution = numeric())

for(n in 1:nrow(flowersFullList)) {
  
  matrixPic <- readJPEG(flowersFullList$original[n])
  
  picProperties <- rbind(picProperties, data.frame(ratio = ncol(matrixPic) / nrow(matrixPic), resolution = ncol(matrixPic) * nrow(matrixPic)))
  
}

picProperties %>% 
  group_by(ratio) %>% 
  summarize(nr = n()) %>%
  ggplot(aes(x = ratio)) + geom_histogram(color = "blue", fill = "white", size = 1) +
  labs(x = "ratio length / height", y = "Nr of pictures", title = "Repartition of the length / height ratio")


picProperties %>% 
  group_by(resolution) %>% 
  summarize(nr = n()) %>%
  ggplot(aes(x = resolution)) + geom_histogram(color = "blue", fill = "white", size = 1) +
  labs(x = "Pictures resolution (in pixels)", y = "Nr of pictures", title = "Repartition of pictures resolutions")



# This function generate a random test set with the same ratio of each flower Type.

CreateUniformPartition <- function(list, ratio, manualSeed) {
  
  set.seed(manualSeed, sample.kind="Rounding")
  
  testIndex <- integer()
  
  distinctFlowerType <- unique(list$flowerType)
  
  for(u in 1:length(distinctFlowerType)) {
    
    firstValue <- min(which(list$flowerType == distinctFlowerType[u])) - 1
    
    fullSamplePerType <- list %>% filter(flowerType == distinctFlowerType[u])
    
    testIndexPerType <- firstValue + sample(nrow(fullSamplePerType), nrow(fullSamplePerType)*ratio, replace = FALSE)
    
    testIndex <- c(testIndex, testIndexPerType)
    
  }
  
  testIndex
  
}


# Make the validation and test sets with 10%.

ratio <- 0.1



# The validation set is the one we will use at the very end to check the algorithm efficiency.

validationIndex <- CreateUniformPartition(flowersFullList, ratio, 1)

trainAndTestSet <- flowersFullList[-validationIndex,]
validationSet <- flowersFullList[validationIndex,]


## Now make the test set with 10% of the remaining flowers.

testIndex <- CreateUniformPartition(trainAndTestSet, ratio, 2)

trainSet <- trainAndTestSet[-testIndex,]
testSet <- trainAndTestSet[testIndex,]

rm(CreateUniformPartition, trainAndTestSet, flowersFullList, testIndex, validationIndex, ratio, picProperties)




# This function returns the number of each pixel of each color ((n+1)^3-colors scale) in the input picture.

ListColorShades <- function(matrixPic, nrLevels) {
  
  # This simple function simplifies the picture color range in n+1 shades of red (0 to n), n+1 shades of green, n+1 shades of blue and returns for each combination the ratio of this color in the picture.
  
  colorMatrix <- matrix(0, nrow = dim(matrixPic)[1], ncol = dim(matrixPic)[2])
  
  colorMatrix <- 100 * round(nrLevels*matrixPic[,,1], 0) + 10 * round(nrLevels*matrixPic[,,2], 0) + round(nrLevels*matrixPic[,,3], 0)
  
  colorList <- data.frame(color = 0)
  
  for(i in 1:nrow(colorMatrix)) {
    
    colorList <- rbind(colorList, data.frame(color = colorMatrix[i,]))
    
  }
  
  # Count the number of non-black pixels.
  
  totalPoints <- colorList %>% filter(color > 0) %>% summarize(nr = n()) %>% pull(nr)
  
  # Compute the ratio of pixels of each color.
  
  colorList %>% filter(color > 0) %>% group_by(color) %>% summarize(nr = round(100 * n() / totalPoints, 0))
}



# This function receives a picture and the number of color levels and returns the ratio of each of the (3(n+1)) colors. 

ComputeThePredictors <- function(matrixPic, nrLevels) {
  
  colorRef <- expand.grid(a = c(0:nrLevels), b = c(0:nrLevels), c = c(0:nrLevels))
  
  colorRef <- colorRef %>% mutate(color = as.numeric(paste(a, b, c, sep = ""))) %>% 
    select(color)
  
  finalColors <- ListColorShades(matrixPic, nrLevels)
  
  finalPred <- colorRef %>% left_join(finalColors, by = c("color")) %>% 
    mutate(nr = ifelse(is.na(nr), 0, nr), color = paste("col", color, sep = ""))
  
  finalPred
  
}



# This function receives a list of pictures and the number of color levels and returns a single line per picture with ((n + 1)^3 columns, each column storing the ratio of this color in this picture. At the end we have the list of pictures and the predictors for each of them.

GetThePredictors <- function(picturesList, nrLevels, isTraining) {
  
  # Create the dataframe to host the results.
  
  colorRef <- expand.grid(a = c(0:nrLevels), b = c(0:nrLevels), c = c(0:nrLevels))
  
  colorRef <- colorRef %>% mutate(color = as.numeric(paste(a, b, c, sep = ""))) %>% 
    select(color)
  
  finalPredictors <- rbind(data.frame(c = "flowerType", nr = 0, stringsAsFactors = FALSE),
                           data.frame(c = "name", nr = 0, stringsAsFactors = FALSE),
                           data.frame(c = paste("col", colorRef$c, sep = ""), nr = 0, stringsAsFactors = FALSE))
  
  finalPredictors <- finalPredictors %>% spread(c, nr) %>% 
    filter(flowerType != "0")
  
  
  # Load each picture, compute the predictors for each picture and add the line to the finalPredictors dataframe hosting predictors of all pictures of the list.
  
  for(i in 1:nrow(picturesList)) {
    
    processedMatrixFlower <- readJPEG(picturesList$source[i])
    
    finalPred <- ComputeThePredictors(processedMatrixFlower, nrLevels)
    
    finalPredictors <- rbind(finalPredictors, data.frame(finalPred %>% 
                                                           mutate(flowerType = picturesList$flowerType[i], name = picturesList$source[i]) %>% 
                                                           spread(color, nr)))
    
  }
  
  
  # Keep colums having at least 1 non null value. Columns corresponding to colors not found in any picture are discarded.
  
  predictorsValues <- finalPredictors %>% select(-name, -flowerType)
  
  
  # If the list of pictures is the training set, it returns all predictors (colors having at least a value) and the flower type, otherwise only the predictors.
  
  if(isTraining) {filteredValues <- data.frame(flowerType = as.factor(finalPredictors$flowerType), predictorsValues[colSums(predictorsValues) > 0])}
  else {filteredValues <- data.frame(predictorsValues) }
  
  filteredValues
  
}  



# Example of predictors computation for a random rose picture.

exampleList <- data.frame(source = paste(getwd(), "/OriginalPictures/rose/18464075576_4e496e7d42_n.jpg", sep = ""), flowerType = "rose", stringsAsFactors = FALSE)

nrLevels <- 2

examplePred <- GetThePredictors(exampleList, nrLevels, TRUE)

as.data.frame(examplePred)



rm(exampleList, nrLevels, examplePred)



trainModels <- c("knn", "rf", "gbm", "glmnet", "rpart", "nnet", "C5.0")

  
compareTrainModels <- data.frame(model = character(), flowerType = character(), name = character(), prediction = character(), nrLevels = numeric())
  

# The training and test sets are processed to get predictors, then each of the 7 models is tested, results are stored in dataframe compareTrainModels. 
#The same operation is repeated with a different number of color levels.
  
  for (nrLevels in 1:6) {
  
  filteredValuesTraining <- GetThePredictors(trainSet, nrLevels, TRUE)
  
  filteredValuesTest <- GetThePredictors(testSet, nrLevels, FALSE)
  
  
# Perform training and predictions for the different models.
  
  for (m in 1:length(trainModels)) {
  
  set.seed(1, sample.kind="Rounding")  
      
  trainingResults <- train(flowerType ~ ., filteredValuesTraining, method = trainModels[m])
  
  prediction <- predict(trainingResults, filteredValuesTest)

  compareTrainModels <- rbind(compareTrainModels, data.frame(model = trainModels[m], name = testSet$source, flowerType = testSet$flowerType, prediction, nrLevels))
  
  }
  }


rm(nrLevels, m, prediction, trainingResults, filteredValuesTraining, filteredValuesTest)




# Results are grouped by flower type and model to get ratio of correct predictions and number of false positive for each model and number of colors.

 results <- compareTrainModels %>%
            mutate(correct = flowerType == prediction) %>% 
            group_by(nrLevels, flowerType, model) %>% 
            summarize(correct = sum(correct), nr = n())
 
 
 falsePositives <- compareTrainModels %>%
                   mutate(correct = flowerType == prediction) %>%
                   filter(correct == FALSE) %>%
                   group_by(nrLevels, prediction, model) %>%
                   summarize(falsePositive = n())
 
 
 finalResults <- results %>% left_join(falsePositives, by = c("model", "nrLevels", "flowerType" = "prediction")) %>%
                         mutate(falsePositive = ifelse(is.na(falsePositive), 0, falsePositive)) %>%
                         mutate(correct = round(100 * correct/nr, 1), 
                                falsePositive = round(100 * falsePositive/nr, 1), 
                                balance = correct - falsePositive) %>%
                         select(- nr)
 
 
 rm(results, falsePositives)

 
 

finalResults %>% ggplot(aes(x = nrLevels)) + ylim(-50, 100) +
                 geom_line(aes(y = correct), color = "green", size = 1) + 
                 geom_line(aes(y = falsePositive), color = "red", size = 1) +
                 geom_line(aes(y = balance), color = "blue", size = 1) +
                 labs(x = "Nr of color levels", y = "Ratio of correct / False positives / Balance", title = "Full overview of performance by model, type of flower and number of color levels") +
                 facet_grid(flowerType~model)




finalResults %>% ggplot(aes(x = nrLevels)) + ylim(-75, 75) +
                 geom_line(aes(y = balance, color = model), size = 1) +
                 labs(x = "Nr of color levels", y = "Balance between accuracy and selectivity", title = "Simplified overview of performance for all models by type of flower and number of color levels") +
                 facet_grid(.~flowerType)




# Get the nr of colors giving the best results and store results with this value in resultsEvolution to compare the performance of the model accross the different steps.

optimalNrColors <- finalResults %>% group_by(nrLevels, flowerType) %>%
  summarize(bestModel = max(balance)) %>%
  group_by(nrLevels) %>%
  summarize(overall = mean(bestModel)) %>%
  arrange(desc(overall)) %>%
  top_n(1) %>%
  select(nrLevels) %>%      # In case of equality, keep the lowest.
  arrange(nrLevels) %>%
  pull(nrLevels)





# Get the models giving the best results for this number of Levels for each type of Flower.

bestModels <- finalResults %>% filter(nrLevels == optimalNrColors) %>%
                               group_by(flowerType) %>%
                               filter(balance == max(balance)) %>%
                               pull(model) %>% 
                               unique()





finalResults %>% filter(nrLevels == optimalNrColors, model %in% bestModels) %>%
                 ggplot(aes(x = flowerType, y = balance, color = model)) + 
                 ylim(0, 80) +
                 labs(x = "Type of flowers", y = "balance accuracy / selectivity", title = "Performance of the best models for the different type of flowers") +
                 geom_point(size = 3)





# Keep only models being the best for at least one type of flower.

resultsEvolution <- finalResults %>% 
  filter(nrLevels == optimalNrColors, model %in% bestModels) %>%
  ungroup() %>%
  select(model, flowerType, balance) %>%
  spread(flowerType, balance) %>%
  mutate(step = "Raw Pictures")


as.data.frame(resultsEvolution[c("step", "model", "daisy", "dandelion", "rose", "sunflower", "tulip")])




resultsEvolutionOverall <- finalResults %>% 
  filter(nrLevels == optimalNrColors, model %in% bestModels) %>%
  group_by(model) %>%
  summarize(overallAccuracy = mean(correct)) %>%
  filter(overallAccuracy == max(overallAccuracy)) %>%
  mutate(step = "Raw Pictures")


as.data.frame(resultsEvolutionOverall[c("step", "model", "overallAccuracy")])


trainModels <- bestModels

rm(bestModels, finalResults)




# This function receives a picture under its matrix form and returns a 2-dimensions matrix of the same height / length with numbers corresponding to the contrast between the pixel and its four neighbour pixels.

ContrastPicture <- function(matrixPic) {
  
  contrastFlower <- matrix(0, nrow = dim(matrixPic)[1], ncol = dim(matrixPic)[2])
  
  # First and last lines and columns are excluded from the computation as they have only 2 or 3 neighbours.
  
  for(i in (2:(nrow(matrixPic)-1))) {
    
    for(j in (2:(ncol(matrixPic)-1))) {
      
      # Compute the distance between a point and its adjacent ones to estimate the local contrast.
      
      contrastFlower[i, j] <- ((matrixPic[i, j, 1] - matrixPic[i+1, j, 1])^2 + (matrixPic[i, j, 2] - matrixPic[i+1, j, 2])^2 + (matrixPic[i, j, 3] - matrixPic[i+1, j, 3])^2)^(1/2) + 
        ((matrixPic[i, j, 1] - matrixPic[i-1, j, 1])^2 + (matrixPic[i, j, 2] - matrixPic[i-1, j, 2])^2 + (matrixPic[i, j, 3] - matrixPic[i-1, j, 3])^2)^(1/2) +
        
        ((matrixPic[i, j, 1] - matrixPic[i, j+1, 1])^2 + (matrixPic[i, j, 2] - matrixPic[i, j+1, 2])^2 + (matrixPic[i, j, 3] - matrixPic[i, j+1, 3])^2)^(1/2) +
        ((matrixPic[i, j, 1] - matrixPic[i, j-1, 1])^2 + (matrixPic[i, j, 2] - matrixPic[i, j-1, 2])^2 + (matrixPic[i, j, 3] - matrixPic[i, j-1, 3])^2)^(1/2)
      
    }
  }
  
  contrastFlower
}




examples <- c("OriginalPictures/sunflower/678714585_addc9aaaef.jpg", 
              "OriginalPictures/tulip/100930342_92e8746431_n.jpg")

for(n in 1:length(examples)) {
  
  pic <- paste(getwd(), "/", examples[n], sep = "")
  
  matrixPic <- readJPEG(pic)
  
  contrastPic <- ContrastPicture(matrixPic)
  
  # To avoid getting in the resulting matrix color range >1, we normalize the resulting matrix.
  
  contrastPic <- contrastPic / max(contrastPic)
  
  writeJPEG(contrastPic, target = paste(getwd(), "/ExamplePictures/BasicContrastPic", n, ".jpg", sep = ""), quality = 1.0, bg = "white")
  
}

rm(pic, matrixPic, contrastPic, n)




# This function receive a matrix form and returns the picture multiplicated by its contrast matrix and normalized.

HighlightHighContrastZones <- function(matrixPic) {
  
  highestContrastFlower <- array(0, dim = dim(matrixPic))
  
  # Get the contrast matrix.
  
  contrastPic <- ContrastPicture(matrixPic)
  
  # Normalization.
  
  highestContrastFlower <- highestContrastFlower / max(contrastPic)
  
  # Multiplicate the contrast matrix by the original picture.
  
  highestContrastFlower[,,1] <- matrixPic[,,1] * contrastPic
  highestContrastFlower[,,2] <- matrixPic[,,2] * contrastPic
  highestContrastFlower[,,3] <- matrixPic[,,3] * contrastPic  
  
  
  highestContrastFlower
  
}




# Application to some examples.

for(n in 1:length(examples)) {
  
  pic <- paste(getwd(), "/", examples[n], sep = "")
  
  matrixPic <- readJPEG(pic)
  
  matrixPic <- HighlightHighContrastZones(matrixPic)
  
  writeJPEG(matrixPic, target = paste(getwd(), "/ExamplePictures/HighestContrastFlower", n, ".jpg", sep = ""), quality = 1.0, bg = "white")
  
}

rm(matrixPic, pic, HighlightHighContrastZones)




# This function receives a matrix form and a quantile (decimal number) and filters from the picture points being in the lowest contrast quantile.

FilterHighestContrastZones <- function(matrixPic, contrastQuantile) {
  
  filteredHighContrastFlower <- array(0, dim = dim(matrixPic))  
  
  contrastScalePic <- ContrastPicture(matrixPic)
  
  contrastTreshold <- quantile(contrastScalePic, prob = contrastQuantile)
  
  contrastMonochrome <- ifelse(contrastScalePic > contrastTreshold, 1, 0) 
  
  filteredHighContrastFlower[,,1] <- matrixPic[,,1] * contrastMonochrome
  filteredHighContrastFlower[,,2] <- matrixPic[,,2] * contrastMonochrome
  filteredHighContrastFlower[,,3] <- matrixPic[,,3] * contrastMonochrome 
  
  filteredHighContrastFlower
  
}




# Application to some examples.

examples <- c("OriginalPictures/daisy/506348009_9ecff8b6ef.jpg",
              "OriginalPictures/tulip/2351637471_5dd34fd3ac_n.jpg")

exampleOfContrastQuantile <- seq(0.2, 0.8, 0.3)


for(n in 1:length(exampleOfContrastQuantile)) {

pic <- paste(getwd(), "/", examples[1], sep = "")

matrixPic <- readJPEG(pic)

highestContrastFlower <- FilterHighestContrastZones(matrixPic, exampleOfContrastQuantile[n])

writeJPEG(highestContrastFlower, target = paste(getwd(), "/ExamplePictures/ExampleContrastQuantile1_", n, ".jpg", sep = ""), quality = 1.0, bg = "white")

}


for(n in 1:length(exampleOfContrastQuantile)) {

pic <- paste(getwd(), "/", examples[2], sep = "")

matrixPic <- readJPEG(pic)

highestContrastFlower <- FilterHighestContrastZones(matrixPic, exampleOfContrastQuantile[n])

writeJPEG(highestContrastFlower, target = paste(getwd(), "/ExamplePictures/ExampleContrastQuantile2_", n, ".jpg", sep = ""), quality = 1.0, bg = "white")

}

rm(highestContrastFlower, matrixPic, pic, n)




# Example of predictors computation for a random rose picture.

exampleList <- data.frame(source = paste(getwd(), "/OriginalPictures/rose/18464075576_4e496e7d42_n.jpg", sep = ""), flowerType = "rose", stringsAsFactors = FALSE)

nrLevels <- 2
contrastQ <- 0.7

matrixPic <- readJPEG(exampleList$source)

matrixPic <- FilterHighestContrastZones(matrixPic, contrastQ)

exampleList$source <- paste(getwd(), "/ExamplePictures/18464075576_4e496e7d42_n_p.jpg", sep = "")

writeJPEG(matrixPic, target = exampleList$source, quality = 1.0, bg = "white")

examplePred <- GetThePredictors(exampleList, nrLevels, TRUE)





as.data.frame(examplePred)



rm(exampleList, nrLevels, contrastQ, matrixPic, examplePred)



# Let's keep the results of the previous training for models and nr of color levels we kept. The original pictures correspond to a processus with a contrast quantile = 0.

compareTrainModels <- compareTrainModels %>% 
  filter(model %in% trainModels & nrLevels == optimalNrColors) %>% 
  mutate(contrastQuantile = 0) %>%
  select(- nrLevels)

# Now use the Temp pictures as source for the training/ prediction.

trainSet <- trainSet %>% mutate(source = temp)
testSet <- testSet %>% mutate(source = temp)


# The processing below apply the contrast filter to the original picture and save the resulting pictures in the Temp folder.

for(contrastQuantile in seq(0.1, 0.9, 0.1)) {

# Processing the training set.

for(n in 1:nrow(trainSet)) {

  matrixPic <- readJPEG(trainSet$original[n])

  matrixPic <- FilterHighestContrastZones(matrixPic, contrastQuantile)
  
  writeJPEG(matrixPic, target = trainSet$temp[n], quality = 1.0, bg = "white")

}


# Processing the test set.

for(n in 1:nrow(testSet)) {

  matrixPic <- readJPEG(testSet$original[n])

  matrixPic <- FilterHighestContrastZones(matrixPic, contrastQuantile) 
 
  writeJPEG(matrixPic, target = testSet$temp[n], quality = 1.0, bg = "white")

}

  

  filteredValuesTraining <- GetThePredictors(trainSet, optimalNrColors, TRUE)
  
  filteredValuesTest <- GetThePredictors(testSet, optimalNrColors, FALSE)
  
  
  for (m in 1:length(trainModels)) {
    
  set.seed(1, sample.kind="Rounding")  
    
  trainingResults <- train(flowerType ~ ., filteredValuesTraining, method = trainModels[m])
  
  prediction <- predict(trainingResults, filteredValuesTest)

  compareTrainModels <- rbind(compareTrainModels, data.frame(model = trainModels[m], name = testSet$source, flowerType = testSet$flowerType, prediction, contrastQuantile))
  
  }

}



rm(trainingResults, prediction, finalPredictorsTest, filteredValuesTest, filteredValuesTraining, matrixPic, contrastQuantile, exampleOfContrastQuantile, examples, m, n)




# Results are grouped by flower type and contrast quantile to get ratio of correct predictions and number of false positive for each model and contrast quantile.

results <- compareTrainModels %>%
           mutate(correct = flowerType == prediction) %>% 
           group_by(flowerType, model, contrastQuantile) %>% 
           summarize(correct = sum(correct), nr = n())


falsePositives <- compareTrainModels %>%
                  mutate(correct = flowerType == prediction) %>%
                  filter(correct == FALSE) %>%
                  group_by(prediction, model, contrastQuantile) %>%
                  summarize(falsePositive = n())


finalResults <- results %>% left_join(falsePositives, by = c("model", "contrastQuantile", "flowerType" = "prediction")) %>%
                        mutate(falsePositive = ifelse(is.na(falsePositive), 0, falsePositive)) %>%
                        mutate(correct = round(100 * correct/nr, 1), 
                               falsePositive = round(100 * falsePositive/nr, 1), 
                               balance = correct - falsePositive) %>%
                        select(- nr)


rm(results, falsePositives, compareTrainModels, m, n, contrastPic)



finalResults %>% ggplot(aes(x = contrastQuantile)) + ylim(0, 100) +
                 geom_line(aes(y = balance, color = model), size = 1) +
                 labs(x = "Contrast quantile", y = "balance accuracy / selectivity", title = "Evolution of performance after contrast processing of pictures with different contrast quantiles") +
                 facet_grid(.~flowerType)


# Get the optimal value of ContrastQuantile.

optimalContrastQuantile <- finalResults %>% group_by(contrastQuantile, model) %>%
                           summarize(avgBalance = mean(balance)) %>%
                           arrange(desc(avgBalance)) %>%
                           ungroup() %>%
                           top_n(1) %>%
                           select(contrastQuantile) %>%
                           arrange(contrastQuantile) %>%        # In case of equality, keep the lowest.
                           top_n(1) %>%
                           pull(contrastQuantile)




resultsEvolution <- finalResults %>% 
  filter(contrastQuantile == optimalContrastQuantile) %>%
  ungroup() %>%
  select(model, flowerType, balance) %>%
  spread(flowerType, balance) %>%
  mutate(step = "Contrast Filter") %>%
  rbind(resultsEvolution)


as.data.frame(resultsEvolution[c("step", "model", "daisy", "dandelion", "rose", "sunflower", "tulip")])




resultsEvolutionOverall <- finalResults %>% 
  filter(contrastQuantile == optimalContrastQuantile) %>%
  group_by(model) %>%
  summarize(overallAccuracy = mean(correct)) %>%
  filter(overallAccuracy == max(overallAccuracy)) %>%
  mutate(step = "Contrast Filter") %>%
  rbind(resultsEvolutionOverall)


as.data.frame(resultsEvolutionOverall[c("step", "model", "overallAccuracy")])




# Processed pictures are to be stored in ProcessedPictures at this stage.

for(n in 1:nrow(trainSet)) {

  matrixPic <- readJPEG(trainSet$original[n])

  matrixPic <- FilterHighestContrastZones(matrixPic, optimalContrastQuantile)
  
  writeJPEG(matrixPic, target = trainSet$processed[n], quality = 1.0, bg = "white")

  }


# Processing the test set.

for(n in 1:nrow(testSet)) {

  matrixPic <- readJPEG(testSet$original[n])

  matrixPic <- FilterHighestContrastZones(matrixPic, optimalContrastQuantile) 
 
  writeJPEG(matrixPic, target = testSet$processed[n], quality = 1.0, bg = "white")

  }

# Now the processed pictures become the source, and target is set back to the temp folder.

rm(matrixPic, n, finalResults)



# This function receives a picture under it matrix form and discard the green or blue pixels with dominance higher than a certain treshold specified as input.

FilterColor <- function(matrixPic, color, coef) {
  
  boolPic <- matrix(0, nrow = dim(matrixPic)[1], ncol = dim(matrixPic)[2])
  filteredSample <- array(0, dim = dim(matrixPic))
  
  # Filter pixels where blue is dominant

  if(color == "green") { boolPic <- ifelse(matrixPic[,,2] > coef * matrixPic[,,1] & matrixPic[,,2] > coef * matrixPic[,,3], 0, 1) }
  if(color == "blue")  { boolPic <- ifelse(matrixPic[,,3] > coef * matrixPic[,,1] & matrixPic[,,3] > coef * matrixPic[,,2], 0, 1) }
  
  
  filteredSample[,,1] <- matrixPic[,,1] * boolPic
  filteredSample[,,2] <- matrixPic[,,2] * boolPic
  filteredSample[,,3] <- matrixPic[,,3] * boolPic
  
  filteredSample
  
}




examples <- c("OriginalPictures/dandelion/7469617666_0e1a014917.jpg", 
              "OriginalPictures/sunflower/3897174387_07aac6bf5f_n.jpg")

exampleOfCoef <- seq(1, 1.5, 0.25)


for(n in 1:length(exampleOfCoef)) {
  
  pic <- paste(getwd(), "/", examples[1], sep = "")
  
  matrixPic <- readJPEG(pic)
  
  matrixPic <- FilterHighestContrastZones(matrixPic, optimalContrastQuantile)
  
  matrixPic <- FilterColor(matrixPic, "green", exampleOfCoef[n])
  
  writeJPEG(matrixPic, target = paste(getwd(), "/ExamplePictures/ExampleGreenFilter", n, ".jpg", sep = ""), quality = 1.0, bg = "white")
  
}


for(n in 1:length(exampleOfCoef)) {
  
  pic <- paste(getwd(), "/", examples[2], sep = "")
  
  matrixPic <- readJPEG(pic)
  
  matrixPic <- FilterHighestContrastZones(matrixPic, optimalContrastQuantile)
  
  matrixPic <- FilterColor(matrixPic, "blue", exampleOfCoef[n])
  
  writeJPEG(matrixPic, target = paste(getwd(), "/ExamplePictures/ExampleBlueFilter", n, ".jpg", sep = ""), quality = 1.0, bg = "white")
  
}

rm(finalResults, pic, matrixPic, n)




# Example of predictors computation for a random rose picture.

exampleList <- data.frame(source = paste(getwd(), "/ExamplePictures/18464075576_4e496e7d42_n_p.jpg", sep = ""), flowerType = "rose", stringsAsFactors = FALSE)

nrLevels <- 2
greenF <- 1.05
blueF <- 1.05

matrixPic <- readJPEG(exampleList$source)

matrixPic <- FilterColor(matrixPic, "green", greenF)

matrixPic <- FilterColor(matrixPic, "blue", blueF)

exampleList$source <- paste(getwd(), "/ExamplePictures/18464075576_4e496e7d42_n_q.jpg", sep = "")

writeJPEG(matrixPic, target = exampleList$source, quality = 1.0, bg = "white")

examplePred <- GetThePredictors(exampleList, nrLevels, TRUE)




as.data.frame(examplePred)




rm(examplePred, exampleList, exampleOfCoef, examples, nrLevels, greenF, blueF, matrixPic)



compareTrainModels <- data.frame(model = character(), flowerType = character(), name = character(), prediction = character(), greenCoef = numeric())

# Processing pictures coming form the contrast filter with the blue filters with different coefficients.

for(blueCoef in seq(1, 1.3, 0.025)) {

# Processing the training set.

for(n in 1:nrow(trainSet)) {

  matrixPic <- readJPEG(trainSet$processed[n])

  matrixPic <- FilterColor(matrixPic, "blue", blueCoef)
  
  writeJPEG(matrixPic, target = trainSet$temp[n], quality = 1.0, bg = "white")

}


# Processing the test set.

for(n in 1:nrow(testSet)) {

  matrixPic <- readJPEG(testSet$processed[n])

  matrixPic <- FilterColor(matrixPic, "blue", blueCoef)

  writeJPEG(matrixPic, target = testSet$temp[n], quality = 1.0, bg = "white")

}

  

  filteredValuesTraining <- GetThePredictors(trainSet, optimalNrColors, TRUE)

  filteredValuesTest <- GetThePredictors(testSet, optimalNrColors, FALSE)



# Train and predict using the different models selected before.

for (m in 1:length(trainModels)) {

set.seed(1, sample.kind="Rounding")  
      
trainingResults <- train(flowerType ~ ., filteredValuesTraining, method = trainModels[m])
  
prediction <- predict(trainingResults, filteredValuesTest)

compareTrainModels <- rbind(compareTrainModels, data.frame(model = trainModels[m], name = testSet$source, flowerType = testSet$flowerType, prediction, blueCoef))
  
}
  
}

rm(prediction, trainingResults, filteredValuesTest, filteredValuesTraining, matrixPic, blueCoef, m, n)




# Results are grouped by flower type and blue filter coefficients to get ratio of correct predictions and number of false positive for each model and blue filter coefficients.

results <- compareTrainModels %>% 
  mutate(correct = flowerType == prediction) %>% 
  group_by(flowerType, model, blueCoef) %>% 
  summarize(correct = sum(correct), nr = n())


falsePositives <- compareTrainModels %>% 
  mutate(correct = flowerType == prediction) %>%
  filter(correct == FALSE) %>%
  group_by(prediction, model, blueCoef) %>%
  summarize(falsePositive = n())


finalResults <- results %>% 
  left_join(falsePositives, by = c("model", "blueCoef", "flowerType" = "prediction")) %>%
  mutate(falsePositive = ifelse(is.na(falsePositive), 0, falsePositive)) %>%
  mutate(correct = round(100 * correct/nr, 1), 
         falsePositive = round(100 * falsePositive/nr, 1), 
         balance = correct - falsePositive) %>%
  select(- nr)


finalResults %>% ggplot(aes(x = blueCoef, color = model)) + ylim(-10, 100) +
  geom_line(aes(y = balance), size = 1) +
  labs(x = "Blue filter coefficient", y = "balance accuracy / selectivity", title = "Performance of the different models for different values of the blue filter coefficient") +
  facet_grid(.~flowerType)


rm(compareTrainModels, results, falsePositives)




# Keep the blue filter coefficient giving the best results.

optimalBlueFilterCoef <- finalResults %>% 
  group_by(blueCoef,model) %>%
  summarize(avgBalance = mean(balance)) %>%
  arrange(desc(avgBalance)) %>%
  ungroup() %>%
  top_n(1) %>%
  select(blueCoef) %>%
  arrange(blueCoef) %>%     # In case of equality, keep the lowest. 
  top_n(1) %>%
  pull(blueCoef)




resultsEvolution <- finalResults %>% 
  filter(blueCoef == optimalBlueFilterCoef) %>%
  ungroup() %>%
  select(model, flowerType, balance) %>%
  spread(flowerType, balance) %>%
  mutate(step = "Blue Filter") %>%
  rbind(resultsEvolution)


as.data.frame(resultsEvolution[c("step", "model", "daisy", "dandelion", "rose", "sunflower", "tulip")])




resultsEvolutionOverall <- finalResults %>% 
  filter(blueCoef == optimalBlueFilterCoef) %>%
  group_by(model) %>%
  summarize(overallAccuracy = mean(correct)) %>%
  filter(overallAccuracy == max(overallAccuracy)) %>%
  mutate(step = "Blue Filter") %>%
  rbind(resultsEvolutionOverall)


as.data.frame(resultsEvolutionOverall[c("step", "model", "overallAccuracy")])



# Processing the processed pictures and replacing them.

for(n in 1:nrow(trainSet)) {

  matrixPic <- readJPEG(trainSet$processed[n])

  matrixPic <- FilterColor(matrixPic, "blue", optimalBlueFilterCoef)
  
  writeJPEG(matrixPic, target = trainSet$processed[n], quality = 1.0, bg = "white")

}


# Processing the test set.

for(n in 1:nrow(testSet)) {

  matrixPic <- readJPEG(testSet$processed[n])

  matrixPic <- FilterColor(matrixPic, "blue", optimalBlueFilterCoef)
 
  writeJPEG(matrixPic, target = testSet$processed[n], quality = 1.0, bg = "white")

}

rm(matrixPic, finalResults, n)




compareTrainModels <- data.frame(model = character(), flowerType = character(), name = character(), prediction = character(), greenCoef = numeric())

# Process the pictures with different levels of green filter coef, store it in the temp folder.

for(greenCoef in seq(1., 1.3, 0.025)) {

# Processing the training set.

for(n in 1:nrow(trainSet)) {

  matrixPic <- readJPEG(trainSet$processed[n])

  matrixPic <- FilterColor(matrixPic, "green", greenCoef)
  
  writeJPEG(matrixPic, target = trainSet$temp[n], quality = 1.0, bg = "white")

}


# Processing the test set.

for(n in 1:nrow(testSet)) {

  matrixPic <- readJPEG(testSet$processed[n])

  matrixPic <- FilterColor(matrixPic, "green", greenCoef)
  
  writeJPEG(matrixPic, target = testSet$temp[n], quality = 1.0, bg = "white")

}

  

  filteredValuesTraining <- GetThePredictors(trainSet, optimalNrColors, TRUE)
  
  filteredValuesTest <- GetThePredictors(testSet, optimalNrColors, FALSE)
  

  
for (m in 1:length(trainModels)) {
    
set.seed(1, sample.kind="Rounding")
  
trainingResults <- train(flowerType ~ ., filteredValuesTraining, method = trainModels[m])
  
prediction <- predict(trainingResults, filteredValuesTest)

compareTrainModels <- rbind(compareTrainModels, data.frame(model = trainModels[m], name = testSet$source, flowerType = testSet$flowerType, prediction, greenCoef))
  
}
  
}


rm(prediction, trainingResults, finalPredictorsTest, filteredValuesTest, filteredValuesTraining, matrixPic, greenCoef, m, n)




# Results are grouped by flower type and green filter coefficients to get ratio of correct predictions and number of false positive for each model and blue filter coefficients.

results <- compareTrainModels %>% 
  mutate(correct = flowerType == prediction) %>% 
  group_by(flowerType, model, greenCoef) %>% 
  summarize(correct = sum(correct), nr = n())


falsePositives <- compareTrainModels %>% 
  mutate(correct = flowerType == prediction) %>%
  filter(correct == FALSE) %>%
  group_by(prediction, model, greenCoef) %>%
  summarize(falsePositive = n())


finalResults <- results %>% 
  left_join(falsePositives, by = c("model", "greenCoef", "flowerType" = "prediction")) %>%
  mutate(falsePositive = ifelse(is.na(falsePositive), 0, falsePositive)) %>%
  mutate(correct = round(100 * correct/nr, 1), 
         falsePositive = round(100 * falsePositive/nr, 1), 
         balance = correct - falsePositive) %>%
  select(- nr)


finalResults %>% ggplot(aes(x = greenCoef, color = model)) + ylim(-5, 100) +
  geom_line(aes(y = balance), size = 1) +
  labs(x = "Green filter coefficient", y = "balance accuracy / selectivity", title = "Performance of the different models for different values of the green filter coefficient") +
  facet_grid(.~flowerType)

rm(results, falsePositives)




# Keep the green filter coefficient giving the best average value for the different flowers.

optimalGreenFilterCoef <- finalResults %>% 
  group_by(greenCoef, model) %>%
  summarize(avgBalance = mean(balance)) %>%
  arrange(desc(avgBalance)) %>%
  ungroup() %>%
  top_n(1) %>%
  select(greenCoef) %>%
  arrange(greenCoef) %>%     # In case of equality, keep the lowest. 
  top_n(1) %>%
  pull(greenCoef)




resultsEvolution <- finalResults %>% 
                    filter(greenCoef == optimalGreenFilterCoef) %>%
                    ungroup() %>%
                    select(model, flowerType, balance) %>%
                    spread(flowerType, balance) %>%
                    mutate(step = "Green Filter") %>%
                    rbind(resultsEvolution)


as.data.frame(resultsEvolution[c("step", "model", "daisy", "dandelion", "rose", "sunflower", "tulip")])




resultsEvolutionOverall <- finalResults %>% 
                           filter(greenCoef == optimalGreenFilterCoef) %>%
                           group_by(model) %>%
                           summarize(overallAccuracy = mean(correct)) %>%
                           filter(overallAccuracy == max(overallAccuracy)) %>%
                           mutate(step = "Green Filter") %>%
                           rbind(resultsEvolutionOverall)

as.data.frame(resultsEvolutionOverall[c("step", "model", "overallAccuracy")])





resultsEvolution %>% filter(step == "Green Filter") %>%
                     select(-step) %>% 
                     gather(flowerType, balance, -model) %>% 
                     ggplot(aes(x = flowerType, y = balance, color = model)) +
                     labs(x = "Flower Types", y = "Balance accuracy / selectivity", title = "Performance of the different models for the different type of flowers") +
                     ylim(0, 80) +
                     geom_point(size = 3)




# We keep results from the previous stage corresponding to the optimal settings of brightness and saturation.

compareTrainModels <- compareTrainModels %>% 
                      filter(greenCoef == optimalGreenFilterCoef) %>%
                      select(-greenCoef)

                      
spreadTrainModel <- spread(compareTrainModels, model, prediction)

spreadTrainModel <- spreadTrainModel %>% select(-name)




# We now build a tree with the package cTree using where each prediction of a model is a predictor.

cTreeTree <- ctree(flowerType ~ ., data = spreadTrainModel)

plot(cTreeTree)




cTreePrediction <- predict(cTreeTree, spreadTrainModel)

cTreeResults <- data.frame(flowerType = spreadTrainModel$flowerType, prediction = cTreePrediction, stringsAsFactors = FALSE)


results <- cTreeResults %>% 
           mutate(correct = flowerType == prediction) %>% 
           group_by(flowerType) %>% 
           summarize(correct = sum(correct), nr = n())


falsePositives <- cTreeResults %>% 
                  mutate(correct = flowerType == prediction) %>%
                  filter(correct == FALSE) %>%
                  group_by(prediction) %>%
                  summarize(falsePositive = n())


finalResults <- results %>% 
                left_join(falsePositives, by = c("flowerType" = "prediction")) %>%
                mutate(falsePositive = ifelse(is.na(falsePositive), 0, falsePositive)) %>%
                mutate(correct = round(100 * correct/nr, 1), 
                       falsePositive = round(100 * falsePositive/nr, 1), 
                       balance = correct - falsePositive) %>%
                select(- nr)



cTreeScores <- finalResults %>%
               ungroup() %>%
               select(flowerType, balance) %>%
               spread(flowerType, balance)





resultsEvolution <- cTreeScores %>% 
                    mutate(model = "cTree", step = "Decision Tree") %>%
                    rbind(resultsEvolution)


as.data.frame(resultsEvolution[c("step", "model", "daisy", "dandelion", "rose", "sunflower", "tulip")])




resultsEvolutionOverall <- finalResults %>% 
  summarize(overallAccuracy = mean(correct)) %>%
  mutate(model = "cTree", step = "Decision Tree") %>%
  rbind(resultsEvolutionOverall)


as.data.frame(resultsEvolutionOverall[c("step", "model", "overallAccuracy")])


rm(finalResults, falsePositives, results, cTreePrediction)




# We now build a tree with the package rpart using where each prediction of a model is a predictor.

rpartTree <- rpart(flowerType ~ ., data = spreadTrainModel)

rpart.plot(rpartTree)




rpartPrediction <- predict(rpartTree, spreadTrainModel)

# rpart doesn't give directly a prediction but gives a probability per flower type.
rpartPrediction <- colnames(rpartPrediction)[max.col(rpartPrediction)]

rpartResults <- data.frame(flowerType = spreadTrainModel$flowerType, prediction = rpartPrediction, stringsAsFactors = FALSE)


results <- rpartResults %>% 
  mutate(correct = flowerType == prediction) %>% 
  group_by(flowerType) %>% 
  summarize(correct = sum(correct), nr = n())


falsePositives <- rpartResults %>% 
  mutate(correct = flowerType == prediction) %>%
  filter(correct == FALSE) %>%
  group_by(prediction) %>%
  summarize(falsePositive = n())


finalResults <- results %>% 
  left_join(falsePositives, by = c("flowerType" = "prediction")) %>%
  mutate(falsePositive = ifelse(is.na(falsePositive), 0, falsePositive)) %>%
  mutate(correct = round(100 * correct/nr, 1), 
         falsePositive = round(100 * falsePositive/nr, 1), 
         balance = correct - falsePositive) %>%
  select(- nr)


rpartScores <- finalResults %>%
  ungroup() %>%
  select(flowerType, balance) %>%
  spread(flowerType, balance)



rm(falsePositives, results, rpartPrediction, rpartResults)




resultsEvolution <- rpartScores %>% 
  mutate(model = "rpart", step = "Decision Tree") %>%
  rbind(resultsEvolution)


as.data.frame(resultsEvolution[c("step", "model", "daisy", "dandelion", "rose", "sunflower", "tulip")])



resultsEvolutionOverall <- finalResults %>% 
  summarize(overallAccuracy = mean(correct)) %>%
  mutate(model = "rpart", step = "Decision Tree") %>%
  rbind(resultsEvolutionOverall)


as.data.frame(resultsEvolutionOverall[c("step", "model", "overallAccuracy")])


rm(finalResults)




rpartAccuracy <- resultsEvolutionOverall %>% 
  filter(step == "Decision Tree", model == "rpart") %>% 
  pull(overallAccuracy)

cTreeAccuracy <- resultsEvolutionOverall %>% 
  filter(step == "Decision Tree", model == "cTree") %>% 
  pull(overallAccuracy)





if(cTreeAccuracy > rpartAccuracy)           # Select the tree giving the best results.
{bestTree <- cTreeTree} else 
{bestTree <- rpartTree}


rm(rpartScores, cTreeScores, cTreeResults, compareTrainModels, spreadTrainModel, cTreeTree, rpartTree)




# training and test data sets are grouped in a single one that will become the final training set. Pictures are processed with the different steps using the optimal parameters.

compareTrainModels <- data.frame(model = character(), name = character(), flowerType = character(), prediction = character())

uniqueTrainingSet <- rbind(trainSet, testSet)


# Processing of training pictures with optimal coefficients found earlier.

for(n in 1:nrow(uniqueTrainingSet)) {
  
  matrixPic <- readJPEG(uniqueTrainingSet$original[n])
  
  matrixPic <- FilterHighestContrastZones(matrixPic, optimalContrastQuantile)
  
  matrixPic <- FilterColor(matrixPic, "blue", optimalBlueFilterCoef) 
  
  matrixPic <- FilterColor(matrixPic, "green", optimalGreenFilterCoef)
  
  writeJPEG(matrixPic, target = uniqueTrainingSet$processed[n], quality = 1.0, bg = "white")
  
}


# Processing the validation set pictures.

for(n in 1:nrow(validationSet)) {
  
  matrixPic <- readJPEG(validationSet$original[n])
  
  matrixPic <- FilterHighestContrastZones(matrixPic, optimalContrastQuantile)
  
  matrixPic <- FilterColor(matrixPic, "blue", optimalBlueFilterCoef) 
  
  matrixPic <- FilterColor(matrixPic, "green", optimalGreenFilterCoef)
  
  writeJPEG(matrixPic, target = validationSet$processed[n], quality = 1.0, bg = "white")
  
}

# Now the processed pictures should be used

uniqueTrainingSet <- uniqueTrainingSet %>% 
  mutate(source = str_replace(source, "Original", "Processed"))

validationSet <- validationSet %>% 
  mutate(source = str_replace(source, "Original", "Processed"))



filteredValuesTraining <- GetThePredictors(uniqueTrainingSet, optimalNrColors, TRUE)

filteredValuesValidation <- GetThePredictors(validationSet, optimalNrColors, FALSE)


for (m in 1:length(trainModels)) { 
  
  set.seed(1, sample.kind="Rounding")
  
  trainingResults <- train(flowerType ~ ., filteredValuesTraining, method = trainModels[m])
  
  prediction <- predict(trainingResults, filteredValuesValidation)
  
  compareTrainModels <- rbind(compareTrainModels, data.frame(model = trainModels[m], name = validationSet$source, flowerType = validationSet$flowerType, prediction))
  
}



# Spreading the results to make them ready for the tree.

spreadValidation <- spread(compareTrainModels, model, prediction)

spreadValidationLight <- spreadValidation %>% select(-name, -flowerType)


# Use the best tree selected earlier, if cTree, the prediction is direct, if rpart, result has to be converted.

if(cTreeAccuracy > rpartAccuracy)           # Select the tree giving the best results.
{treePrediction <- predict(bestTree, spreadValidationLight) } else 
{treePrediction <- predict(bestTree, spreadValidationLight)
treePrediction <- colnames(treePrediction)[max.col(treePrediction)]} #if the best is rpart, a line need to be added to transform the prediction from a probability to a factor.


validationResults <- data.frame(name = spreadValidation$name, flowerType = spreadValidation$flowerType, prediction = treePrediction, stringsAsFactors = FALSE)

validationFinalResults <- validationResults %>%
  mutate(correct = flowerType == prediction) %>%
  group_by(flowerType) %>%
  summarise(accuracy = mean(correct))


validationResultsOverview <- spread(validationFinalResults, flowerType, accuracy)

rm(trainSet, matrixPic, testSet, n, prediction, trainingResults, compareTrainModels)

as.data.frame(validationResultsOverview[c("daisy", "dandelion", "rose", "sunflower", "tulip")])



wrongPredAll <- validationResults %>% 
  filter(prediction != flowerType) %>%
  mutate(name = str_replace(name, "Processed", "Original")) %>%
  unique()



wrongPred <- wrongPredAll %>% 
  filter(prediction == "daisy" & flowerType != "daisy") %>%
  select(name) %>%
  top_n(3)