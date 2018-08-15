Q3\_Answer
================
dan

Big Data Analytics Assignment (Text Analytics, Natural Language Processing & Sentiment Analytics)
=================================================================================================

Question 3: Construction of Representation Vectors for Documents using word2vec Approach.
-----------------------------------------------------------------------------------------

``` r
# Question 3 ##########
library(tm)
library(wordVectors)
library(e1071)
library(readr)
```

``` r
# Q3a ##########
nVec <- 100
posText <- readLines("rt-polarity.pos")
negText <- readLines("rt-polarity.neg")
allText <- c(posText, negText)
corp <- VCorpus(VectorSource(allText))
UnigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 1), paste0, collapse = " "), use.names = FALSE)
# Remove Punctuation
corp <- tm_map(corp, removePunctuation)
# Remove Extra Spaces
corp <- tm_map(corp, stripWhitespace)
# Compute the tf-idf for each word
tdm <- TermDocumentMatrix(corp, control=list(weighting=weightTfIdf, tokenize=UnigramTokenizer))
```

    ## Warning in weighting(x): empty document(s): 7918

``` r
tfidf <- as.matrix(tdm)
# combine all reviews into single file
doc <- c()
for (i in 1:length(corp)) {
    doc <- c(doc, content(corp[[i]]))
}
cat(UnigramTokenizer(doc), file="reviews.txt")
if (!file.exists("reviews_vectors.bin")) {
    model <- train_word2vec("reviews.txt",
                            "reviews_vectors.bin",
                            vectors=nVec,
                            threads=4,
                            window=10,
                            iter=10,
                            negative_samples=0)
} else model <- read.vectors("reviews_vectors.bin")
```

    ## Filename ends with .bin, so reading in binary format

    ## Reading a word2vec binary file of 4497 rows and 100 columns

``` r
write.txt.word2vec <- function(model,filename) {
    filehandle <- file(filename,"wb")
    dim <- dim(model)
    writeChar(as.character(dim[1]),filehandle,eos=NULL)
    writeChar(" ",filehandle,eos=NULL)
    writeChar(as.character(dim[2]),filehandle,eos=NULL)
    writeChar("\n",filehandle,eos=NULL)
    names <- rownames(model)
    # I just store the rownames outside the loop, here.
    i <- 1
    names <- rownames(model)
    silent <- apply(model,1,function(row) {
        # EOS must be null for this to work properly, because, ridiculously,
        # 'eos=NULL' is the command that tells R *not* to insert a null string
        # after a character.
        writeChar(paste0(names[i]," "),filehandle,eos=NULL)
        text <- paste(as.character(row),collapse=" ")
        writeChar(paste(text,"\n"),filehandle,eos=NULL)
        i <<- i+1
    })
    close(filehandle)
}
bin_model <- wordVectors::read.binary.vectors("reviews_vectors.bin")
```

    ## Reading a word2vec binary file of 4497 rows and 100 columns

``` r
write.txt.word2vec(bin_model, "reviews_model.txt")
vector_set <- read.table("reviews_model.txt", skip=1, stringsAsFactor=FALSE)
rownames(vector_set) <- vector_set$V1
vector_set <- vector_set[,-1]
names(vector_set) <- paste0("V",1:nVec)

sen <- NULL
class <- NULL
for (j in 1:length(doc)) {
    x <- UnigramTokenizer(doc[j])
    vec_rep <- vector_set[x,]
    if (! all(is.na(vec_rep))){
        sen <- rbind(sen, c(sapply(seq(ncol(vec_rep)), function(i) {
            min(vec_rep[,i], na.rm=TRUE)}), 
            sapply(seq(ncol(vec_rep)), function(i) {
                max(vec_rep[,i], na.rm=TRUE)})))
        
        if (j <= length(posText)) {
            class <- c(class,1)
        } else {
            class <- c(class,0)
        }
    }
}
class <- factor(class)
```

``` r
# Gaussian Kernel
svm.model <- svm(x=sen, y=class)
pred <- predict(svm.model, sen)
table(class, pred)
```

    ##      pred
    ## class    0    1
    ##     0 4744  579
    ##     1  726 4601

``` r
mean(class==pred)
```

    ## [1] 0.8774648

``` r
# Polynomial Kernel
svm.model <- svm(x=sen, y=class, kernel="polynomial")
pred <- predict(svm.model, sen)
table(class, pred)
```

    ##      pred
    ## class    0    1
    ##     0 5078  245
    ##     1  245 5082

``` r
mean(class==pred)
```

    ## [1] 0.9539906

``` r
# Q3b ##########
q3b <- function(n){
    nVec <- n
    bin.name <- paste0("reviews_vectors",n,".bin")
    txt.name <- paste0("reviews_model",n,".txt")
    
    if (!file.exists(bin.name)) {
        model <- train_word2vec("reviews.txt",
                                bin.name,
                                vectors=nVec,
                                threads=4,
                                window=10,
                                iter=10,
                                negative_samples=0)
    } else model <- read.vectors(bin.name)
    bin_model <- wordVectors::read.binary.vectors(bin.name)
    write.txt.word2vec(bin_model, txt.name)
    vector_set <- read.table(txt.name, skip=1, stringsAsFactor=FALSE)
    rownames(vector_set) <- vector_set$V1
    vector_set <- vector_set[,-1]
    names(vector_set) <- paste0("V",1:nVec)
    
    sen <- NULL
    class <- NULL
    for (j in 1:length(doc)) {
        x <- UnigramTokenizer(doc[j])
        vec_rep <- vector_set[x,]
        if (! all(is.na(vec_rep))){
            sen <- rbind(sen, c(sapply(seq(ncol(vec_rep)), function(i) {
                min(vec_rep[,i], na.rm=TRUE)}), 
                sapply(seq(ncol(vec_rep)), function(i) {
                    max(vec_rep[,i], na.rm=TRUE)})))
        
            if (j <= length(posText)) {
                class <- c(class,1)
            } else {
                class <- c(class,0)
            }
        }
    }
    class <- factor(class)
    
    svm.model <- svm(x=sen, y=class)
    pred <- predict(svm.model, sen)
    return(mean(class==pred))
}

q3b.result <- data.frame(nVec=c(50,100,200))
accuracy.list <- list()
for (i in 1:length(q3b.result$nVec)){
    accuracy.list[[i]] <- q3b(q3b.result$nVec[i])
}
```

    ## Filename ends with .bin, so reading in binary format

    ## Reading a word2vec binary file of 4497 rows and 50 columns
    ## Reading a word2vec binary file of 4497 rows and 50 columns

    ## Filename ends with .bin, so reading in binary format

    ## Reading a word2vec binary file of 4497 rows and 100 columns
    ## Reading a word2vec binary file of 4497 rows and 100 columns

    ## Filename ends with .bin, so reading in binary format

    ## Reading a word2vec binary file of 4497 rows and 200 columns
    ## Reading a word2vec binary file of 4497 rows and 200 columns

``` r
q3b.result$accuracy <- unlist(accuracy.list)
```

``` r
q3b.result
```

    ##   nVec  accuracy
    ## 1   50 0.8584038
    ## 2  100 0.8769953
    ## 3  200 0.8969014

``` r
# Q3c ##########
nVec <- 200

bin.name <- paste0("text8.bin")
txt.name <- paste0("reviews_model.txt")

model <- read.vectors(bin.name)
```

    ## Filename ends with .bin, so reading in binary format

    ## Reading a word2vec binary file of 71290 rows and 200 columns

``` r
bin_model <- wordVectors::read.binary.vectors(bin.name)
```

    ## Reading a word2vec binary file of 71290 rows and 200 columns

``` r
write.txt.word2vec(bin_model, txt.name)
vector_set <- read.table(txt.name, skip=1, stringsAsFactor=FALSE)
rownames(vector_set) <- vector_set$V1
vector_set <- vector_set[,-1]
names(vector_set) <- paste0("V",1:nVec)

sen <- NULL
class <- NULL
for (j in 1:length(doc)) {
    x <- UnigramTokenizer(doc[j])
    vec_rep <- vector_set[x,]
    if (! all(is.na(vec_rep))){
        sen <- rbind(sen, c(sapply(seq(ncol(vec_rep)), function(i) {
            min(vec_rep[,i], na.rm=TRUE)}), 
            sapply(seq(ncol(vec_rep)), function(i) {
                max(vec_rep[,i], na.rm=TRUE)})))
        
        if (j <= length(posText)) {
            class <- c(class,1)
        } else {
            class <- c(class,0)
        }
    }
}
class <- factor(class)
```

``` r
# Gaussian Kernel
svm.model <- svm(x=sen, y=class)
pred <- predict(svm.model, sen)
table(class, pred)
```

    ##      pred
    ## class    0    1
    ##     0 4782  544
    ##     1  635 4693

``` r
mean(class==pred)
```

    ## [1] 0.8893373

``` r
# Polynomial Kernel
svm.model <- svm(x=sen, y=class, kernel="polynomial")
pred <- predict(svm.model, sen)
table(class, pred)
```

    ##      pred
    ## class    0    1
    ##     0 5297   29
    ##     1  175 5153

``` r
mean(class==pred)
```

    ## [1] 0.9808523
