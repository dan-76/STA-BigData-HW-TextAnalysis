Q2\_Answer
================
dan

Big Data Analytics Assignment (Text Analytics, Natural Language Processing & Sentiment Analytics)
=================================================================================================

Question 2: Construct the Data Set for Sentiment Analysis of Movie Reviews by tidytext Approach.
------------------------------------------------------------------------------------------------

data source: <http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>

*file downloaded and unzipped in seperate locatoin with symbolic link in folder named aclImdb*

``` r
# Question 2 ##########
library(tidytext)
library(tidyverse)
library(purrr)
library(readr)
```

``` r
# Q2a ##########
#categorize words as very negative to very positive and add some movie-specific words
neg_extra <- c("second-rate", "moronic", "third-rate",
              "flawed", "juvenile", "boring", "distasteful", "ordinary",
              "disgusting", "senseless", "static", "brutal", "confused",
              "disappointing", "bloody", "silly", "tired", "predictable",
              "stupid", "uninteresting", "trite", "uneven", "outdated",
              "dreadful", "bland")
neu_extra <- c("suspenseful","low-budget","dramatic","high-charged", "sentimental",
              "fantasy","slow","romantic","satirical","fast-moving", "oddball",
              "picaresque","big-budget","wacky")
pos_extra <- c("first-rate", "insightful", "clever", "charming",
              "intriguing", "comical", "charismatic", "enjoyable", "absorbing",
              "sensitive", "powerful", "pleasant", "surprising",
              "thought-provoking", "imaginative", "unpretentious")
vPos_extra <- c("uproarious", "riveting", "fascinating", "dazzling", "legendary")

# Q2b ##########
afinn <- get_sentiments("afinn")
 
# Q2c ##########
afinn_vNeg <- afinn %>%
    filter(score==-5 | score==-4) %>%
    select(word) %>%
    bind_rows()
afinn_neg <- afinn %>%
    filter(score==-3 | score==-2 | score==-1) %>%
    select(word) %>%
    bind_rows(list(word=neg_extra))
afinn_neu <- afinn %>%
    filter(score==0) %>%
    select(word) %>%
    bind_rows(list(word=neu_extra))
afinn_pos <- afinn %>%
    filter(score==3 | score==2 | score==1) %>%
    select(word) %>%
    bind_rows(list(word=pos_extra))
afinn_vPos <- afinn %>%
    filter(score==5 | score==4) %>%
    select(word) %>%
    bind_rows(list(word=vPos_extra))

# Q2di ##########
file.path <- './aclImdb/train/pos'
reviews <- as.data.frame(list.files(file.path),stringsAsFactors=F)
colnames(reviews) <- c("file")
reviews$file <- paste(file.path,reviews$file,sep="/")

# Q2dii ##########
reviews$text <- unlist(map(reviews$file, read_lines))

# Q2diii ##########
reviews$id <- row_number(rownames(reviews))

# Q2div ##########
reviews <- reviews %>%
    select(id,text)

# Q2dv ##########
reviews <- reviews %>%
    unnest_tokens(word,text)

# Q2dvi ##########
reviews <- reviews %>%
    anti_join(stop_words, by=c("word"="word"))

# Q2e ##########
vNeg_count <- afinn_vNeg %>%
    inner_join(reviews, by=c("word"="word")) %>%
    group_by(id, word) %>%
    summarise(n=n()) %>%
    mutate(status="vNeg")
neg_count <- afinn_neg %>%
    inner_join(reviews, by=c("word"="word")) %>%
    group_by(id, word) %>%
    summarise(n=n()) %>%
    mutate(status="neg")
neu_count <- afinn_neu %>%
    inner_join(reviews, by=c("word"="word")) %>%
    group_by(id, word) %>%
    summarise(n=n()) %>%
    mutate(status="neu")
pos_count <- afinn_pos %>%
    inner_join(reviews, by=c("word"="word")) %>%
    group_by(id, word) %>%
    summarise(n=n()) %>%
    mutate(status="pos")
vPos_count <- afinn_vPos %>%
    inner_join(reviews, by=c("word"="word")) %>%
    group_by(id, word) %>%
    summarise(n=n()) %>%
    mutate(status="vPos")

# Q2f ##########
all_count <- bind_rows(vNeg_count,neg_count,neu_count,pos_count,vPos_count)

# Q2g ##########
all_count <- spread(all_count,status,n)

# Q2h ##########
all_count[is.na(all_count)] <- 0

# Q2i ##########
all_count <- all_count %>%
    group_by(id) %>%
    summarise(vNeg=sum(vNeg),neg=sum(neg),neu=sum(neu),pos=sum(pos),vPos=sum(vPos))

# Q2j ##########
sent_data <- function(path){
    file.path <- path
    
    neg_extra <- c("second-rate", "moronic", "third-rate",
                   "flawed", "juvenile", "boring", "distasteful", "ordinary",
                   "disgusting", "senseless", "static", "brutal", "confused",
                   "disappointing", "bloody", "silly", "tired", "predictable",
                   "stupid", "uninteresting", "trite", "uneven", "outdated",
                   "dreadful", "bland")
    neu_extra <- c("suspenseful","low-budget","dramatic","high-charged", "sentimental",
                   "fantasy","slow","romantic","satirical","fast-moving", "oddball",
                   "picaresque","big-budget","wacky")
    pos_extra <- c("first-rate", "insightful", "clever", "charming",
                   "intriguing", "comical", "charismatic", "enjoyable", "absorbing",
                   "sensitive", "powerful", "pleasant", "surprising",
                   "thought-provoking", "imaginative", "unpretentious")
    vPos_extra <- c("uproarious", "riveting", "fascinating", "dazzling", "legendary")
    afinn <- get_sentiments("afinn")
    afinn_vNeg <- afinn %>%
        filter(score==-5 | score==-4) %>%
        select(word) %>%
        bind_rows()
    afinn_neg <- afinn %>%
        filter(score==-3 | score==-2 | score==-1) %>%
        select(word) %>%
        bind_rows(list(word=neg_extra))
    afinn_neu <- afinn %>%
        filter(score==0) %>%
        select(word) %>%
        bind_rows(list(word=neu_extra))
    afinn_pos <- afinn %>%
        filter(score==3 | score==2 | score==1) %>%
        select(word) %>%
        bind_rows(list(word=pos_extra))
    afinn_vPos <- afinn %>%
        filter(score==5 | score==4) %>%
        select(word) %>%
        bind_rows(list(word=vPos_extra))
    
    reviews <- as.data.frame(list.files(file.path),stringsAsFactors=F)
    colnames(reviews) <- c("file")
    reviews$file <- paste(file.path,reviews$file,sep="/")
    reviews$text <- unlist(map(reviews$file, read_lines))
    reviews$id <- row_number(rownames(reviews))
    reviews <- reviews %>%
        select(id,text) %>%
        unnest_tokens(word,text) %>%
        anti_join(stop_words, by=c("word"="word"))
    vNeg_count <- afinn_vNeg %>%
        inner_join(reviews, by=c("word"="word")) %>%
        group_by(id, word) %>%
        summarise(n=n()) %>%
        mutate(status="vNeg")
    neg_count <- afinn_neg %>%
        inner_join(reviews, by=c("word"="word")) %>%
        group_by(id, word) %>%
        summarise(n=n()) %>%
        mutate(status="neg")
    neu_count <- afinn_neu %>%
        inner_join(reviews, by=c("word"="word")) %>%
        group_by(id, word) %>%
        summarise(n=n()) %>%
        mutate(status="neu")
    pos_count <- afinn_pos %>%
        inner_join(reviews, by=c("word"="word")) %>%
        group_by(id, word) %>%
        summarise(n=n()) %>%
        mutate(status="pos")
    vPos_count <- afinn_vPos %>%
        inner_join(reviews, by=c("word"="word")) %>%
        group_by(id, word) %>%
        summarise(n=n()) %>%
        mutate(status="vPos")
    all_count <- bind_rows(vNeg_count,neg_count,neu_count,pos_count,vPos_count)
    all_count <- spread(all_count,status,n)
    all_count[is.na(all_count)] <- 0
    all_count <- all_count %>%
        group_by(id) %>%
        summarise(vNeg=sum(vNeg),neg=sum(neg),neu=sum(neu),pos=sum(pos),vPos=sum(vPos))
    
    return(all_count)
}

# Q2k ##########
pos <- sent_data('./aclImdb/train/pos')
pos$class <- 1
neg <- sent_data('./aclImdb/train/neg')
neg$class <- 0
sent_train <- bind_rows(pos,neg)

pos <- sent_data('./aclImdb/test/pos')
pos$class <- 1
neg <- sent_data('./aclImdb/test/neg')
neg$class <- 0
sent_test <- bind_rows(pos,neg)

nrow(sent_train)
```

    ## [1] 24907

``` r
nrow(sent_test)
```

    ## [1] 24910
