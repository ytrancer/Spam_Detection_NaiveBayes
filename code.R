library(readr)
library(tm)
library(SnowballC)
library(wordcloud)
library(e1071)
library(gmodels)

sms_raw <- read_csv("https://raw.githubusercontent.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/master/Chapter04/sms_spam.csv")

# As 'type' is a character vector, we are converting it to a factor.
sms_raw$type <-factor(sms_raw$type) 
table(sms_raw$type)

# Create a source object used in VSource
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
inspect(sms_corpus[1:2])

# To view the actual message text you have to use 'as.character()'
as.character(sms_corpus[[1]])

# To view multiple messages use lapply.
lapply(sms_corpus[1:2], as.character)

# Clean the corpus with tm_map series of transformations and save in corpus_clean.

# Convert to lower case.
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
as.character(sms_corpus_clean[[1]])

# Remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

# Stopwords
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

# Punctuation
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

# Stemming
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

# Strip additional whitespace
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

# Create a document-term matrix (DTM)
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

# Another way is to run DocumentTermMatrix overriding the default parametres
sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

# Create a training and testing set.

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

# Save a pair of vectors with labels for each row in the training and testing matrices
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

# Compare the proportion of spam in the training and test data frames
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels)) # Both are at about 13% spam

# Data Visualisation
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

# Compare cloud of spam/ham
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

# Create a vector of words with frequency > 5
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)

# Filter DTM to only include freqent words
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

# Define a function to convert counts to yes or no.
convert_counts <- function(x) {
  x <- ifelse(x>0, "Yes", "No")
}

# Apply the function to columns of the sparce matrix.
# MARGIN = 2 means apply to columns
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

# Training the model
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

# Evaluating model performance
sms_test_pred <- predict(sms_classifier, sms_test)

# Use CrossTable() from gmodels.
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, 
           prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('predicted', 'actual'))

# A new model with Laplace = 1
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
