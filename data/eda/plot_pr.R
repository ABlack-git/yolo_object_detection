library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(scales)

###MODEL 6L#####
#precision
tr_prec_1 <- read.csv('/Users/mac/Desktop/model+6l_valid/train/prec/run_10_11_2018__18-12-tag-train_avg_prec.csv')
tr_prec_2 <- read.csv('/Users/mac/Desktop/model+6l_valid/train/prec/run_10_11_2018__21-10-tag-train_avg_prec.csv')
tr_prec_3 <- read.csv('/Users/mac/Desktop/model+6l_valid/train/prec/run_11_11_2018__0-20-tag-train_avg_prec.csv')
tr_prec_4 <- read.csv('/Users/mac/Desktop/model+6l_valid/train/prec/run_11_11_2018__3-22-tag-train_avg_prec.csv')

tr_prec <- rbind(tr_prec_1, tr_prec_2, tr_prec_3, tr_prec_4)

val_prec_1 <- read.csv('/Users/mac/Desktop/model+6l_valid/valid/prec/run_10_11_2018__18-12-tag-validation_avg_prec.csv')
val_prec_2 <- read.csv('/Users/mac/Desktop/model+6l_valid/valid/prec/run_10_11_2018__21-10-tag-validation_avg_prec.csv')
val_prec_3 <- read.csv('/Users/mac/Desktop/model+6l_valid/valid/prec/run_11_11_2018__0-20-tag-validation_avg_prec.csv')
val_prec_4 <- read.csv('/Users/mac/Desktop/model+6l_valid/valid/prec/run_11_11_2018__3-22-tag-validation_avg_prec.csv')

val_prec <- rbind(val_prec_1, val_prec_2, val_prec_3, val_prec_4)

ggplot(tr_prec, aes(x=Step, y=Value))+
  geom_point(aes(color='Training precision'), shape=20)+
  geom_line(aes(color='Training precision'))+
  geom_point(data=val_prec, color='red', shape=20)+
  geom_line(data=val_prec, aes(color='Validation precision'))+
  scale_color_manual("", breaks=c('Validation precision', 'Training precision'), values=c('blue', 'red'))+
  labs(y='Precision')+
  theme(legend.position = 'top', legend.text = element_text(size=12))
#recall

tr_rec_1 <- read.csv('/Users/mac/Desktop/model+6l_valid/train/recall/1.csv')
tr_rec_2 <- read.csv('/Users/mac/Desktop/model+6l_valid/train/recall/2.csv')
tr_rec_3 <- read.csv('/Users/mac/Desktop/model+6l_valid/train/recall/3.csv')
tr_rec_4 <- read.csv('/Users/mac/Desktop/model+6l_valid/train/recall/4.csv')

tr_rec <- rbind(tr_rec_1, tr_rec_2, tr_rec_3, tr_rec_4)

val_rec_1 <- read.csv('/Users/mac/Desktop/model+6l_valid/valid/recall/1.csv')
val_rec_2 <- read.csv('/Users/mac/Desktop/model+6l_valid/valid/recall/2.csv')
val_rec_3 <- read.csv('/Users/mac/Desktop/model+6l_valid/valid/recall/3.csv')
val_rec_4 <- read.csv('/Users/mac/Desktop/model+6l_valid/valid/recall/4.csv')

val_rec <- rbind(val_rec_1, val_rec_2, val_rec_3, val_rec_4)

ggplot(tr_rec, aes(x=Step, y=Value))+
  geom_point(aes(color='Training recall'), shape=20)+
  geom_line(aes(color='Training recall'))+
  geom_point(data=val_rec, color='red', shape=20)+
  geom_line(data=val_rec, aes(color='Validation recall'))+
  scale_color_manual("", breaks=c('Validation recall', 'Training recall'), values=c('blue', 'red'))+
  labs(y='Recall')+
  theme(legend.position = 'top', legend.text = element_text(size=12))

####MODEL 8L####
#precision
tr_prec_1 <- read.csv('/Users/mac/Desktop/model_8l/train/prec/1.csv')
tr_prec_2 <- read.csv('/Users/mac/Desktop/model_8l/train/prec/2.csv')
tr_prec_3 <- read.csv('/Users/mac/Desktop/model_8l/train/prec/3.csv')

tr_prec <- rbind(tr_prec_1, tr_prec_2, tr_prec_3)

val_prec_1 <- read.csv('/Users/mac/Desktop/model_8l/val/prec/1.csv')
val_prec_2 <- read.csv('/Users/mac/Desktop/model_8l/val/prec/2.csv')
val_prec_3 <- read.csv('/Users/mac/Desktop/model_8l/val/prec/3.csv')

val_prec <- rbind(val_prec_1, val_prec_2, val_prec_3)

ggplot(tr_prec, aes(x=Step, y=Value))+
  geom_point(aes(color='Training precision'), shape=20)+
  geom_line(aes(color='Training precision'))+
  geom_point(data=val_prec, color='red', shape=20)+
  geom_line(data=val_prec, aes(color='Validation precision'))+
  scale_color_manual("", breaks=c('Validation precision', 'Training precision'), values=c('blue', 'red'))+
  labs(y='Precision')+
  theme(legend.position = 'top', legend.text = element_text(size=12))

#recall
tr_rec_1 <- read.csv('/Users/mac/Desktop/model_8l/train/recall/1.csv')
tr_rec_2 <- read.csv('/Users/mac/Desktop/model_8l/train/recall/2.csv')
tr_rec_3 <- read.csv('/Users/mac/Desktop/model_8l/train/recall/3.csv')

tr_rec <<- rbind(tr_rec_1, tr_rec_2, tr_rec_3)

val_rec_1 <- read.csv('/Users/mac/Desktop/model_8l/val/recall/1.csv')
val_rec_2 <- read.csv('/Users/mac/Desktop/model_8l/val/recall/2.csv')
val_rec_3 <- read.csv('/Users/mac/Desktop/model_8l/val/recall/3.csv')

val_rec <- rbind(val_rec_1, val_rec_2, val_rec_3)

ggplot(tr_rec, aes(x=Step, y=Value))+
  geom_point(aes(color='Training recall'), shape=20)+
  geom_line(aes(color='Training recall'))+
  geom_point(data=val_rec, color='red', shape=20)+
  geom_line(data=val_rec, aes(color='Validation recall'))+
  scale_color_manual("", breaks=c('Validation recall', 'Training recall'), values=c('blue', 'red'))+
  labs(y='Recall')+
  theme(legend.position = 'top', legend.text = element_text(size=12))