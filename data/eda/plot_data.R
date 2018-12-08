library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(scales)
###TEST_SET####

##################
#boxes per image#
#################
bbpi_testset <- read.csv('/Users/mac/Documents/Study/IND/data_stats/complete_ds/csvs/Test set/boxes_per_image.csv')
ggplot(bbpi_testset, aes(x=num.of.boxes))+
  #geom_bar(aes(y=..count../sum(..count..)))+
  geom_histogram(binwidth = 2, aes(y=..count../sum(..count..)), color='black', size=0.25, fill='tomato1', center=1)+
  #geom_histogram(binwidth = 5, aes(y=..count..), color='white', center=2.5)+
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 0.9, by=0.025)) +
  scale_x_continuous(breaks = seq(0,500, by=10)) +
  labs(y='Percentage of images', x='Number of boxes')

############
#Dimensions#
############
dims_testset <-read.csv('/Users/mac/Documents/Study/IND/data_stats/complete_ds/csvs/Test set/boxes_dims.csv')

ggplot(dims_testset, aes(x=width,y=height))+
  geom_bin2d(binwidth=2, aes(fill=..count../sum(..count..)))+
  scale_fill_gradientn(colours = brewer.pal(6,"Oranges"),label=scales::percent)+
  scale_x_continuous(breaks = seq(0,100, by=10)) +
  scale_y_continuous(breaks = seq(0,150, by=10)) +
  theme_light()+
  labs(x='Width [px]', y='Height [px]', fill='')

############
#Distances#
###########
dists_testset <- read.csv('/Users/mac/Documents/Study/IND/data_stats/complete_ds/csvs/Test set/distances.csv')
ggplot(dists_testset, aes(x=x.dist,y=y.dist))+
  geom_bin2d(binwidth=5, aes(fill=..count../sum(..count..)))+
  scale_fill_gradientn(colours = brewer.pal(6,"Oranges"),label=scales::percent, breaks=seq(0 , 0.01, by=0.002))+
  theme_light()+
  labs(x='x component of the distance [px]', y='y component of the distance [px]', fill='')+
  coord_cartesian(xlim = c(0,50), ylim=c(0,50))
  
ggplot(dists_testset, aes(x=total.dist))+
  geom_histogram(binwidth=5, center=2.5, aes(y=..count../sum(..count..)), color='black', size=0.25, fill='tomato1')+
  coord_cartesian(xlim = c(0,50))+
  scale_y_continuous(labels = scales::percent) +
  labs(x='Total distance [px]', y='% of total number of distances')

#####VAL_SET#####
##################
#boxes per image#
#################
bbpi_valset <- read.csv('/Users/mac/Documents/Study/IND/data_stats/complete_ds/csvs/Validation set/boxes_per_image.csv')
ggplot(bbpi_valset, aes(x=num.of.boxes))+
  #geom_bar(aes(y=..count../sum(..count..)))+
  geom_histogram(binwidth = 2, aes(y=..count../sum(..count..)), color='black', size=0.25, fill='tomato1',center=1)+
  #geom_histogram(binwidth = 5, aes(y=..count..), color='white', center=2.5)+
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 0.9, by=0.025)) +
  scale_x_continuous(breaks = seq(0,500, by=10)) +
  labs(y='Percentage of images', x='Number of boxes')

############
#Dimensions#
############
dims_valset <-read.csv('/Users/mac/Documents/Study/IND/data_stats/complete_ds/csvs/Validation set/boxes_dims.csv')

#add more values to axis
ggplot(dims_valset, aes(x=width,y=height))+
  geom_bin2d(binwidth=2, aes(fill=..count../sum(..count..)))+
  scale_fill_gradientn(colours = brewer.pal(6,"Oranges"),label=scales::percent)+
  scale_x_continuous(breaks = seq(0,100, by=10)) +
  scale_y_continuous(breaks = seq(0,150, by=10)) +
  theme_light()+
  labs(x='Width [px]', y='Height [px]', fill='')

############
#Distances#
###########
dists_valset <- read.csv('/Users/mac/Documents/Study/IND/data_stats/complete_ds/csvs/Validation set/distances.csv')
ggplot(dists_valset, aes(x=x.dist,y=y.dist))+
  geom_bin2d(binwidth=5, aes(fill=..count../sum(..count..)))+
  scale_fill_gradientn(colours = brewer.pal(6,"Oranges"),label=scales::percent, breaks=seq(0 , 0.01, by=0.002))+
  theme_light()+
  labs(x='x component of the distance [px]', y='y component of the distance [px]', fill='')+
  coord_cartesian(xlim = c(0,50), ylim=c(0,50))
  
ggplot(dists_valset, aes(x=total.dist))+
  geom_histogram(binwidth=5, center=2.5, aes(y=..count../sum(..count..)), color='black', size=0.25, fill='tomato1')+
  coord_cartesian(xlim = c(0,50))+
  scale_y_continuous(labels = scales::percent) +
  labs(x='Total distance [px]', y='% of total number of distances')

#####TRAIN_SET#####

##################
#boxes per image#
#################
bbpi_trainset <- read.csv('/Users/mac/Documents/Study/IND/data_stats/complete_ds/csvs/Training set/boxes_per_image.csv')
ggplot(bbpi_trainset, aes(x=num.of.boxes))+
  #geom_bar(aes(y=..count../sum(..count..)))+
  geom_histogram(binwidth = 2, aes(y=..count../sum(..count..)), color='black', size=0.25, fill='tomato1',center=1)+
  #geom_histogram(binwidth = 5, aes(y=..count..), color='white', center=2.5)+
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 0.9, by=0.025)) +
  scale_x_continuous(breaks = seq(0,500, by=15)) +
  labs(y='Percentage of images', x='Number of boxes')
 

############
#Dimensions#
############
dims_trainset <-read.csv('/Users/mac/Documents/Study/IND/data_stats/complete_ds/csvs/Training set/boxes_dims.csv')

#add more values to axis
ggplot(dims_trainset, aes(x=width,y=height))+
  geom_bin2d(binwidth=2, aes(fill=..count../sum(..count..)))+
  scale_fill_gradientn(colours = brewer.pal(6,"Oranges"),label=scales::percent)+
  scale_x_continuous(breaks = seq(0,400, by=10)) +
  scale_y_continuous(breaks = seq(0,400, by=10)) +
  theme_light()+
  labs(x='Width [px]', y='Height [px]', fill='')

############
#Distances#
###########
dists_trainset <- read.csv('/Users/mac/Documents/Study/IND/data_stats/complete_ds/csvs/Training set/distances.csv')
ggplot(dists_trainset, aes(x=x.dist,y=y.dist))+
  geom_bin2d(binwidth=5, aes(fill=..count../sum(..count..)))+
  scale_fill_gradientn(colours = brewer.pal(6,"Oranges"),label=scales::percent, breaks=seq(0 , 0.01, by=0.002))+
  theme_light()+
  labs(x='x component of the distance [px]', y='y component of the distance [px]', fill='')+
  coord_cartesian(xlim = c(0,50), ylim=c(0,50))

ggplot(dists_trainset, aes(x=total.dist))+
  geom_histogram(binwidth=5, center=2.5, aes(y=..count../sum(..count..)), color='black', size=0.25, fill='tomato1')+
  coord_cartesian(xlim = c(0,50))+
  scale_y_continuous(labels = scales::percent) +
  labs(x='Total distance [px]', y='% of total number of distances')
