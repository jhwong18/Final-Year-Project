library(vtreat)
library(xlsx)
library(cluster) 

#-------------------------------------------------------------------------
# Setting up the training and testing dataset
set.seed(123)
setwd("C:\\Users\\User\\Documents")
data = read.csv('vipdatamodelling6.csv', header = TRUE, sep=",")
data[,c(1,2,3,4,5,6,7,9)] = NULL
pred = as.matrix(data)

#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# K means clustering


#Elbow Method for finding the optimal number of clusters
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
wss <- sapply(1:k.max, 
              function(k){kmeans(pred, k)$tot.withinss})
wss
plot(1:k.max, wss, main = "Elbow plot (Within cluster sum of squares)",
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares") # Figure 5.1.1

# Get cluster means for 3 cluster solution 
#aggregate(cluster, by=list(kmeans$cluster),FUN=mean)
kmm = kmeans(pred,3,nstart = 50,iter.max = 15) # start with 3 cluster solution
cluster=data.frame(pred,cluster=kmm$cluster)
write.xlsx(cluster, file = "3cluster.xlsx")
C1=cluster[(cluster[,2]==1),]
C2=cluster[(cluster[,2]==2),]
C3=cluster[(cluster[,2]==3),]
C1;C2;C3
sil = silhouette (kmm$cluster,dist(pred))
windows() 
plot(sil,  main="K-means (3 Cluster) silhouette score")  # Figure 5.1.4
#-------------------------------------------------------------------------



#-------------------------------------------------------------------------
# Not Used
# # gap statistic
# library(plyr)
# library(ggplot2)
# data <- pred
# # Given a matrix `data`, where rows are observations and columns are individual dimensions, compute and plot the gap statistic (according to a uniform reference distribution).
# gap_statistic = function(data, min_num_clusters = 1, max_num_clusters = 5, num_reference_bootstraps = 5) {
#   num_clusters = min_num_clusters:max_num_clusters
#   actual_dispersions = maply(num_clusters, function(n) dispersion(data, n))
#   ref_dispersions = maply(num_clusters, function(n) reference_dispersion(data, n, num_reference_bootstraps))
#   mean_ref_dispersions = ref_dispersions[ , 1]
#   stddev_ref_dispersions = ref_dispersions[ , 2]
#   gaps = mean_ref_dispersions - actual_dispersions
#   
#   print(plot_gap_statistic(gaps, stddev_ref_dispersions, num_clusters))
#   
#   print(paste("The estimated number of clusters is ", num_clusters[which.max(gaps)], ".", sep = ""))
#   
#   list(gaps = gaps, gap_stddevs = stddev_ref_dispersions)
# }
# 
# # Plot the gaps along with error bars.
# plot_gap_statistic = function(gaps, stddevs, num_clusters) {
#   qplot(num_clusters, gaps, xlab = "# clusters", ylab = "gap", geom = "line", main = "Estimating the number of clusters via the gap statistic") + geom_errorbar(aes(num_clusters, ymin = gaps - stddevs, ymax = gaps + stddevs), size = 0.3, width = 0.2, colour = "darkblue")
# }
# 
# # Calculate log(sum_i(within-cluster_i sum of squares around cluster_i mean)).
# dispersion = function(data, num_clusters) {
#   # R's k-means algorithm doesn't work when there is only one cluster.
#   if (num_clusters == 1) {
#     cluster_mean = aaply(data, 2, mean)
#     distances_from_mean = aaply((data - cluster_mean)^2, 1, sum)
#     log(sum(distances_from_mean))
#   } else {	
#     # Run the k-means algorithm `nstart` times. Each run uses at most `iter.max` iterations.
#     k = kmeans(data, centers = num_clusters)
#     # Take the sum, over each cluster, of the within-cluster sum of squares around the cluster mean. Then take the log. This is `W_k` in TWH's notation.
#     log(sum(k$withinss))
#   }
# }
# 
# # For an appropriate reference distribution (in this case, uniform points in the same range as `data`), simulate the mean and standard deviation of the dispersion.
# reference_dispersion = function(data, num_clusters, num_reference_bootstraps) {
#   dispersions = maply(1:num_reference_bootstraps, function(i) dispersion(generate_uniform_points(data), num_clusters))
#   mean_dispersion = mean(dispersions)
#   stddev_dispersion = sd(dispersions) / sqrt(1 + 1 / num_reference_bootstraps) # the extra factor accounts for simulation error
#   c(mean_dispersion, stddev_dispersion)
# }
# 
# # Generate uniform points within the range of `data`.
# generate_uniform_points = function(data) {
#   # Find the min/max values in each dimension, so that we can generate uniform numbers in these ranges.
#   mins = aaply(data, 2, min)
#   maxs = apply(data, 2, max)
#   
#   num_datapoints = nrow(data)
#   # For each dimension, generate `num_datapoints` points uniformly in the min/max range.
#   uniform_pts = maply(1:length(mins), function(dim) runif(num_datapoints, min = mins[dim], max = maxs[dim]))
#   uniform_pts = t(uniform_pts)
# }
# library(reshape2)
# t <- melt(data)
# gap_statistic(t)
# dev.off()

# library("phyloseq")
# library("cluster")
# library("ggplot2")
# pam1 = function(x, k){list(cluster = pam(x,k, cluster.only=TRUE))}
# x <- t
# gskmn = clusGap(t, FUN=pam1, K.max = 10, B = 1)
# plot_clusgap = function(clusgap, title="Gap Statistic calculation results"){
#   require("ggplot2")
#   gstab = data.frame(clusgap$Tab, k=1:nrow(clusgap$Tab))
#   p = ggplot(gstab, aes(k, gap)) + geom_line() + geom_point(size=5)
#   p = p + geom_errorbar(aes(ymax=gap+SE.sim, ymin=gap-SE.sim))
#   p = p + ggtitle(title)
#   return(p)
# }
# plot_clusgap(gskmn)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# K-medoids Method
library(cluster)
#Elbow Method for finding the optimal number of clusters
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
wss <- sapply(1:k.max, 
              function(k){pam(pred, k, metrix="euclidean")$tot.withinss})
wss
plot(1:k.max, wss, main = "Elbow plot (Within cluster sum of squares)",
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares") # Figure 5.1.2

kmedoids <- pam(pred, 3, metric="euclidean") # 3 cluster solution
cluster=data.frame(pred,cluster=kmedoids$cluster)
write.xlsx(cluster, file = "3cluster_kmedoid.xlsx")
MC1=cluster[(cluster[,2]==1),]
MC2=cluster[(cluster[,2]==2),]
MC3=cluster[(cluster[,2]==3),]
sil = silhouette (kmedoids$cluster,dist(pred))  # Figure 5.1.5
windows() 
plot(sil,  main="K-medoids (3 Cluster) silhouette score")

par(mfrow=c(2,3))
hist(as.matrix(C1[,1]),main="Cluster 1 (K-means)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.1.3
hist(as.matrix(C2[,1]),main="Cluster 2 (K-means)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.1.3 
hist(as.matrix(C3[,1]),main="Cluster 3 (K-means)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.1.3

hist(as.matrix(MC1[,1]),main="Cluster 1 (K-medoids)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.1.3
hist(as.matrix(MC2[,1]),main="Cluster 2 (K-medoids)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.1.3
hist(as.matrix(MC3[,1]),main="Cluster 3 (K-medoids)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.1.3

# Get cluster means 
# aggregate(X,by=list(kmedoids$cluster),FUN=mean)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Hierarchical clustering 
# Ward Linkage 3 Cluster
ds <- dist(pred, method="euclidean")
ward=hclust(ds, method="ward")
par(mfrow=c(1,1))
plot(ward, hang=-1,  main=" Ward Linkage : Standardized Euclidean Distance") # Figure 5.2.1
wardclusterCut1 <- cutree(ward, 3) 
library(reshape2)
t <- melt(pred)
t1 <- melt(wardclusterCut1)
t11 <- cbind(t,t1)
write.xlsx(t11, file = "3cluster_ward.xlsx")

C1 <- t11[t11[,4]==1,]
C2 <- t11[t11[,4]==2,]
C3 <- t11[t11[,4]==3,]
sil = silhouette (wardclusterCut1,dist(pred))  
windows() 
plot(sil,  main="Ward Linkage (3 Cluster) silhouette score") # Figure 5.2.6


par(mfrow=c(1,3))
hist(as.matrix(C1[,3]),main="Cluster 1 (Ward)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.5
hist(as.matrix(C2[,3]),main="Cluster 2 (Ward)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.5
hist(as.matrix(C3[,3]),main="Cluster 3 (Ward)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.5

#-------------------------------------------------------------------------
# Ward Linkage 4 Cluster 
wardclusterCut2 <- cutree(ward, 4)
library(reshape2)
t <- melt(pred)
t2 <- melt(wardclusterCut2)
t22 <- cbind(t,t2)
C1 <- t22[t22[,4]==1,]
C2 <- t22[t22[,4]==2,]
C3 <- t22[t22[,4]==3,]
C4 <- t22[t22[,4]==4,]
sil = silhouette (wardclusterCut2,dist(pred))
windows() 
plot(sil,  main="Ward Linkage (4 Cluster) silhouette score") # Figure 5.2.2



par(mfrow=c(1,4))
hist(as.matrix(C1[,3]),main="Cluster 1 (Ward)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.7
hist(as.matrix(C2[,3]),main="Cluster 2 (Ward)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.7
hist(as.matrix(C3[,3]),main="Cluster 3 (Ward)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.7
hist(as.matrix(C4[,3]),main="Cluster 4 (Ward)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.7
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Average Linkage 3 Cluster
clusters <- hclust(ds, method = 'average')
plot(clusters, hang=-1,  main="Average Linkage : Standardized Euclidean Distance")
avgclusterCut1 <- cutree(clusters, 3)
t <- melt(pred)
t3 <- melt(avgclusterCut1)
t33 <- cbind(t,t3)
C1 <- t33[t33[,4]==1,]
C2 <- t33[t33[,4]==2,]
C3 <- t33[t33[,4]==3,]
sil = silhouette (avgclusterCut1,dist(pred))
windows() 
plot(sil,  main="Average Linkage (3 Cluster) silhouette score") # Figure 5.2.14

par(mfrow=c(1,3))
hist(as.matrix(C1[,3]),main="Cluster 1 (average)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.13
hist(as.matrix(C2[,3]),main="Cluster 2 (average)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.13
hist(as.matrix(C3[,3]),main="Cluster 3 (average)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.13
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Average Linkage 4 Cluster
avgclusterCut2 <- cutree(clusters, 4)
t4 <- melt(avgclusterCut2)
t44 <- cbind(t,t4)
C1 <- t44[t44[,4]==1,]
C2 <- t44[t44[,4]==2,]
C3 <- t44[t44[,4]==3,]
C4 <- t44[t44[,4]==4,]
sil = silhouette (avgclusterCut2,dist(pred))
windows() 
plot(sil,  main="Average Linkage (4 Cluster) silhouette score") # Figure 5.2.16

par(mfrow=c(2,3))
hist(as.matrix(C1[,3]),main="Cluster 1 (average)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.15
hist(as.matrix(C2[,3]),main="Cluster 2 (average)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.15
hist(as.matrix(C3[,3]),main="Cluster 3 (average)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.15
hist(as.matrix(C4[,3]),main="Cluster 4 (average)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.15
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
# Single Linkage 3 Cluster
single <- hclust(ds, method = 'single')
plot(single, hang=-1,  main="Single Linkage : Standardized Euclidean Distance")
singleclusterCut1 <- cutree(single, 3)
t5 <- melt(singleclusterCut1)
t55 <- cbind(t,t5)
C1 <- t55[t55[,4]==1,]
C2 <- t55[t55[,4]==2,]
C3 <- t55[t55[,4]==3,]
sil = silhouette (singleclusterCut1,dist(pred))
windows() 
plot(sil,  main="Single Linkage (3 Cluster) silhouette score") # Figure 5.2.10

par(mfrow=c(1,3))
hist(as.matrix(C1[,3]),main="Cluster 1 (single)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.9
hist(as.matrix(C2[,3]),main="Cluster 2 (single)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.9
hist(as.matrix(C3[,3]),main="Cluster 3 (single)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.9

#-------------------------------------------------------------------------
# Complete Linkage 3 Cluster
complete <- hclust(ds, method = 'complete')
plot(complete, hang=-1,  main="Complete Linkage : Standardized Euclidean Distance")
completeclusterCut1 <- cutree(complete, 3)
t6 <- melt(completeclusterCut1)
t66 <- cbind(t,t6)
C1 <- t66[t66[,4]==1,]
C2 <- t66[t66[,4]==2,]
C3 <- t66[t66[,4]==3,]
#C4 <- t66[t66[,4]==4,]
sil = silhouette (completeclusterCut1 ,dist(pred))
windows() 
plot(sil,  main="Complete Linkage (3 Cluster) silhouette score") # Figure 5.2.11

par(mfrow=c(2,2))
hist(as.matrix(C1[,3]),main="Cluster 1 (Complete)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.12
hist(as.matrix(C2[,3]),main="Cluster 2 (Complete)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.12
hist(as.matrix(C3[,3]),main="Cluster 3 (Complete)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.12
#hist(as.matrix(C4[,3]),main="Cluster 4 (Complete)",ylab="Frequency",xlab="Full cut promotion demand") # Figure 5.2.12

#-------------------------------------------