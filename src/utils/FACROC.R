#' @title FACROC
#'
#' @description Plots the FACROC of a clustering solution.
#'

interpolate_roc_fun <- function(perf_in, n_grid = 40000) {
  x_vals <- unlist(perf_in$fpr)
  y_vals <- unlist(perf_in$tpr)
  stopifnot(length(x_vals) == length(y_vals))
  roc_approx <- stats::approx(x_vals, y_vals, n = n_grid)
  return(roc_approx)
}
#'
#'
#'
facroc_plot <- function(non_protected_roc, protected_roc, non_protected_group_name = NULL,
                       protected_group_name = NULL, fout = NULL,facroc_vals=NULL) {
  # check that number of points are the same
  stopifnot(length(non_protected_roc$fpr) == length(non_protected_roc$tpr),
            length(non_protected_roc$fpr) == length(protected_roc$tpr),
            length(non_protected_roc$fpr) == length(protected_roc$tpr))
  if (!is.null(fout)) {
    #pdf file
    grDevices::pdf(fout, width = 4, height = 4)
    #png file
    #grDevices::png(fout, width = 720, height = 720)
  }
  # set some graph parameters
  non_protected_color <- "red"
  protected_color <- "blue"
  non_protected_group_label <- non_protected_group_name
  protected_group_label <- protected_group_name
  plot_title <- paste("FACROC = ",round(facroc_vals,digits = 4), sep = "")
  
  # add labels, if given
  graphics::plot(non_protected_roc$x, non_protected_roc$y, col = non_protected_color,
                     type = "l", lwd = 1.5, main = plot_title,
                     xlab = substitute(paste(bold("False Positive Rate"))), ylab = substitute(paste(bold("True Positive Rate"))))
  # draw polygon; reverse ordering used to close polygon by ending near start
  # point
  graphics::polygon(x = c(non_protected_roc$x, rev(protected_roc$x)),
                        y = c(non_protected_roc$y, rev(protected_roc$y)),
                        col = "grey", border = NA)
  graphics::lines(non_protected_roc$x, non_protected_roc$y, col = non_protected_color,
                      type = "l", lwd = 1.5)
  graphics::lines(protected_roc$x, protected_roc$y, col = protected_color,
                      type = "l", lwd = 1.5)
  graphics::legend("bottomright",
                       legend = c(non_protected_group_label, protected_group_label),
                       col = c(non_protected_color, protected_color), lty = 1)
  if (!is.null(fout)) {
        grDevices::dev.off()
      }
}

compute_facroc <- function(auccResult_protected,auccResult_non_protected,protected_attribute="Gender",protected="Female",non_protected="Male", showPlot = TRUE, filename = NULL) {
  if (class(auccResult_protected) != 'aucc' || class(auccResult_non_protected) != 'aucc') {
    stop('Object auccResults has to come from function aucc, with returnRates = TRUE')
  }
  # initialize data structures
  fr <- 0
  
  non_protected_roc_fun <- interpolate_roc_fun(auccResult_non_protected)
  
  protected_roc_fun <- interpolate_roc_fun(auccResult_protected)
  # use function approximation to compute slice statistic
  # via piecewise linear function
  stopifnot(identical(non_protected_roc_fun$x, protected_roc_fun$x))
  f1 <- stats::approxfun(non_protected_roc_fun$x,
                         non_protected_roc_fun$y - protected_roc_fun$y)
  f2 <- function(x) abs(f1(x))  # take the positive value
  slice <- stats::integrate(f2, 0, 1, subdivisions = 10000L)$value
  fr <- fr + slice
  # plot these or write to file
  if (showPlot == TRUE) {
    output_filename <- filename
    facroc_plot(non_protected_roc_fun, protected_roc_fun, non_protected, protected, fout = output_filename,facroc_vals = fr)
  }
  return(fr)
}


FACROC <- function(auccResult1,auccResult2,protected_attribute="Gender",protected="Female",non_protected="Male", showPlot = TRUE, rLine = TRUE, sample = TRUE, sampleSize = 500,size = 1, lineType = 1) {
  
  FPR <- TPR <- NULL
  
  if (class(auccResult1) != 'aucc' || class(auccResult2) != 'aucc') {
    stop('Object auccResults has to come from function aucc, with returnRates = TRUE')
  }
  
  if (sample && sampleSize < length(auccResult1$fpr)) {
    subsample1 <- c(1,sample(1:length(auccResult1$fpr),sampleSize),length(auccResult1$fpr))
    rDF1 <- data.frame(cbind(auccResult1$fpr[subsample1],auccResult1$tpr[subsample1]))
  } else {
    rDF1 <- data.frame(cbind(auccResult1$fpr,auccResult1$tpr))
  }
  
  colnames(rDF1) <- c('FPR','TPR')
  
  rPlot <- ggplot2::ggplot(rDF1,aes(x=FPR,y=TPR,color = protected)) +
      ggplot2::geom_line(size = size, linetype = lineType)  + 
      ggplot2::theme_light()
  
  
  if (sample && sampleSize < length(auccResult2$fpr)) {
    subsample2 <- c(1,sample(1:length(auccResult2$fpr),sampleSize),length(auccResult2$fpr))
    rDF2<- data.frame(cbind(auccResult2$fpr[subsample2],auccResult2$tpr[subsample2]))
  } else {
    rDF2 <- data.frame(cbind(auccResult2$fpr,auccResult2$tpr))
  }
  
  colnames(rDF2) <- c('FPR','TPR')
  
  
  rPlot <- rPlot +
      ggplot2::geom_line(data=rDF2,ggplot2::aes(x=FPR,y=TPR,color = non_protected), size = size, linetype = lineType) + 
      ggplot2::theme_light()  + scale_color_manual(name = protected_attribute, values = c("red","blue")) +
      ggplot2::theme(legend.position = c(0.8, 0.2))
  
  
  if (rLine) {
    rPlot <- rPlot +
      ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1), color = 'darkgrey', linetype=2)
  }
  
  if (showPlot) {
    print(rPlot)
  }
  
  rPlot
}


