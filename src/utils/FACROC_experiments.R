source("FACROC.R")
library(clusterConfusion)

facroc_experiment <- function(dataset, clustering_result, figure_out, protected_attr, protected_group, non_protected_group, protected_label, non_protected_label) {
  data <- read.csv(file = dataset)
  clustering <- read.csv(file = clustering_result)
  fileout <- figure_out 
  
  data_f <- data[data[[protected_attr]]==protected_group,]
  clustering_f <- clustering[clustering[[protected_attr]]==protected_group,]
  if (nrow(data_f) > 3000) {
    idx <- sample(1:nrow(data_f), 3000)
    data_f <- data_f[idx,]
    clustering_f <- clustering_f[idx,]
  }
  clustering_f <- clustering_f[c('cluster')]
  clustering_f <- clustering_f$cluster
  evaluation_f <- aucc(clustering_f, dataset = data_f, returnRates=TRUE)
  
  data_m <- data[data[[protected_attr]]==non_protected_group,]
  clustering_m <- clustering[clustering[[protected_attr]]==non_protected_group,]
  if (nrow(data_m) > 3000) {
    idx <- sample(1:nrow(data_m), 3000)
    data_m <- data_m[idx,]
    clustering_m <- clustering_m[idx,]
  }
  clustering_m <- clustering_m[c('cluster')]
  clustering_m <- clustering_m$cluster
  evaluation_m <- aucc(clustering_m, dataset = data_m, returnRates=TRUE)
  
  facroc <- compute_facroc(auccResult_protected = evaluation_f, auccResult_non_protected = evaluation_m, 
                           protected_attribute = protected_attr,protected=protected_label,non_protected=non_protected_label,
                           showPlot = TRUE, filename = fileout)
  return(facroc)
}

datasets <- list(
  list(name="adult", file="data/adult_clean.csv", p_attr="gender", p_group="Female", np_group="Male", p_label="Female", np_label="Male"),
  list(name="germancredit", file="data/german_data_credit.csv", p_attr="gender", p_group="female", np_group="male", p_label="Female", np_label="Male"),
  list(name="compas", file="data/compas_clean.csv", p_attr="race", p_group="White", np_group="Non-White", p_label="White", np_label="Non-White"),
  list(name="credit", file="data/credit_clean.csv", p_attr="SEX", p_group="F", np_group="M", p_label="Female", np_label="Male"),
  list(name="studentmat", file="data/studentmat.csv", p_attr="gender", p_group="F", np_group="M", p_label="Female", np_label="Male"),
  list(name="studentpor", file="data/student-por-encode.csv", p_attr="gender", p_group="F", np_group="M", p_label="Female", np_label="Male")
)

methods <- c("kmean", "hierarchical", "fairlet", "scalable", "proportionally")

for (ds in datasets) {
  cat("\nProcessing dataset:", ds$name, "\n")
  for (method in methods) {
    clustering_file <- paste0("clustering/", method, "_", ds$name, ".csv")
    figure_file <- paste0(ds$name, ".", method, ".facroc.pdf")
    
    if (file.exists(clustering_file)) {
      cat("  Running method:", method, "\n")
      tryCatch({
        score <- facroc_experiment(dataset=ds$file, clustering_result=clustering_file, 
                                  figure_out=figure_file, protected_attr=ds$p_attr, 
                                  protected_group=ds$p_group, non_protected_group=ds$np_group, 
                                  protected_label=ds$p_label, non_protected_label=ds$np_label)
        cat("    FACROC =", score, "\n")
      }, error = function(e) {
        cat("    Error running", method, ":", e$message, "\n")
      })
    } else {
      cat("  Missing clustering file for:", method, "\n")
    }
  }
}
