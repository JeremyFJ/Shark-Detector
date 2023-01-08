require(keras)
require(hash)
require(data.table)
require(stringr)

single_spec = c("Carcharias_taurus", "Carcharodon_carcharias", "Cetorhinus_maximus", 
    "Galeocerdo_cuvier", "Galeorhinus_galeus", "Galeus_melastomus", "Prionace_glauca", 
    "Rhincodon_typus", "Triaenodon_obesus")
'%!in%' <- function(x,y)!('%in%'(x,y))
addMod = function(mod, dict) { # creates model dictionary
  dict[[mod]] <- load_model_tf(paste0("/home/csteam/SDv2/species/models/", mod, "_mod"))
  return(dict)
}
load("./labels/GSC_label.R")

img_path = paste0("images/",list.files("images/"))
mod_list <<- hash() # create dictionary
mod_list[['Genus']] <- load_model_tf("models/GSC_mod") # load GSC

for (i in img_path) {
im <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(im)
x <- array_reshape(x, c(1, dim(x)))
x <- x / 255
pred <- mod_list[['Genus']] %>% predict(x)
pred <- data.frame("Shark" = label_list, "Prediction" = t(pred))
pred$Shark <- as.character(pred$Shark)
pred <- pred[order(pred$Prediction, decreasing=T),][1:5,]
topGen <- pred$Shark[1]
if (!any(single_spec %like% topGen) & topGen != "other_genus") {
    if (topGen %!in% names(mod_list)) {
            mod_list <<- addMod(topGen, mod_list)
        }
        load(paste0("./labels/",topGen,"_label.R"))
        pred_spec <- mod_list[[topGen]] %>% predict(x)
        pred_spec <- data.frame("Shark" = label_list, "Prediction" = t(pred_spec))
        pred_spec$Shark <- as.character(pred_spec$Shark)
        pred$Shark[1] <- pred_spec$Shark[1]
        }
        if (any(single_spec %like% pred$Shark[1])) {
            pred$Shark[1] <- gsub("_"," ",single_spec[grep(pred$Shark[1], single_spec)])
        }
        if (pred$Shark[1]=="other_genus") {
            pred$Shark[1] <- "Unknown genus"
        }

}
