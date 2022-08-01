#--------------------------------------------------------------------------
# Load libraries
#--------------------------------------------------------------------------
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd

data = {'frame':[], 'Species': [], 'Top-2 Genus': [], 'Top-3 Genus': []}
dat = pd.DataFrame(data)
#-------------------------------------------------------------------------------------------------------
# Classification functions
#-------------------------------------------------------------------------------------------------------

# formats shark-detected image for classification
def img_to_classify(det_img, size, batch_size):
    img_1 = img_to_array(det_img.resize(size))
    img_1 = img_1.reshape((batch_size, img_1.shape[0], 
                        img_1.shape[1], img_1.shape[2]))
    img_1 = img_1/255
    return img_1

# predicts top 3 genera
def gsc_predict(gsc, img_1, gsc_labels):
    gen_pred = gsc.predict(img_1)
    gen_pred_dict = dict(zip(gsc_labels, gen_pred.tolist()[0]))
    gen_top3 = sorted(gen_pred_dict, key=gen_pred_dict.get, reverse=True)[:3]
    return gen_top3

# checks for particular genera that only contain one species
def single_spec_check(gen_top3, single_spec, frame, dat):
    import re
    # assign species (in the csv) name based on genus with one species
    r = re.compile(gen_top3[0]+"*") 
    species_name = list(filter(r.match, single_spec))[0] # matches genus name with species
    species_name = species_name.split("_")[0] + " " + species_name.split("_")[1]
    frame_df = pd.DataFrame([[frame, species_name, gen_top3[1], gen_top3[2]]], 
                        columns=list(dat.columns))
    dat = pd.concat([frame_df, dat])
    print(species_name + "\n" + gen_top3[1] + " " + gen_top3[2])
    return dat
    
# predicts top species
def sscg_predict(sscg, img_1, frame, gen_top3, dat):
    spec_pred = sscg[gen_top3[0]][0].predict(img_1)
    spec_pred_dict = dict(zip(sscg[gen_top3[0]][1], spec_pred.tolist()[0]))
    spec_top = sorted(spec_pred_dict, key=spec_pred_dict.get, reverse=True)[:1]
    species_name = spec_top[0]
    species_name = species_name.split("_")[0] + " " + species_name.split("_")[1]
    frame_df = pd.DataFrame([[frame, species_name, gen_top3[1], gen_top3[2]]], 
                        columns=list(dat.columns))
    dat = pd.concat([frame_df, dat])
    print(species_name + "\n" + gen_top3[1] + " " + gen_top3[2])
    return dat