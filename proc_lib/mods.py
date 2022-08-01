#-------------------------------------------------------------------------------------------------------
# Load libraries
#-------------------------------------------------------------------------------------------------------
import os
from keras.models import load_model
import pickle
#--------------------------------------------------------------------------
# Directories
#--------------------------------------------------------------------------
spec_models = "models/species/"
spec_labels = "label_list/labels_py/"
gsc_lab = "label_list/"
os.chdir("../run/")

# for f in os.listdir('frames/'):
#    os.remove('frames/' + f)

gsc_labels = sorted(pickle.loads(open(gsc_lab + "GSC.pickle", "rb").read()))
single_spec = ["Carcharias_taurus", "Carcharodon_carcharias", "Cetorhinus_maximus", 
    "Galeocerdo_cuvier", "Galeorhinus_galeus", "Galeus_melastomus", "Prionace_glauca", 
    "Rhincodon_typus", "Triaenodon_obesus"]
#-------------------------------------------------------------------------------------------------------
# Loading classification models function
#-------------------------------------------------------------------------------------------------------
def load_SC(taxonomy="genus"):
    if taxonomy=="genus":
        model = load_model("models/GSC_mod/")
        print("GSC loaded")
    elif taxonomy=="species":
        model = dict()
        for mod in os.listdir(spec_models):
            gen_name = mod.split("_")[0]
            # load class labels for each SSCg model
            classes = sorted(pickle.loads(open(spec_labels + gen_name + ".pickle", "rb").read()))
            # create dictionary of models and labels
            model[gen_name] = [load_model(spec_models + mod),classes] 
            print("\r" + gen_name + "--loaded")
        print("SSCg loaded\n")
    return model