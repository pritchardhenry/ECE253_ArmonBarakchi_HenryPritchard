from GraphingFunctions import plotGraph

import numpy as np
#calc Statistics and make it a pdf and lessen white space
#write zoom in function
# Create 2 random grayscale images (100×100) to simulate test data
noisy_imgs = [np.random.rand(1000,1000) for _ in range(2)]
clean_imgs = [np.clip(img - 0.1*np.random.rand(1000,1000), 0, 1) for img in noisy_imgs]
org_imgs = [np.clip(img - 0.1*np.random.rand(1000,1000), 0, 1) for img in noisy_imgs]

# Call function (no original images)
plotGraph(clean_imgs, noisy_imgs, org_imgs, save_path="/Users/armonbarakchi/Desktop/ECE253_ArmonBarakchi_HenryPritchard/test.pdf")