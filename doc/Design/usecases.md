There are four different use cases for this project:

1. User will input electrolyte working ion of choice for battery design.
- Software will ask for electrolyte working ion
- Every possible compatable electrode material for the\ 
working ion will be filtred out from the BatteryData.csv dataframe\
generated and continuously updated from the Materials Project Battery Database.
- The trained ML model will predict the following parameters
	- Volumetric Capacity
	- Gravimetric Capacity
	- Volume Change
- For user validation, the loss function of the training data\
will be visually plotted and overlayed with the testing data.

2. Look for important trends in predicted parameters.
- The user will be able to relate one predicted parameter with another
- Depending on the desired design specification, the user will interact\
with the ploted data to select certain points in the dataset

3. Classify electrode material for specific battery design.
- From the selected point in the dataset, classify the electrode material

4. A simple and easy to use software to navigate through the Materials Project Database
