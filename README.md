# Mahjong Tile Identifier
A project (in progress) that can identify the class of a tile from a picture,
with the classes as follows: [bamboos, chars, dots, dragons, flowers, winds]. 
Bamboos, chars, and dots each contain its respective numbers 1-9. Dragons include
the red 中, green fa, and blank tile. Flowers include both the flower and season
tiles 1-4. Winds include the 4 directional pieces 東, 西, 南, 北.


# Current models
All models that you see currently are the results of modifying 
parameters (epochs, hidden units, etc). Although all
models as of currently aren't very good (~3% test accuracy).
\\
6/28/2023:
Most recent model ver10 has an increased accuracy of ~30%

# How to use:
Navigate to file prediction.py and change the model number of line 71 to your liking,
then replace the image file on line 77 to one of your choice, then run. 