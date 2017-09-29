from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as st
import numpy as np
import pandas as pd
import tifffile as tiff
import sys
import random as r
import scipy as sp
import re
import matplotlib.cm as cm
cmap_ktor=LinearSegmentedColormap.from_list('blacktored',['black','red'])
cmap_ktob=LinearSegmentedColormap.from_list('blacktoblue',['black','blue'])

sys.setrecursionlimit(40000) 

# Random number generator that generates a random color in the format (0-1.0, 0-1.0,0-1.0) for use in the graphing 
def random():
    return (r.random(), r.random(), r.random())

# Method accepts a list of all of the files that are to be analyzed. Produces a maximum intensity projection for the 
# Cy3 (TMR-Dextran, macropinosomes) pictures that have multiple z-Stacks. Maps these new images with their name, one with regular images and one with maximum intensity projections
# and passes them into regex method to be sorted by treatment
def run(pictureNames):
    nameToPicture = {}
    nameToMaxIntensity = {}
    for fileName in pictureNames:
        currentImage = tiff.imread(fileName)
        isZStack = fileName.find("Midvolume")
        if isZStack is -1:
            currentImage = currentImage.max(axis=0)
            nameToMaxIntensity[fileName] = currentImage
        else:
            nameToPicture[fileName] = currentImage
    return regex({"DAPI":nameToPicture, "Cy3":nameToMaxIntensity})

# From name of picture, extracts treatment group information using a regex and stores this information along with the original name into a map 
# of DAPI (nuclei) and Cy3 (Macropinosomes) passes these images into findArea
def regex(pictures):
    dapi, cy3 = "DAPI", "Cy3"
    dapiMap, cy3Map = pictures[dapi], pictures[cy3]
    regexedDAPIMap, regexedCy3Map = {}, {}
    parseName = re.compile('AL1288\s[A-Z][0-9]{1,2}\simage\s[0-9]{1,2}\s[0-9]?\s?([A-Z]{1,3}[0-9]{1,3}-?[0-9]?\s([A-Z]{3,4}[0-9]{0,3}))')
    for key in dapiMap.keys():
        nameBreakDownDAPI = tuple([(parseName.match(key)).group(i) for i in range(0, 2)])
        regexedDAPIMap[nameBreakDownDAPI] = dapiMap[key]
    for keys in cy3Map.keys():
        nameBreakDownCy3 = tuple([(parseName.match(keys)).group(i) for i in range(0, 2)])
        regexedCy3Map[nameBreakDownCy3] = cy3Map[keys]         
    return countMacropinosomes({dapi:regexedDAPIMap, cy3:regexedCy3Map})

# Takes in regexed images and for each image, counts the number of nuclei in these images (DAPi channel)
# and number of macropinosomes (Cy3 channel) and maps these numbers with the regexed picture names ObjectCount
# is the method that counts both number of nuclei and macropinosomes. Pairs names with count of either nuclei or macropinosomes
def countMacropinosomes(regexedPictures):
    # these two values are determined by manual thresholding images in imageJ
    thresholdDAPI = 2000
    thresholdCy3 = 9300
    # this is the number of pixels that nuclei and macropinosome should take up with current magnification
    nucleiPixels = 900
    macropinosomePixels = 1
    dapi, cy3 = "DAPI", "Cy3"
    dapiMap, cy3Map = regexedPictures[dapi], regexedPictures[cy3]
    statsDAPIMap, statsCy3Map = {}, {}
    for pictureNameDAPI in dapiMap.keys():
        currentImageDAPI = dapiMap[pictureNameDAPI]
        nucleiCount = objectCount(currentImageDAPI, thresholdCy3, nucleiPixels)
        statsDAPIMap[pictureNameDAPI] = nucleiCount         
    for pictureNameCy3 in cy3Map.keys():    
        currentImageCy3 = cy3Map[pictureNameCy3]
        macropinosomeCount = objectCount(currentImageCy3, thresholdCy3, macropinosomePixels)
        statsCy3Map[pictureNameCy3] = macropinosomeCount 
    return pairRatio({dapi:statsDAPIMap, cy3:statsCy3Map})        


# Calculates macropinocytosis index by dividing a particular image's macropinosome count by nuclei count and maps this 
# macropinocytotic index to its name. Also creates a global map mapForStatisticalAnalysis for statistical calculations
# passes this map into the makeAverages method
def pairRatio(statsMap):
    pairedMap = {}
    for key in statsMap:
        for imageName in statsMap[key].keys():
            pairedMap[imageName] = (((statsMap["Cy3"])[imageName])/((statsMap["DAPI"])[imageName]))
    global mapForStatisticalAnalysis
    mapForStatisticalAnalysis = pairedMap    
    return makeAverages(pairedMap)

# Averages all of the macropinocytotic indeces for one treatment group and calculates the standard deviation for each group
def makeAverages(pairedMap):
    averagesMap = {}
    usedKeys = []
    stDevMap = {}
    #Aggregates all the statistics of one treatment group 
    for imageName in pairedMap.keys():
    	#imageName is a tuple with the the first element in the tuple corresponding to the treatment group 
    	#which maps to its statistics. Interates over all possible treatment groups.
        if imageName[1] not in usedKeys:
            currentKey = imageName[1]
            listOfAreas = []
            #finds all other entries in the map in the current treatment group being averaged
            for key in pairedMap.keys():
                if currentKey == key[1]:
                    listOfAreas.append(pairedMap[key])
            usedKeys.append(imageName[1])
            averagesMap[imageName] = (np.mean(listOfAreas))
            stDevMap[imageName] = (np.std(listOfAreas)/np.sqrt(len(listOfAreas)))
    return graph((averagesMap, stDevMap))

#Takes in the averaged information and displays this information in bar graph format, segregated by treatment group,
#and includes error bars.
def graph(inputStats):
    treatmentGroups = inputStats[0].keys()
    usedTreatment = []
    #These two variable vary based on what coumpounds you use to treat the cells, and also what type of cells you are treating
    #Changing experimental conditions regarding these two fields will necessitate changing these two fields and the regex method 
    compounds = ['DMSO', "EIPA", "ARS853"]
    cellLines = ['H23', 'RIN1-1', 'RIN1-3', 'RIN2-1', 'RIN2-2', 'RIN2-3']
    compoundToStats = {}
    #this for loop algorithm orders the data into a list that seperates data of different compounds using a hashmap
    #then for each compound, makes a list whose order cooresponds to the order of each cellLine in the cellLines list for easy
    #graphing
    for i in range(0, len(compounds)):
        compound = compounds[i]
        values = []
        stDev = []
        for treatment in treatmentGroups:
            if compound in treatment[1]:
                for cellLine in cellLines:
                    if cellLine in treatment[1]: 
                        index = cellLines.index(cellLine)
                        values.insert(index, (inputStats[0][treatment]))
                        stDev.insert(index, (inputStats[1][treatment]))
        compoundToStats[compound] = [values, stDev]         
    numberCellLines = len(cellLines)
    ind = np.arange(numberCellLines)  # the x locations for the groups
    width = 0.25       # the width of the bars
    fig, ax = plt.subplots()
    rects = []
    for i in range(0, len(compounds)):
        rect = ax.bar(ind + width*i, compoundToStats[compounds[i]][0], width, color=random(), yerr=compoundToStats[compounds[i]][1])
        rects.append(rect)
    ax.set_ylabel('Level of Macropinocytosis')
    ax.set_title('Macropinocytosis Assay')
    ax.set_xticks(ind + width)
    xLabels = ()
    for line in cellLines:
        xLabels = xLabels + (line,)
    ax.set_xticklabels(xLabels)
    treatmentCompounds = ()
    for molecule in compounds:
        treatmentCompounds = treatmentCompounds + (molecule,)
    newTuple = ()
    for element in rects:
        newTuple = newTuple + (element[0],)
    ax.legend(newTuple, treatmentCompounds)
    plt.ion()
    plt.show()

#Method allows user to calculate statistical signficance between two treatment groups    
def statistics(group1, group2):

    twoConditions = {}
    group1Means = []
    group2Means = []
    for keys in mapForStatisticalAnalysis:
        if group1 == keys[1]:
            group1Means.append(mapForStatisticalAnalysis[keys][0])
        if group2 == keys[1]:
            group2Means.append(mapForStatisticalAnalysis[keys][0])       
    twoConditions[group1] = group1Means
    twoConditions[group2] = group2Means
    print('P value ', st.ttest_ind(group1Means, group2Means, axis=0, equal_var=False)[1])  

#Determines the number of objects of minimum size specified (for either nuclei or macropinosomes) in the input picture.
#The input picture is first thresholded 
def objectCount(picture, threshold, minimumSize):
    count = 0
    usedPixels = set()
    dimensions = np.shape(picture)
    rows = dimensions[0]
    columns = dimensions[1]
    for i in range(0, rows):
        for j in range(0, columns):
            if (i, j) not in usedPixels and picture[i][j] >= threshold:
                countedData = objectCountHelper((i,j), picture, threshold, minimumSize, 0, usedPixels, 0)
                usedPixels = countedData[0]
                if countedData[1] >= minimumSize:
                    count += 1                                                        
    return count

#Recursive helper function that identifies objects by scanning all positive pixels (pixels above threshold value) and finds all other positive pixels connnected to that
#pixel through positive pixels 
def objectCountHelper(currentPoint, picture, threshold, minimumSize, count, usedPixels, counter):
    usedPixels.add(currentPoint)
    interatablePixels = set()
    dimensions = np.shape(picture)
    rowMax = dimensions[0]
    columnMax = dimensions[1]
    for i in range(currentPoint[0] - 1, currentPoint[0] + 2):
        for j in range(currentPoint[1] - 1, currentPoint[1] + 2):
            if (i,j) not in usedPixels and inBounds(i, j, dimensions) and picture[i][j] >= threshold:
                temporaryPoint = (i, j)
                interatablePixels.add(temporaryPoint)                                 
    if len(interatablePixels) == 0:
        return [usedPixels, 1]
    else:
        for element in interatablePixels:
            if element not in usedPixels:
                currentCall = objectCountHelper(element, picture, threshold, minimumSize, count, usedPixels, counter)     
                usedPixels = currentCall[0]
                count += currentCall[1]
        return [usedPixels, count]    

#determines if pixel value is legal
def inBounds(i, j, dimensions):
    if i < dimensions[0] and i > -1:
        if j < dimensions[1] and j > -1:
            return True
    return False