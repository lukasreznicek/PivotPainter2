import bpy, math


def get_closest_sq_resolution(pixels):
    '''Returns: resolution list, redundant pixels'''
    pixels_squared = math.sqrt(pixels)

    def next_power_of_2(x):
        return 1 if x == 0 else 2**math.ceil(math.log2(x))

    horizontal_resolution = next_power_of_2(pixels_squared)
    vertical_resolution_iter = horizontal_resolution


    while vertical_resolution_iter*horizontal_resolution >= pixels:
        vertical_resolution_iter -= 1

    return [horizontal_resolution,vertical_resolution_iter+1]





def get_closest_resolution(pixels):
    '''
    Input: int numner of pixels
    Returns: [horizontal,vertical]
    
    '''
    pixels_squared = math.sqrt(pixels)

    horizontal_resolution = math.ceil(pixels_squared)
    vertical_resolution_iter = horizontal_resolution

    while vertical_resolution_iter*horizontal_resolution >= pixels:
        vertical_resolution_iter -= 1


    return [horizontal_resolution,vertical_resolution_iter+1]


def get_closest_resolution_simple(pixels):
    '''Returns: resolution list'''
    return [ceil(pixels/2),2]

def find_texture_dimensions(ObjectToProcessCount):
    '''Original from 3ds Max Pivot Painter 2.0'''
    DecrementerTotal = 1600 #small enough to avoid uv precision issues without using high precision values
    evenNumber = (ObjectToProcessCount%2.0)==0
    HalfEvenNumber = ((ObjectToProcessCount/2.0)%2.0)==0
    HalfNumber = math.ceil(ObjectToProcessCount/2.0)
    modResult = 1
    RowCounter = 1
    newDecrementerTotal = HalfNumber if HalfNumber < DecrementerTotal else DecrementerTotal
    decrementAmount =  2 if HalfEvenNumber == True else 1
    complete = False

    while complete == False:
        modResult = ObjectToProcessCount%newDecrementerTotal
        complete = modResult == 0 or newDecrementerTotal < 1 
        
        if complete== False:
            newDecrementerTotal-=decrementAmount 
        if newDecrementerTotal < 1:
            newDecrementerTotal=1
    
    if newDecrementerTotal==1 or ((ObjectToProcessCount/newDecrementerTotal)>DecrementerTotal):
            y=math.floor(math.sqrt(ObjectToProcessCount))
            x=math.ceil((ObjectToProcessCount/math.floor(y)))
            return [int(x),int(y)]
    else:
        return [int(newDecrementerTotal),int((ObjectToProcessCount/newDecrementerTotal))]


