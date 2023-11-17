import bpy, os, math, numpy, bmesh, random, mathutils
from . import utils


class OBJECT_OT_lr_pivot_painter_export(bpy.types.Operator):
    '''Select one or multiple parent objects, children objects are processed automatically.\n\nFRONT: X axis\nUP: Z axis\n\nOBJECTS NEED TO BE IN WORLD ZERO, then exported with generated UVs'''
    bl_idname = "object.lr_pivot_painter_export"
    bl_label = "Export textures for pivot painter 2.0"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context): 
        myprops = bpy.context.scene.pivot_painter_2

    
        if bpy.data.is_saved == False:
            self.report({'ERROR'}, 'Please save blender file. Aborting.')
            return {'FINISHED'}

        selected_obj = bpy.context.selected_objects

        object_list = []
        for obj in selected_obj:
            if obj.parent:
                continue
            object_list.append(obj)
            object_list.extend(obj.children_recursive)

        
        resolution = utils.find_texture_dimensions(len(object_list)) #Get image resolution based on number of objects



        # Create list of UVs
        resolution_x = resolution[0]
        resolution_y = resolution[1]

        np_uv = numpy.array([[0.0,0.0]]*(resolution_x*resolution_y))
        np_uv = np_uv.reshape(resolution_y,resolution_x,2)

        uv_u = 0.5
        uv_v = 0.5
        for V in range(0,len(np_uv)):
            uv_u_t = uv_u
            for U in range(0,len(np_uv[0])):
                np_uv[V][U][0]= uv_u_t
                np_uv[V][U][1]= uv_v
                uv_u_t +=1.0                
            
            uv_v -=1

        np_uv = np_uv.reshape(resolution_x*resolution_y,2)
        #Rescale
        for i in np_uv:
            i[0] = i[0]/resolution_x
            i[1] = ((i[1]-1)/resolution_y)+1

        uv_list = np_uv.tolist()
        # ----------------------------------


        #EDIT UVs ----------------------------

        uv_index = myprops.uv_coordinate-1 
        uv_pp_name = 'PPIndex'
        
        for index,obj in enumerate(object_list):
            bm_obj = bmesh.new()
            bm_obj.from_mesh(obj.data)

            while len(bm_obj.loops.layers.uv)-1 < uv_index-1:
                bm_obj.loops.layers.uv.new('UVMap')
            if uv_index > len(bm_obj.loops.layers.uv)-1:
                bm_obj.loops.layers.uv.new(uv_pp_name)

            uv = bm_obj.loops.layers.uv[uv_index]

            for face in bm_obj.faces:
                for loop in face.loops:
                    loop[uv].uv = uv_list[index][0],uv_list[index][1]
            
            bm_obj.to_mesh(obj.data) #Write bmesh


        def pixels_for_pivot_position_16_bit(objects):
            pixels= []            
            for i in range(0,len(objects)):
                pixels.append([objects[i].matrix_world[0][3]*100, (1-(objects[i].matrix_world[1][3]*100)), objects[i].matrix_world[2][3]*100])

            return pixels


        def pixels_for_vector_ld(objects, axis):
            '''
            Axis: 0 = X, 1 = Y, 2 = Z axis
            '''
            pixels = []

            for index,obj in enumerate(objects):
                matrix_copy = obj.matrix_world.normalized()
                #MAX HAS TRANSPOSED MATRIX
                matrix_copy_transposed = matrix_copy.transposed()
                matrix_copy_transposed.normalize()
                pixels.append(findConstantBiasScaleVectorValues(matrix_copy_transposed[axis]))
            return pixels


        def pixels_for_alpha_random_value_per_element(number_of_objects):
            rand_val = []
            for value in range(0,number_of_objects):
                rand_val.append(random.uniform(0,1))

            return rand_val

        #Alpa Stuff
        def clamp(num,cMin,cMax):
            result = num
            if result < cMin:
                result = cMin
            else:
                if result > cMax:
                    result = cMax
            return result


        def findMaxBoundingBoxDistanceAlongVector(objects,axis,ld = False):
            #Werified works
            maxdist_list = []
            for obj in objects:
                maxdist = obj.dimensions[axis]*100 #Convert to cm
                if ld == True:
                    maxdist = (clamp(math.ceil(maxdist/8.0),1.0,256.0) / 256.0) #compress to up to 2048
                maxdist_list.append(maxdist)

            return maxdist_list


        def pixels_for_alpha_find_parent_object_array_index(object_array): #Parent Index ( Float - Up to 2048 )
            array_index = []
            for obj in object_array:
                if obj.parent == None:
                    array_index.append(object_array.index(obj)+0.5)
                else:
                    array_index.append(object_array.index(obj.parent)+0.5)
            return array_index


        def find_number_of_steps_to_base_parent(child):
            count = 0
            obj = child
            while obj.parent:
                obj = obj.parent
                count +=1 
            return count
	
        def flatten_int_to_0to1_float(int_list):
            temp_list = []
            maxStepCount = max(int_list)
            for i in int_list:
                temp_list.append(float(i)/maxStepCount)
            return temp_list


        #Universal
        def constantBiasScaleScalar(my_scalar):
            return (my_scalar+1.0)/2.0


        def findConstantBiasScaleVectorValues(objectArray):
            normalizedI = objectArray
            return [constantBiasScaleScalar(normalizedI[0]),(1.0-(constantBiasScaleScalar(normalizedI[1]))),constantBiasScaleScalar(normalizedI[2])]


        #packTextureBits f16
        def packTextureBits(f16):
            f16= int(f16)
            f16+=1024
            sign = (f16 & 0x8000)<<16

            if (f16 & 0x7fff) == 0:
                expVar = 0
            else:
                expVar = ((((f16 >> 10)& 0x1f)-15+127)<< 23)
            
            mant =(f16 & 0x3ff)<< 13
            f16= (sign | expVar) | mant
            tmp=numpy.array(f16, dtype=numpy.int32)
            tmp.dtype = numpy.float32
            return tmp

        def packVectorIntsIntoFloats (objectArray):
            tArray=[]
            for i in objectArray:
                tArray.append([packTextureBits(i[1]),packTextureBits(i[2]),packTextureBits(i[3])])
            return tArray

        def pack_ints_into_floats(value_array):
            tArray=[]
            for value in value_array:
                tArray.append(packTextureBits(value))
            return tArray


        def create_image(rgb_name = None,alpha_name = None,image_format ='TARGA', pixels = None,alpha_values = None, resolution=None, hdr = False, is_data = False, alpha = True, image_location = None):
            ''' name: str
                image_format: TARGA,OPEN_EXR
                pixels: [[R,G,B,A],[R,G,B,A],[R,G,B,A]...]
                resolution: [x,y] optional else automatically generated

                returns = resolution [x,y]
            '''

            if pixels == None:
                return

            if alpha_values == None:
                for pixel in pixels:
                    pixel.append(0)
            else:
                
                if len(pixels) == len(alpha_values):
                    for pixel,alpha_value in zip(pixels,alpha_values):
                        pixel.append(alpha_value)
                else:
                    self.report({'INFO'}, 'Alpha values and Pixel values mismatch')

            if alpha_name == None:
                alpha_name = ''


            if image_location == None:
                image_location = bpy.path.abspath('//')

            elif image_location.startswith('//'):
                image_location = bpy.path.abspath(image_location)

            else:
                image_location = bpy.path.abspath(image_location)


            if bpy.data.is_saved == False:
                self.report({'ERROR'}, 'Please save blender file. Aborting.')
                return {'FINISHED'}    
            
            if image_format == 'TARGA':
                img_extension = '.TGA'

            elif image_format == 'OPEN_EXR':
                img_extension = '.EXR'


            if rgb_name is None:
                rgb_name = 'PivotPainterImage'




            #Empty Pixels value
            empty_pixel = [0,0,0,0]

            #Fill remaining pixels with black
            redundant_pixes_amount = resolution[0]*resolution[1]-len(pixels)
            for i in range(0,redundant_pixes_amount):
                pixels.append(empty_pixel)

            #Flip image vertically using numpy (blender by default starts in bottom left corner)
            np_array = numpy.array(pixels)
            np_array = np_array.reshape(resolution[1],resolution[0],4)
            np_array = numpy.flipud(np_array)
            np_array = np_array.flatten()
            pixels = np_array.tolist()

            #print(f'bpy.data.images.new(rgb_name, width=resolution[0], height=resolution[1], alpha = True, float_buffer={hdr}, is_data = {sRGB})')
            # Create blank Image
            image = bpy.data.images.new(rgb_name, width=resolution[0], height=resolution[1], alpha = True, float_buffer=hdr, is_data = is_data)

            # Assign pixels
            image.pixels = pixels


            # Write Image
            image.filepath_raw = os.path.join(image_location,rgb_name+'_'+alpha_name+img_extension)
            image.file_format = image_format
            image.save()

        
        #TEXTURES BASE NAME
        if not myprops.export_name:
            image_name_base = bpy.context.active_object.name
        else:
            image_name_base = myprops.export_name

        #ASSEMBLY
        image_name_prefix = 'T_'
        image_rgb_props = [myprops.image_1_rgb, myprops.image_2_rgb]
        image_alpha_props = [myprops.image_1_alpha, myprops.image_2_alpha]  
        
        # ---- IMAGE 1 RGB ----
        for prop_rgb,prop_alpha in zip(image_rgb_props,image_alpha_props):

            if prop_rgb != 'OP0': 
                if prop_rgb == 'OP1': #Pivot Position (16-bit)
                    img_pixels = pixels_for_pivot_position_16_bit(object_list)
                    img_hdr = True
                    img_is_data = True
                    img_format = 'OPEN_EXR'
                    img_name = f'{image_name_prefix}{image_name_base}_PivPos16bit'
    
                if prop_rgb == 'OP4': #X Vector (8-bit)
                    img_name = f'{image_name_prefix}{image_name_base}_XVector8bit'
                    img_pixels = pixels_for_vector_ld(object_list,0)
                    img_hdr = False
                    img_is_data = True
                    img_format = 'TARGA'

                if prop_rgb == 'OP5': #Y Vector (8-bit)
                    img_name = f'{image_name_prefix}{image_name_base}_YVector8bit'
                    img_pixels = pixels_for_vector_ld(object_list,1)
                    img_hdr = False
                    img_is_data = True
                    img_format = 'TARGA'

                if prop_rgb == 'OP6': #Z Vector (8-bit)
                    img_name = f'{image_name_prefix}{image_name_base}_ZVector8bit'
                    img_pixels = pixels_for_vector_ld(object_list,2)
                    img_hdr = False
                    img_is_data = True
                    img_format = 'TARGA'


                    #---- IMAGE 1 Alpha ----

                if prop_alpha == 'OP1': #Parent Index (Int as Float)
                    img_alpha_values=pack_ints_into_floats(pixels_for_alpha_find_parent_object_array_index(object_list))
                    img_alpha_name = 'ParentIndexInt_UV'+str(uv_index+1)

                if prop_alpha == 'OP3': #Random 0-1 Value Per Element
                    img_alpha_values = pixels_for_alpha_random_value_per_element(len(object_list))
                    img_alpha_name = 'Random0-1Value_UV'+str(uv_index+1)

                if prop_alpha == 'OP6': #Normalized 0-1 Hierarchy position
                    img_alpha_values = flatten_int_to_0to1_float([find_number_of_steps_to_base_parent(i) for i in object_list])
                    img_alpha_name = 'HierarchyPosition0-1_UV'+str(uv_index+1)
                    
                if prop_alpha == 'OP10': #Parent Index ( Float - Up to 2048 )
                    img_alpha_values = pixels_for_alpha_find_parent_object_array_index(object_list)
                    img_alpha_name = 'ParentIndexFloat_UV'+str(uv_index+1)

                if prop_alpha == 'OP11': #X Extent Divided by 2048 - 2048 Max
                    img_alpha_values = findMaxBoundingBoxDistanceAlongVector(object_list,0,True)
                    img_alpha_name = 'XExtentDividedby2048_UV'+str(uv_index+1)

                if prop_alpha == 'OP12': #Y Extent Divided by 2048 - 2048 Max
                    img_alpha_values = findMaxBoundingBoxDistanceAlongVector(object_list,1,True)
                    img_alpha_name = 'YExtentDividedby2048_UV'+str(uv_index+1)

                if prop_alpha == 'OP13': #Z Extent Divided by 2048 - 2048 Max
                    img_alpha_values = findMaxBoundingBoxDistanceAlongVector(object_list,2,True)
                    img_alpha_name = 'YExtentDividedby2048_UV'+str(uv_index+1)

                #print(f'create_image(rgb_name ={img_name},alpha_name= img_alpha_name, image_format = {img_format}, resolution={resolution}, pixels = {img_pixels}, alpha_values = img_alpha_values, hdr = {img_hdr}, sRGB = {img_sRGB},image_location=myprops.export_path)')
                create_image(rgb_name =img_name,alpha_name= img_alpha_name, image_format = img_format, resolution=resolution, pixels = img_pixels, alpha_values = img_alpha_values, hdr = img_hdr, is_data = img_is_data,image_location=myprops.export_path)
            


        self.report({'INFO'}, 'Done')


        return {'FINISHED'}




class OBJECT_OT_lr_attribute_increment_int_values(bpy.types.Operator):
    '''
    Multiple object selection. Active int attributes will be incremented on vertex domain per object. Decimal values stay unchanged. 
    Active object gets 0.
    '''
    bl_idname = "geometry.lr_set_per_obj_attribute"
    bl_label = "Increments int attribute on vertex domain"
    bl_options = {'REGISTER', 'UNDO'}

    # name: bpy.props.StringProperty(
    #     name="",
    #     description="Enter a string",
    #     default="Attribute",
    # )

    @classmethod
    def poll(cls, context): 
        return context.mode == 'OBJECT' or context.mode == 'EDIT_MESH'
        
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        selected_objects = bpy.context.selected_objects
        selected_objects.remove(bpy.context.active_object)
        selected_objects.insert(0,bpy.context.active_object)

        for index,obj in enumerate(selected_objects):
            for attribute in obj.data.attributes.active.data:
                attribute.value = attribute.value%1.0 + index

        return {'FINISHED'}


class OBJECT_OT_lr_rebuild(bpy.types.Operator):
    '''
    
    Breaks selected objects into subcomponent based on values in Elements attribute.
    
    Whole number = Subelement index, this list includes indexes mentioned below
    .1 = subelement pivot point.
    .2 = Subelement X axis.
    .3 = Subelement Y axis.
    ._01 = Second and third decimal is parent subelement index. If unspecified parent is index 0.
    '''


    bl_idname = "object.lr_rebuild"
    bl_label = "Breaks down mesh into subcomponents"
    bl_options = {'REGISTER', 'UNDO'}

    remove_extra: bpy.props.BoolProperty(
        name="Remove Extra",
        description="Remove vertices for pivot position. X axis and Y Axis",
        default=False,
    )


    # @classmethod
    # def poll(cls, context): 
    #     return context.mode == 'OBJECT' or context.mode == 'EDIT_MESH'
        
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

            


    def execute(self, context): 
        objs = bpy.context.selected_objects

        
        def parent_objects(child_obj, parent_obj):
            # Store the child object's world matrix
            child_world_matrix = child_obj.matrix_world.copy()

            # Set the child object's parent to the parent object
            child_obj.parent = parent_obj
            child_obj.matrix_world = child_world_matrix  # Restore the world matrix

        @staticmethod
        def vec_to_rotational_matrix(v1,v2,v3):
            """
            type = mathutils.Vector()
            v1,v2,v3 = Vectors for X,Y,Z axis. 
            """
            # Create the rotational matrix
            return Matrix((v1, v2, v3)).transposed()

        @staticmethod
        def gram_schmidt_orthogonalization(v1, v2, v3):
            # Normalize the vectors
            v1.normalize()
            v2.normalize()
            v3.normalize()
            
            # Create the orthogonal basis using Gram-Schmidt orthogonalization
            u1 = v1
            u2 = v2 - (v2.dot(u1) * u1)
            u2.normalize()
            u3 = v3 - (v3.dot(u1) * u1) - (v3.dot(u2) * u2)
            u3.normalize()
            
            orthogonal_vector_basis = (u1, u2, u3)
            return orthogonal_vector_basis

        @staticmethod
        def calculate_z_vector(v1, v2):
            # Calculate the cross product of v1 and v2
            z_vector = v1.cross(v2)
            z_vector.normalize()
            return z_vector

        @staticmethod
        def set_origin_rotation(obj, rotation_matrix_to):
            '''
            obj:
            object origin that is going to be rotated
            
            Rotational Matrix:
            Matrix without scale and translation influence (bpy.contextobject.matrix_world.to_3x3().normalized().to_4x4())
            
            Requires: mathutils.Matrix
            '''

            matrix_world = obj.matrix_world
            
            Rloc = matrix_world.to_3x3().normalized().to_4x4().inverted() @ rotation_matrix_to

            #Object rotation
            obj.matrix_world = (Matrix.Translation(matrix_world.translation) @ rotation_matrix_to @ Matrix.Diagonal(matrix_world.to_scale()).to_4x4())
            
            #Mesh rotation
            obj.data.transform(Rloc.inverted())

        @staticmethod
        def local_to_global_directional_vector(obj, local_vector):
            '''
            Translation of the object does not matter. Purely for rotation
            vector points one unit from object origin.
            
            '''
            # Ensure the object is valid and has a matrix
            if not isinstance(obj, bpy.types.Object) or not obj.matrix_world:
                raise ValueError("Invalid object or object has no world matrix")

            # Create a 4x4 matrix representing the object's world transformation
            world_matrix = obj.matrix_world

            # Convert the local vector to a 4D vector (homogeneous coordinates)
            local_vector_homogeneous = local_vector.to_4d()

            # Multiply the local vector by the object's world matrix to get the global vector
            global_vector_homogeneous = world_matrix @ local_vector_homogeneous

            # Convert the resulting 4D vector back to a 3D vector (removing homogeneous coordinate)
            global_vector = global_vector_homogeneous.to_3d()

            return global_vector

        @staticmethod
        def matrix_decompose(matrix_world):
            ''' 
            returns active_obj_mat_loc, active_obj_mat_rot, active_obj_mat_sca 
            reconstruct by loc @ rotQuat @ scale 
            '''
            
            loc, rotQuat, scale = matrix_world.decompose()

            active_obj_mat_loc = Matrix.Translation(loc)
            active_obj_mat_rot = rotQuat.to_matrix().to_4x4()
            active_obj_mat_sca = Matrix()
            for i in range(3):
                active_obj_mat_sca[i][i] = scale[i]

            return active_obj_mat_loc, active_obj_mat_rot, active_obj_mat_sca

        @staticmethod
        def move_origin_to_coord(obj,x,y,z):
            
            co_translation_vec = Vector((x,y,z))

            obj_translation_vec = obj.matrix_world.to_translation()
            obj_mat_loc, obj_mat_rot, obj_mat_sca = matrix_decompose(obj.matrix_world)
            
            mat_co = Matrix.Translation((x,y,z))


            new_mat = mat_co @ obj_mat_rot @ obj_mat_sca
            new_mat_mesh = new_mat.inverted() @ obj.matrix_world
            
            
            obj.matrix_world = new_mat

            is_object = True
            if bpy.context.object.mode !='OBJECT':
                is_object = False
                store_mode = bpy.context.object.mode
                bpy.context.object.mode = 'OBJECT'

            obj.data.transform(new_mat_mesh)

            if is_object == False:
                bpy.context.object.mode = store_mode
        @staticmethod
        def get_global_vertex_position(obj, vertex_index):
            """
            Get the global vertex position for a given object and vertex index.
            
            Parameters:
            obj (bpy.types.Object): The object containing the vertex.
            vertex_index (int): The index of the vertex.
            
            Returns:
            mathutils.Vector: The vertex position in global space.
            """
            if not obj or obj.type != 'MESH':
                # print("Invalid object or not a mesh.")
                return None
            
            # Get the mesh data of the object
            mesh = obj.data
            
            # Ensure the vertex index is valid
            if vertex_index < 0 or vertex_index >= len(mesh.vertices):
                # print("Invalid vertex index.")
                return None
            
            # Access the vertex's local coordinates
            local_vertex_co = mesh.vertices[vertex_index].co
            
            # Get the global coordinates of the vertex
            global_vertex_co = obj.matrix_world @ local_vertex_co
            
            return global_vertex_co
        
        @staticmethod
        def element_separate(obj,element_indexes,parent = None, origin_coords = None):
            '''
            Removes one element
            Parameters:
            obj (bpy.types.Object): The object containing the vertex.
            element_indexes [int,int...]: vertex index list that is going to be detached
            
            Returns:
            (bpy.types.Object): Detached mesh with only specified indexes.
            '''

            if obj.type == 'MESH':

                selected_obj = bpy.context.selected_objects
                active_obj = bpy.context.active_object
                bpy.ops.object.select_all(action='DESELECT')

                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)


                bpy.ops.object.duplicate()
                obj_separated = bpy.context.object

                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='DESELECT')
                bpy.ops.object.mode_set(mode='OBJECT')


                for vert in obj_separated.data.vertices:
                    if vert.index not in element_indexes:
                        vert.select = True

                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.delete(type='VERT')
                bpy.ops.object.mode_set(mode='OBJECT')            

                if origin_coords !=None:
                    move_origin_to_coord(obj_separated,
                                         origin_coords[0],
                                         origin_coords[1],
                                         origin_coords[2])

                #parent
                if parent !=None:
                    parent_objects(obj_separated,parent)

                


                #restore selection
                bpy.ops.object.select_all(action='DESELECT')
                for obj in selected_obj:
                    obj.select_set(True)
                bpy.context.view_layer.objects.active = active_obj
            
            return obj_separated



        for obj in objs:

            obj.data = obj.data.copy() #Make object unique. Remove instancing.
            
            # print(f'{obj.data.users = }')

            act_obj = bpy.context.active_object
            bpy.context.view_layer.objects.active = obj
            for modifier in obj.modifiers: # Apply all modifier
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.context.view_layer.objects.active = act_obj
            

            attr_name = 'Elements'

            if attr_name not in obj.data.attributes.keys():
                    message = f'Attribute {attr_name} not found on {obj.name}. Skipping.'
                    self.report({'INFO'}, message)
                    continue


            sub_elements_attr = None
            for attr in obj.data.attributes.values():
                if attr.name == attr_name:
                    sub_elements_attr = attr


            sub_elements_attr_data = sub_elements_attr.data.values()


            
            # 0  = Parent object, 
            # Whole number = subelement index, this list includes indexes mentioned below
            # .1 = subelement pivot point.
            # .2 = Subelement X axis.
            # .3 = Subelement Y axis.
            attr_info = {}
            sub_element_len = 0
            for index,data in enumerate(sub_elements_attr_data):
                i_val = round(data.value,3)
                # print(f'ATTRIBUTE DATA: \n{i_val}')
                i_val_int = int(i_val)

                if i_val_int not in attr_info:
                    attr_info[i_val_int] = {
                        'index': [],
                        'pivot_index': [],
                        'x_axis_index': [],
                        'y_axis_index': [],
                        'parent_element_id': None,
                        'object':None
                    }

                attr_info[i_val_int]['index'].append(index)

                if round(i_val%1,1) == 0.1:
                    sub_element_len += 1
                    attr_info[i_val_int]['pivot_index'].append(index) #Vertex index that belongs to Pivot. Find: vertex[index].co
                    attr_info[i_val_int]['parent_element_id'] = int(100*(round(i_val*10%1,2))) #This number points to a key in this dictionary. (Parent obj).
                if round(i_val%1,1) == 0.2:
                    attr_info[i_val_int]['x_axis_index'].append(index) #Vertex index where x axis points to.
                if round(i_val%1,1) == 0.3:
                    attr_info[i_val_int]['y_axis_index'].append(index) #Vertex index where Y axis points to.



            attr_info_ordered = OrderedDict(sorted(attr_info.items(), key=lambda x: x[0]))

            # print(f'{attr_info_ordered= }')

            pivot_position = []
            elements = []
            for idx in attr_info_ordered:

                #ORIGIN POSITION GET
                if attr_info_ordered[idx]['pivot_index']:
                    pivot_position = get_global_vertex_position(obj, attr_info_ordered[idx]['pivot_index'][0])
                else:
                    pivot_position = None




                #--- ORIGIN ROTATION ---
                
                #Get Directional Vector X
                if attr_info_ordered[idx]['x_axis_index'] != []:
                    origin_idx = attr_info_ordered[idx]['pivot_index'][0]
                    x_axis_idx = attr_info_ordered[idx]['x_axis_index'][0]
                    attr_info_ordered[idx]['x_axis_index']
                    directional_vector_x = (obj.matrix_world @ obj.data.vertices[x_axis_idx].co) - (obj.matrix_world @ obj.data.vertices[origin_idx].co) 
                else:
                    self.report({'ERROR'}, "Missing X axis. _.2")

                #Get Directional Vector Y
                if attr_info_ordered[idx]['y_axis_index'] != []:
                    origin_idx = attr_info_ordered[idx]['pivot_index'][0]
                    y_axis_idx = attr_info_ordered[idx]['y_axis_index'][0]
                    attr_info_ordered[idx]['x_axis_index']
                    directional_vector_y = (obj.matrix_world @ obj.data.vertices[y_axis_idx].co) - (obj.matrix_world @ obj.data.vertices[origin_idx].co)                 
                else:
                    self.report({'ERROR'}, "Missing X axis. _.3")

                #Get Directional Vector Z
                if attr_info_ordered[idx]['pivot_index'] != []:
                    directional_vector_z = obj.data.vertices[attr_info_ordered[idx]['pivot_index'][0]].normal @ obj.matrix_world.inverted()
                    directional_vector_z = directional_vector_z.normalized()
                else:
                    self.report({'ERROR'}, "Missing pivot position. Vert _.1")

                orthagonal_xyz_axis =gram_schmidt_orthogonalization(directional_vector_x, directional_vector_y, directional_vector_z)
                
                #Rotational matrix from orthagonal axis vectors
                rotational_matrix = vec_to_rotational_matrix(orthagonal_xyz_axis[0],orthagonal_xyz_axis[1],orthagonal_xyz_axis[2])

                if self.remove_extra: #Remove verticies which belong to pivot point x axis and y axis. 
                    attr_info[idx]['index'].remove(attr_info[idx]['x_axis_index'][0])
                    attr_info[idx]['index'].remove(attr_info[idx]['y_axis_index'][0])
                    attr_info[idx]['index'].remove(attr_info[idx]['pivot_index'][0])


                # ------ DUPLICATE ELEMENT INICIES AND SET ORIGIN POSITION ------
                element = element_separate(obj, attr_info[idx]['index'], parent =None, origin_coords = pivot_position)  #Add object information into ordered dictionary
                element.name = obj.name + '_part' + '_'+str(idx)
                attr_info_ordered[idx]['object'] = element #Assign detached object to dictionary

                # ------ ORIGIN ROTATION SET ------
                set_origin_rotation(element,rotational_matrix.to_4x4())



            # ------ SELECT MAKE ID0 ACTIVE AND PARENT  ------
            for idx in attr_info_ordered:
                    
                attr_info_ordered[idx]['object'].select_set(True)
                if idx == 0:
                    bpy.context.view_layer.objects.active = attr_info_ordered[idx]['object']

                if attr_info_ordered[idx]['parent_element_id'] != None: 

                    if attr_info_ordered[idx]['parent_element_id'] != idx:
                        parent_id = attr_info_ordered[idx]['parent_element_id']
                        parent_objects(attr_info_ordered[idx]['object'], attr_info_ordered[parent_id]['object']) 



            # for element in attr_info_ordered:
            #     print(f'ATTRIBUTE INFO #{element}: \n{attr_info_ordered[element]}')

            for col in obj.users_collection:
                col.objects.unlink(obj)

            bpy.data.objects.remove(obj)
            


        return {'FINISHED'}








