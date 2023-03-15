# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
import bpy

bl_info = {
    "name" : "PivotPainter2",
    "author" : "Lukas Reznicek, Jonathan Lindquist",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}



from .operators.operators import OBJECT_OT_lr_pivot_painter_export
from bpy.props import IntProperty, CollectionProperty, StringProperty,FloatVectorProperty,BoolProperty,EnumProperty


def img1_alpha_callback(scene, context):
    painter2 = bpy.context.scene.pivot_painter_2

    items = []
    if painter2.image_1_rgb == 'OP0':
        items = []
    elif painter2.image_1_rgb == 'OP1' or painter2.image_1_rgb == 'OP2' or painter2.image_1_rgb == 'OP3': #LDR
        items.append(('OP1', 'Parent Index (Int as Float)',''))
        # items.append(('OP2', 'Number of Steps From Root',''))
        items.append(('OP3', 'Random 0-1 Value Per Element',''))
        # items.append(('OP4', 'Bounding Box Diameter',''))
        # items.append(('OP5', 'Selection Order (Int as Float)',''))
        items.append(('OP6', 'Normalized 0-1 Hierarchy position',''))
        # items.append(('OP7', 'Object X Width',''))
        # items.append(('OP8', 'Object Y Depth',''))
        # items.append(('OP9', 'Object Z Height',''))
        items.append(('OP10', 'Parent Index (Float - Up to 2048)',''))

    elif painter2.image_1_rgb == 'OP4' or painter2.image_1_rgb == 'OP5' or painter2.image_1_rgb == 'OP6': #HDR
        items.append(('OP6', 'Normalized 0-1 Hierarchy position',''))
        items.append(('OP3', 'Random 0-1 Value Per Element',''))
        items.append(('OP11', 'X Extent Divided by 2048 - 2048 Max',''))
        items.append(('OP12', 'Y Extent Divided by 2048 - 2048 Max',''))
        items.append(('OP13', 'Z Extent Divided by 2048 - 2048 Max',''))
    return items


def img2_alpha_callback(scene, context):
    painter2 = bpy.context.scene.pivot_painter_2

    items = []
    if painter2.image_2_rgb == 'OP0':
        items = []
    elif painter2.image_2_rgb == 'OP1' or painter2.image_2_rgb == 'OP2' or painter2.image_2_rgb == 'OP3': #LDR
        items.append(('OP1', 'Parent Index (Int as Float)',''))
        # items.append(('OP2', 'Number of Steps From Root',''))
        items.append(('OP3', 'Random 0-1 Value Per Element',''))
        # items.append(('OP4', 'Bounding Box Diameter',''))
        # items.append(('OP5', 'Selection Order (Int as Float)',''))
        items.append(('OP6', 'Normalized 0-1 Hierarchy position',''))
        # items.append(('OP7', 'Object X Width',''))
        # items.append(('OP8', 'Object Y Depth',''))
        # items.append(('OP9', 'Object Z Height',''))
        items.append(('OP10', 'Parent Index (Float - Up to 2048)',''))

    elif painter2.image_2_rgb == 'OP4' or painter2.image_2_rgb == 'OP5' or painter2.image_2_rgb == 'OP6': #HDR
        items.append(('OP11', 'X Extent Divided by 2048 - 2048 Max',''))
        items.append(('OP12', 'Y Extent Divided by 2048 - 2048 Max',''))
        items.append(('OP13', 'Z Extent Divided by 2048 - 2048 Max',''))
        items.append(('OP6', 'Normalized 0-1 Hierarchy position',''))
        items.append(('OP3', 'Random 0-1 Value Per Element',''))

    return items

# Properties 
# To acess properties: bpy.data.scenes['Scene'].pivot_painter_2
# Is assigned by pointer property below in class registration.
class pivot_painter2_settings(bpy.types.PropertyGroup):


    image_1_rgb:bpy.props.EnumProperty(name= 'RGB', description= '',default = 1, items= [
    ('OP0', 'Do Not Render',''),
    ('OP1', 'Pivot Position (16-bit)',''),
    #('OP2', 'Origin Position(16-bit)',''),
    #('OP3', 'Origin Extents(16-bit)',''),
    ('OP4', 'X Vector(8-bit)',''),
    ('OP5', 'Y Vector(8-bit)',''),
    ('OP6', 'Z Vector(8-bit)','')])
    image_1_alpha:bpy.props.EnumProperty(name= 'Alpha', description= '', items= img1_alpha_callback)
 
    image_2_rgb:bpy.props.EnumProperty(name= 'RGB', description= '',default = 2, items= [
    ('OP0', 'Do Not Render',''),
    ('OP1', 'Pivot Position (16-bit)',''),
    #('OP2', 'Origin Position(16-bit)',''),
    #('OP3', 'Origin Extents(16-bit)',''),
    ('OP4', 'X Vector(8-bit)',''),
    ('OP5', 'Y Vector(8-bit)',''),
    ('OP6', 'Z Vector(8-bit)','')])


    image_2_alpha:bpy.props.EnumProperty(name= 'Alpha', description= '', items= img2_alpha_callback)

    export_path:bpy.props.StringProperty(name="Folder", description="Texture output location. \n// = .blend file location\n//..\ = .blend file parent folder", default="//", maxlen=1024,subtype='DIR_PATH')
    uv_coordinate: bpy.props.IntProperty(name="Texture Coordinate", description="Location of Pivot Painter custom UVs. Starts with 1", default=2, min = 1, soft_max = 5)
    



class VIEW3D_PT_pivot_painter2(bpy.types.Panel):
    bl_label = "Pivot Painter 2"
    bl_idname = "OBJECT_PT_pivot_painter2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PivotPainter2'


    def draw(self, context):

        pivot_painter_2 = context.scene.pivot_painter_2
        #myprops = bpy.context.scene['pivot_painter_2']

        layout = self.layout.box()
        layout.label(text="UVs")

        row = layout.row(align=True)
        row.prop(pivot_painter_2, "uv_coordinate")



        #IMAGE 1 ----
        layout = self.layout.box()
        layout.label(text="Image 1")

        row = layout.row(align=True)
        row.prop(pivot_painter_2, "image_1_rgb")
        row = layout.row(align=True)
        if pivot_painter_2.image_1_rgb != 'OP0': 
            row.prop(pivot_painter_2, "image_1_alpha")


        #IMAGE 2 ----
        layout = self.layout.box()
        layout.label(text="Image 2")

        row = layout.row(align=True)
        row.prop(pivot_painter_2, "image_2_rgb")
        row = layout.row(align=True)
        if pivot_painter_2.image_2_rgb != 'OP0': 
            row.prop(pivot_painter_2, "image_2_alpha")





        layout = self.layout.box()
        row = layout.row(align=True)
        row.prop(pivot_painter_2, "export_path")
        row = layout.row(align=True)
        row.scale_y = 2
        row.operator("object.lr_pivot_painter_export", text="Process Hierarchy", icon = 'EXPORT')


classes = [pivot_painter2_settings,VIEW3D_PT_pivot_painter2, OBJECT_OT_lr_pivot_painter_export]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pivot_painter_2 = bpy.props.PointerProperty(type=pivot_painter2_settings)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.pivot_painter_2
