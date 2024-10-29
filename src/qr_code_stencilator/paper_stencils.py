import numpy as np
import cv2
from dotmap import DotMap
import toml

INCH_TO_CM = 2.54

def merge_data(base: dict,donor: dict):
    """Merges two dictionaries by replacing all overlapping keys with donor values and adding all unique donor key value pairs.
    This function runs inplace and modifies the base dictionary object.

    Args:
        base (dict): _description_
        donor (dict): _description_
    """

    for k, v in donor.items():
        if v is dict:
            merge_data(base[k],v)
        else:
            base[k] = v


class PaperStencilGeneratorPreset(DotMap):

    @classmethod
    def convert_preset_to_metric(cls, preset_data:dict[str,any]):
        """Converts all parameters that are in inches to centimeters

        Args:
            preset_data (dict[str,any]): The dict with preset data to convert
        """

        excluded_keys = set([
            'dpi',
            'paper_formats'
        ])

        for k, v in preset_data.items():
            if v is dict and k not in excluded_keys:
                PaperStencilGeneratorPreset.convert_preset_to_metric(preset_data[k])
            elif (v is float or v is int) and k not in excluded_keys:
                preset_data[k] *= INCH_TO_CM

    @classmethod
    def load_from_toml_multi(cls, presets_toml:list[str]):
        presets = [DotMap(toml.loads(p)) for p in presets_toml]
        imperial_mode = presets[0].imperial_mode
        final_data = presets[0]
        for preset_data in presets: 
            if preset_data.imperial_mode is bool:
                imperial_mode = preset_data.imperial_mode
            if imperial_mode:
                PaperStencilGeneratorPreset.convert_preset_to_metric(preset_data)
            merge_data(final_data,preset_data)
               
        return PaperStencilGeneratorPreset(final_data)
    
def resize_stencil(image: np.ndarray, target_width: int) -> np.ndarray:
    # Resize the image using nearest neighbor interpolation to avoid smoothing
    resized_image = cv2.resize(image, (target_width, target_width), interpolation=cv2.INTER_NEAREST)
    
    return resized_image

class PaperStencilGenerator:
    preset: PaperStencilGeneratorPreset

    def __init__(self, preset: PaperStencilGeneratorPreset, paper_formats: DotMap|dict):
        self.preset = preset
        paper_formats = DotMap(paper_formats).paper_formats
        merge_data(paper_formats, preset.paper_formats)

        paper_format = preset.paper.format

        if paper_format in paper_formats.metric.keys():
            paper_size_cm = paper_formats.metric[paper_format]
        elif paper_format in paper_formats.imperial.keys():
            paper_size_cm = paper_formats.metric[paper_format]*INCH_TO_CM
        else:
            raise KeyError(f"ERROR: Could not find paper format '{paper_format}'")
        
        dpi = preset.printing.dpi
        paper_size_px = [round(x*(dpi/INCH_TO_CM)) for x in paper_size_cm]

        using_imperial = preset.imperial_mode

        print_margin_cm = preset.printing.margins
        print_margin_px = round(paper_size_px[0]*(print_margin_cm/paper_size_cm[0]))

        print_stroke_width_cm = preset.printing.stroke_width
        print_stroke_width_px = round(paper_size_px[0]*(print_stroke_width_cm/paper_size_cm[0]))

        design_fill_pattern_gaps_cm = preset.design.diagonal_lines_spacing
        design_fill_pattern_gaps_px = round(paper_size_px[0]*(design_fill_pattern_gaps_cm/paper_size_cm[0]))

        print_target_qr_code_width_px = paper_size_px[0]-2*print_margin_px

        design_fill_pattern_mode = preset.design.diagonal_line_fill_mode

        self.paper_size_px = paper_size_px
        
        self.print_margin_px = print_margin_px
        self.print_stroke_width_px = print_stroke_width_px
        self.print_target_qr_code_width_px = print_target_qr_code_width_px
        
        self.design_fill_pattern_mode = design_fill_pattern_mode
        self.design_fill_pattern_gaps_px = design_fill_pattern_gaps_px

    def generate_stencil(self, raw_stencil: np.ndarray, index: int) -> np.ndarray:
        paper_stencil = np.ones(self.paper_size_px[::-1])
        upscaled_stencil = resize_stencil(raw_stencil,self.print_target_qr_code_width_px)

        stroke_width = self.print_stroke_width_px+(self.print_stroke_width_px%2)
        kernel = np.ones((stroke_width//2,stroke_width//2), np.uint8) 
        stencil_edges = np.minimum(cv2.dilate(cv2.Canny(upscaled_stencil.astype(np.uint8),0.5,1),kernel),1)


        py = (self.paper_size_px[1]//2)-(self.print_target_qr_code_width_px//2)
        center_rect = (
            slice(py, py+self.print_target_qr_code_width_px),
            slice(self.print_margin_px, -self.print_margin_px),
        )
        
        if self.design_fill_pattern_mode != 'DISABLED':
            fill_pattern_mask = (upscaled_stencil < 1) if self.design_fill_pattern_mode == 'OUTSIDE' else (upscaled_stencil > 0)

            w,h = self.paper_size_px
            pixel_coords = np.mgrid[0:h, 0:w]
            fill_pattern_lines = ((pixel_coords[0] + pixel_coords[1]) % self.design_fill_pattern_gaps_px)
            fill_pattern_lines = (fill_pattern_lines < np.sqrt(stroke_width**2 / 2))

            paper_stencil[center_rect][fill_pattern_lines[center_rect] & fill_pattern_mask] = 0

        paper_stencil[center_rect][stencil_edges>0] = 0

        for i in range(index+1):
            try:
                paper_stencil = cv2.circle(paper_stencil,(round(self.print_margin_px+stroke_width+2.5*stroke_width*i),round(py-2.1*stroke_width)),stroke_width//2,0,stroke_width)
            except:
                pass

        return paper_stencil
    