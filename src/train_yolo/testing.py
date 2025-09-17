from ultralytics import YOLO
from ultralytics.utils import LOGGER
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import LOGGER
class ModelFixer:
    """Fix model architecture to match expected number of classes"""
    
    @staticmethod
    def fix_detection_head(model, target_nc=3):
        """
        Fix the detection head to output correct number of classes
        
        Args:
            model: YOLO model
            target_nc: Target number of classes (3 for person, violence, weapon)
        """
        LOGGER.info(f"Fixing detection head for {target_nc} classes")
        
        # Get the detection head (last layer)
        detection_head = model.model.model[-1]
        
        if not isinstance(detection_head, Detect):
            raise ValueError(f"Expected Detect head, got {type(detection_head)}")
        
        # Store original parameters
        original_nc = detection_head.nc
        LOGGER.info(f"Original nc: {original_nc}, Target nc: {target_nc}")
        
        if original_nc == target_nc:
            LOGGER.info("Model already has correct number of classes")
            return model
        
        # Update nc
        detection_head.nc = target_nc
        
        # Get the channel dimensions for cv2 and cv3
        # cv2 is for bounding box regression (4 coords * na)  
        # cv3 is for classification (nc * na)
        na = detection_head.na  # number of anchors
        
        LOGGER.info(f"Number of anchors (na): {na}")
        LOGGER.info(f"Updating cv3 layers for {target_nc} classes")
        
        # Reinitialize cv3 layers (classification)
        new_cv3 = nn.ModuleList()
        
        for i, old_cv3 in enumerate(detection_head.cv3):
            # Get the input channels from the previous layer
            if isinstance(old_cv3, nn.Sequential):
                # Find the last layer to get input channels
                for layer in reversed(old_cv3):
                    if hasattr(layer, 'in_channels'):
                        in_channels = layer.in_channels
                        break
                else:
                    # Fallback: use the input channels of the first Conv2d layer
                    in_channels = old_cv3[0].conv.in_channels if hasattr(old_cv3[0], 'conv') else 64
                
                # Create new sequential block with correct output channels
                new_sequential = nn.Sequential()
                
                # Copy all layers except the last one
                for j, layer in enumerate(old_cv3[:-1]):
                    new_sequential.add_module(str(j), layer)
                
                # Add new final layer with correct output channels
                final_layer = nn.Conv2d(in_channels, target_nc * na, kernel_size=1, stride=1)
                new_sequential.add_module(str(len(old_cv3) - 1), final_layer)
                
                new_cv3.append(new_sequential)
                
                LOGGER.info(f"cv3[{i}]: Updated final layer to output {target_nc * na} channels")
        
        # Replace cv3
        detection_head.cv3 = new_cv3
        
        # Update other head parameters
        detection_head.ch = target_nc + 5  # classes + 5 (4 bbox + 1 obj)
        
        LOGGER.info("✅ Detection head fixed successfully")
        
        # Verify the fix
        ModelFixer.verify_fix(model, target_nc)
        
        return model
    
    @staticmethod
    def verify_fix(model, expected_nc):
        """Verify that the fix worked"""
        LOGGER.info("Verifying model fix...")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 416, 416)
        
        model.model.eval()
        with torch.no_grad():
            output = model.model(dummy_input)
        
        if isinstance(output, (list, tuple)) and len(output) > 0:
            pred_shape = output[0].shape
            LOGGER.info(f"Output shape after fix: {pred_shape}")
            
            # Calculate expected channels: (4 bbox + 1 obj + nc classes) * na
            detection_head = model.model.model[-1]
            na = detection_head.na
            expected_channels = (4 + 1 + expected_nc) * na
            
            if pred_shape[1] == expected_channels:
                LOGGER.info(f"✅ Fix successful! Output has {expected_channels} channels as expected")
                return True
            else:
                LOGGER.error(f"❌ Fix failed! Expected {expected_channels} channels, got {pred_shape[1]}")
                return False
        else:
            LOGGER.error("❌ Unexpected output format")
            return False

    def apply_model_fix(model_path, target_nc=3):
        LOGGER.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        fixed_model = ModelFixer.fix_detection_head(model, target_nc)
        return fixed_model

    if __name__ == "__main__":
        model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\detect\train33\weights\best.pt"
        fixed_model = apply_model_fix(model_path, target_nc=3)
        fixed_model.save(r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\detect\train33\weights\best_fixed.pt")
        LOGGER.info("Fixed model saved as best_fixed.pt")