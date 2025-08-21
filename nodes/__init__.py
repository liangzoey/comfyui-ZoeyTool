import os
import sys
import logging

# ���ò��Ŀ¼·��
plugin_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, plugin_dir)

# ������־
logger = logging.getLogger("ZoeyTool")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(f"Loading Zoey Tool plugin from: {plugin_dir}")
logger.info(f"Plugin directory contents: {os.listdir(plugin_dir)}")

try:
    # ��ʽ��������ģ��
    from batch_image_cropper import NODE_CLASS_MAPPINGS as CROPPER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CROPPER_DISPLAY
    from zoey_tool import NODE_CLASS_MAPPINGS as ZOEY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ZOEY_DISPLAY
    from multifunctional_image_editor import NODE_CLASS_MAPPINGS as EDITOR_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as EDITOR_DISPLAY
    from image_edit_prompt_generator import NODE_CLASS_MAPPINGS as PROMPT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PROMPT_DISPLAY
    
    # ���Ե���mask_draw_rectangle
    try:
        from mask_draw_rectangle import NODE_CLASS_MAPPINGS as MASK_DRAW_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MASK_DRAW_DISPLAY
        logger.info("Successfully imported mask_draw_rectangle module")
    except ImportError as e:
        logger.error(f"Failed to import mask_draw_rectangle: {str(e)}")
        # ���˷����������ӳ��
        MASK_DRAW_MAPPINGS = {}
        MASK_DRAW_DISPLAY = {}
    
    # ���µ����ֻ��ƽڵ�ӳ����ӵ�ZOEY_MAPPINGS��
    ZOEY_MAPPINGS.update(MASK_DRAW_MAPPINGS)
    ZOEY_DISPLAY.update(MASK_DRAW_DISPLAY)
    
    # �ϲ�����ӳ��
    NODE_CLASS_MAPPINGS = {
        **CROPPER_MAPPINGS,
        **ZOEY_MAPPINGS,
        **EDITOR_MAPPINGS,
        **PROMPT_MAPPINGS
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        **CROPPER_DISPLAY,
        **ZOEY_DISPLAY,
        **EDITOR_DISPLAY,
        **PROMPT_DISPLAY
    }
    
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    
    logger.info("Zoey Tool plugin loaded successfully")

except Exception as e:
    logger.exception(f"Critical error loading Zoey Tool plugin: {str(e)}")
    # ȷ����ʹ����ʧ��Ҳ�ṩ��ӳ��
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']