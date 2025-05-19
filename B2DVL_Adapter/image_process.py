import re
from PIL import Image, ImageDraw, ImageFont
import os
from io_utils import load_json_gz
from bev_renderer import generate_anno_rgb_image
import cv2

TMP_IMAGE_PATH = './tmp_images'

def get_contrast_color(rgb_color):
    r, g, b = rgb_color
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.5 else "black"

def parse_label(input_string):
    outer_pattern = r'\(\s*<([^<>]*(?:<[^<>]*>)*[^<>]*)>\s*\)'
    outer_matches = re.finditer(outer_pattern, input_string)
    
    result = {}

    for outer_match in outer_matches:
        outer_content = f"(<{outer_match.group(1)}>)"
        id_match = re.match(r"\(<\s*([^<>\s]+)", outer_content)
        if not id_match:
            continue
        obj_id = id_match.group(1)
        
        result[obj_id] = {}
        result[obj_id]['original_str'] = outer_content
        
        cam_pattern = r"<(\w+),([\d.]+),([\d.]+)>"
        cam_matches = re.finditer(cam_pattern, outer_content)
        
        for cam_match in cam_matches:
            cam_name = cam_match.group(1)
            coords = [float(cam_match.group(2)), float(cam_match.group(3))]
            result[obj_id][cam_name] = coords

    return result

def generate_concat_camera_images(camera_images):
    front_left_img_path = camera_images.get("CAM_FRONT_LEFT")
    front_img_path = camera_images.get("CAM_FRONT")
    front_right_img_path = camera_images.get("CAM_FRONT_RIGHT")
    back_right_img_path = camera_images.get("CAM_BACK_RIGHT")
    back_img_path = camera_images.get("CAM_BACK")
    back_left_img_path = camera_images.get("CAM_BACK_LEFT")

    try:
        front_left_img = Image.open(front_left_img_path)
        front_img = Image.open(front_img_path)
        front_right_img = Image.open(front_right_img_path)
        back_right_img = Image.open(back_right_img_path)
        back_img = Image.open(back_img_path)
        back_left_img = Image.open(back_left_img_path)

        front_concat = Image.new("RGB", (front_left_img.width + front_img.width + front_right_img.width, front_left_img.height))
        front_concat.paste(front_left_img, (0, 0))
        front_concat.paste(front_img, (front_left_img.width, 0))
        front_concat.paste(front_right_img, (front_left_img.width + front_img.width, 0))

        back_concat = Image.new("RGB", (back_right_img.width + back_img.width + back_left_img.width, back_right_img.height))
        back_concat.paste(back_right_img, (0, 0))
        back_concat.paste(back_img, (back_right_img.width, 0))
        back_concat.paste(back_left_img, (back_right_img.width + back_img.width, 0))

        return front_concat, back_concat

    except Exception as e:
        print(f"Error processing images: {e}")
        return None, None

def concatenate_camera_images(camera_images, frame_number, worker_id, save_path, save_name):
    front_concat, back_concat = generate_concat_camera_images(camera_images)
    front_concat_path = None
    back_concat_path = None

    if front_concat is not None:
        os.makedirs(os.path.join(save_path, save_name, f"concat_front"), exist_ok=True)
        front_concat_path = os.path.join(save_path, save_name, "concat_front", f"{frame_number:05d}.jpg")
        front_concat.save(front_concat_path)
    
    if back_concat is not None:
        os.makedirs(os.path.join(save_path, save_name, f"concat_back"), exist_ok=True)
        back_concat_path = os.path.join(save_path, save_name, "concat_back", f"{frame_number:05d}.jpg")
        back_concat.save(back_concat_path)
    
    return front_concat_path, back_concat_path

def generate_anno_img(img_path, objects_dict, max_size, font_size, cam_name):
    img_resized = None
    try:
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = img.resize((new_width, new_height))

        draw = ImageDraw.Draw(img_resized)
        try:
            font = ImageFont.truetype("SarasaMonoSC-Regular.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        text_width = draw.textlength(cam_name, font=font)
        text_x = (new_width - text_width) // 2
        text_y = 10
        draw.text((text_x, text_y), cam_name, fill="white", font=font)

        for obj_id, cam_data in objects_dict.items():
            if cam_name in cam_data:
                coords = cam_data[cam_name]
                x, y = int(coords[0] * scale), int(coords[1] * scale)
                pixel_color = (0, 0, 0)
                try:
                    pixel_color = img_resized.getpixel((x, y))
                except Exception as e:
                    pass
                contrast_color = get_contrast_color(pixel_color)

                draw.ellipse(
                    [(x - 5, y - 5), (x + 5, y + 5)],
                    fill=contrast_color,
                    outline=contrast_color,
                )
                draw.text((x + 10, y - 10), obj_id, fill=contrast_color, font=font)
    except Exception as e:
        # maybe index of of range, but it's ok
        print(f"Error processing image {img_path}: {e}") #
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = img.resize((new_width, new_height))

    return img_resized

def annotate_images(objects_dict, camera_images, frame_number, worker_id, save_path, save_name, history_dict, anno_data, max_size=800, font_size=20):
    os.makedirs(os.path.join(save_path, save_name), exist_ok=True)
    annotated_images = {}

    for cam_name, img_path in camera_images.items():
        annotated_images[cam_name] = img_path
        if cam_name.startswith('CAM'):
            # img_resized = generate_anno_img(img_path, objects_dict, max_size, font_size, cam_name)
            img_resized = generate_anno_rgb_image(anno_data=anno_data,
                                                  sensor_key=cam_name,
                                                  raw_img=cv2.imread(img_path),
                                                  history_dict=history_dict,
                                                  frame=frame_number,
                                                  history_count=5)
            os.makedirs(os.path.join(save_path, save_name, f"{cam_name}_anno"), exist_ok=True)
            output_path = os.path.join(save_path, save_name, f"{cam_name}_anno", f"{frame_number:05d}.jpg")
            cv2.imwrite(str(output_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 20])
            annotated_images[cam_name] = output_path

    return annotated_images

def process_bubble_image(bubble, worker_id, save_name, history_dict, anno_file):
    TMP_PATH = TMP_IMAGE_PATH
    label_dict = parse_label(bubble.words)
    anno_data = load_json_gz(anno_file)
    
    new_images = []
    for image_dict in bubble.images:
        new_dict = annotate_images(label_dict, image_dict, bubble.frame_number, worker_id, TMP_PATH, save_name, history_dict, anno_data)
        front_concat_path, back_concat_path = concatenate_camera_images(new_dict, bubble.frame_number, worker_id, TMP_PATH, save_name)
        new_dict['CAM_FRONT_CONCAT'] = front_concat_path
        new_dict['CAM_BACK_CONCAT'] = back_concat_path
        new_images.append(new_dict)
    bubble.images = new_images

    new_extra_images = []
    for image_dict in bubble.extra_images:
        new_dict = annotate_images(label_dict, image_dict, bubble.frame_number, worker_id, TMP_PATH, save_name, history_dict, anno_data)
        front_concat_path, back_concat_path = concatenate_camera_images(new_dict, bubble.frame_number, worker_id, TMP_PATH, save_name)
        new_dict['CAM_FRONT_CONCAT'] = front_concat_path
        new_dict['CAM_BACK_CONCAT'] = back_concat_path
        new_extra_images.append(new_dict)
    bubble.extra_images = new_extra_images

    for key in label_dict.keys():
        new_str = f"({key}"
        for cam in label_dict[key].keys():
            if cam.startswith('CAM'):
                new_str += f", {cam}"
        new_str += ")"
        bubble.words = bubble.words.replace(label_dict[key]['original_str'], new_str)

    return bubble


