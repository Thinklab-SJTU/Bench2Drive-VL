import base64
import requests
import os
from .prompt_utils import *

def local_image_to_base64(image_path):
    if image_path == "whatever":
        return None # just for format test
    try:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return None

        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")

        mime_type = image_path.split('.')[-1].lower()
        if mime_type not in ["jpg", "jpeg", "png", "gif", "bmp"]:
            raise ValueError(f"Unsupported image type: {mime_type}")

        return f"data:image/{mime_type};base64,{base64_string}"

    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        return None


def web_image_to_base64(image_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    response = requests.get(image_url, headers=headers)
    print(response)
    content_type = response.headers.get('Content-Type', '').lower()

    if 'image/jpeg' in content_type:
        mime_type = 'jpeg'
    elif 'image/png' in content_type:
        mime_type = 'png'
    elif 'image/gif' in content_type:
        mime_type = 'gif'
    elif 'image/bmp' in content_type:
        mime_type = 'bmp'
    else:
        raise ValueError(f"Unsupported image type: {content_type}")

    base64_string = base64.b64encode(response.content).decode('utf-8')
    return f"data:image/{mime_type};base64,{base64_string}"

def get_image_desc_from_vlm_api(image_url, vlm_url, vlm_str, VLM_TOKEN):
    from openai import OpenAI

    client = OpenAI(
        api_key=VLM_TOKEN,
        base_url=vlm_url
    )

    response = client.chat.completions.create(
        model=vlm_str,
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail."
                }
            ]
        }],
        max_tokens = 32768
    )

    desc = response.choices[0].message.content
    return desc

def get_image_descriptions(images_dict, image_frame_list, start_frame, end_frame,
                           frame_rate, template_func,
                           use_all_cameras, use_base64):
    """
    Returns content list about descriptions of all images in range (start_frame, end_frame]
    """
    content = []
    image_dirs = []
    is_placeholder = False
    if isinstance(template_func("whatever", False), str):
        is_placeholder = True
        content = ""
    for frame_number in image_frame_list:
        if start_frame < frame_number and frame_number <= end_frame:
            if frame_number in images_dict:
                # print(f"[debug] images_dict[frame_number]['CAM_FRONT'] = {images_dict[frame_number]['CAM_FRONT']}")
                if use_all_cameras:
                    if not is_placeholder:
                        # multi-view inference
                        content.append({
                            "type": "text",
                            "text": get_concat_img_desc(frame_number, frame_rate)
                        })
                    else:
                        content += get_concat_img_desc(frame_number, frame_rate)
                    for key in images_dict[frame_number].keys():
                        if key in ['CAM_FRONT_CONCAT', 'CAM_BACK_CONCAT']:
                            # content.append({
                            #     "type": "text",
                            #     "text": f"Image from {key}: "
                            # })
                            image_dirs.append(images_dict[frame_number][key])
                            if not is_placeholder:
                                content.append(template_func(images_dict[frame_number][key], use_base64))
                            else:
                                content += template_func(images_dict[frame_number][key], use_base64)
                else:
                    # single-view inference:
                    image_dirs.append(images_dict[frame_number]['CAM_FRONT'])
                    if not is_placeholder:
                        content.append({
                            "type": "text",
                            "text": get_front_img_desc(frame_number, frame_rate)
                        })
                        content.append(template_func(images_dict[frame_number]['CAM_FRONT'], use_base64))
                    else:
                        content += get_front_img_desc(frame_number, frame_rate)
                        content += template_func(images_dict[frame_number]['CAM_FRONT'], use_base64)

    return content, image_dirs

def get_carla_image_descriptions(images_dict, image_frame_list, start_frame, end_frame,
                                 frame_rate, template_func,
                                 use_all_cameras, use_bev, use_base64):
    """
    Returns content list about descriptions of all images in range (start_frame, end_frame]
    """
    content = []
    image_dirs = []
    is_placeholder = False
    if isinstance(template_func("whatever", False), str):
        is_placeholder = True
        content = ""
    for frame_number in image_frame_list:
        if start_frame < frame_number and frame_number <= end_frame:
            if frame_number in images_dict:
                # print(f"[debug] images_dict[frame_number]['ANNO_CAM_FRONT'] = {images_dict[frame_number]['ANNO_CAM_FRONT']}")
                if use_bev:
                    image_dirs.append(images_dict[frame_number]['ANNO_BEV'])
                    if not is_placeholder:
                        content.append({
                            "type": "text",
                            "text": get_bev_traj_img_desc(frame_number, frame_rate)
                        })
                        content.append(template_func(images_dict[frame_number]['ANNO_BEV'], use_base64))
                    else:
                        content += get_bev_traj_img_desc(frame_number, frame_rate)
                        content += template_func(images_dict[frame_number]['ANNO_BEV'], use_base64)
                else:
                    if use_all_cameras:
                        # multi-view inference
                        if not is_placeholder:
                            content.append({
                                "type": "text",
                                "text": get_concat_traj_img_desc(frame_number, frame_rate)
                            })
                        else:
                            content += get_concat_traj_img_desc(frame_number, frame_rate)
                        for key in images_dict[frame_number].keys():
                            if key in ['CAM_FRONT_CONCAT', 'CAM_BACK_CONCAT']:
                                # content.append({
                                #     "type": "text",
                                #     "text": f"Image from {key}: "
                                # })
                                image_dirs.append(images_dict[frame_number][key])
                                if not is_placeholder:
                                    content.append(template_func(images_dict[frame_number][key], use_base64))
                                else:
                                    content += template_func(images_dict[frame_number][key], use_base64)
                    else:
                        # single-view inference:
                        image_dirs.append(images_dict[frame_number]['ANNO_CAM_FRONT'])
                        if not is_placeholder:
                            content.append({
                                "type": "text",
                                "text": get_front_traj_img_desc(frame_number, frame_rate)
                            })
                            content.append(template_func(images_dict[frame_number]['ANNO_CAM_FRONT'], use_base64))
                        else:
                            content += get_front_traj_img_desc(frame_number, frame_rate)
                            content += template_func(images_dict[frame_number]['ANNO_CAM_FRONT'], use_base64)

    return content, image_dirs